import pdb
from functools import reduce
from typing import List, Optional, Union

import yaml
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, autocast
from torch.distributions.distribution import Distribution

from mld.models.architectures.tools.embeddings import TimestepEmbedding, Timesteps
from mld.models.operator import PositionalEncoding
from mld.models.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask

from model.vq.bottleneck import Bottleneck, NoBottleneck


"""
vae

skip connection encoder 
skip connection decoder

mem for each decoder layer
"""

class AutoMldVae(nn.Module):
    def __init__(self,
                 nfeats: int,
                 latent_dim: tuple = [1, 256],
                 h_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 arch: str = "all_encoder",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 position_embedding: str = "learned",
                 rep_mode: str = "vae", # choice ["vae", "bottleneck"]
                 bottleneck_file: str = "./config_files/config_hydra/bottleneck/hfsq.yaml",
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.h_dim = h_dim
        input_feats = nfeats
        output_feats = nfeats
        self.arch = arch
        self.mlp_dist = False
        self.pe_type = "mld"

        self.query_pos_encoder = build_position_encoding(
            self.h_dim, position_embedding=position_embedding)
        self.query_pos_decoder = build_position_encoding(
            self.h_dim, position_embedding=position_embedding)

        encoder_layer = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.h_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)
        self.encoder_latent_proj = nn.Linear(self.h_dim, self.latent_dim)

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                                  decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.h_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                  decoder_norm)
        else:
            raise ValueError("Not support architecture!")
        self.decoder_latent_proj = nn.Linear(self.latent_dim, self.h_dim)

        self.global_motion_token = nn.Parameter(
            torch.randn(self.latent_size * 2, self.h_dim))

        self.skel_embedding = nn.Linear(input_feats, self.h_dim)
        self.final_layer = nn.Linear(self.h_dim, output_feats)

        self.register_buffer('latent_mean', torch.tensor(0))
        self.register_buffer('latent_std', torch.tensor(1))
        
        self.rep_mode = rep_mode
        if self.rep_mode == "bottleneck":
            bottleneck_kwargs = EasyDict(yaml.load(open(bottleneck_file, 'r'), Loader=yaml.FullLoader))
            bottleneck_name = bottleneck_kwargs.pop('bottleneck_name')
            self.bottleneck = Bottleneck(bottleneckBlock=bottleneck_name, **bottleneck_kwargs)
        else:
            self.bottleneck = NoBottleneck()

    def reparameterize(self, mu, logvar):
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        # return mu + eps * std  # 重参数化采样
    
        logvar = torch.clamp(logvar, min=-10, max=10)  # avoid numerical issues, otherwise denoiser rollout can break
        # if torch.isnan(mu).any() or torch.isinf(mu).any() or torch.isnan(logvar).any() or torch.isinf(logvar).any():
        #     pdb.set_trace()

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample() # (1, bs, latent_dim)
        return latent, dist
    
    def encode(
            self,
            future_motion, history_motion,
            scale_latent: bool = False,
            encode_mode: str = "quantizeds",
    ) -> Union[Tensor, Distribution]:
        device = future_motion.device
        bs, nfuture, nfeats = future_motion.shape
        nhistory = history_motion.shape[1]

        x = torch.cat((history_motion, future_motion), dim=1)  # [bs, H+F, nfeats]
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        if self.rep_mode == "bottleneck":
            latent = self.bottleneck.encode(x, encode_mode=encode_mode) # (bs, nframes, latent_dim)
            return latent, x
        elif self.rep_mode == "vae":
            # Switch sequence and batch_size because the input of
            # Pytorch Transformer is [Sequence, Batch size, ...]
            x = x.permute(1, 0, 2)  # [bs, nframes, h_dim] -> [nframes, bs, h_dim]

            # Each batch has its own set of tokens
            dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

            # adding the embedding token for all sequences
            xseq = torch.cat((dist, x), 0)

            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq)[:dist.shape[0]]  # [2*latent_size, bs, h_dim]
            dist = self.encoder_latent_proj(dist)  # [2*latent_size, bs, latent_dim]

            # content distribution
            mu = dist[0:self.latent_size, ...]
            logvar = dist[self.latent_size:, ...]
            latent, dist = self.reparameterize(mu, logvar)
        
            if scale_latent:  # only used during denoiser training
                latent = latent / self.latent_std
            return latent, latent

    def decode(self, z: Tensor, history_motion, nfuture,
               scale_latent: bool = False,
               decode_mode: str = "quantizeds",
               ):
        bs = history_motion.shape[0]
        if scale_latent:  # only used during denoiser training
            z = z * self.latent_std
        
        if self.rep_mode == "bottleneck":
            z = z.permute(1, 0, 2)  # [bs, nframes, latent_dim]
            z = self.bottleneck.decode(z, decode_mode=decode_mode)
        z = self.decoder_latent_proj(z)  # [latent_size, bs, latent_dim] => [latent_size, bs, h_dim]
        queries = torch.zeros(nfuture, bs, self.h_dim, device=z.device)
        history_embedding = self.skel_embedding(history_motion).permute(1, 0, 2)  # [nhistory, bs, h_dim]

        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((z, history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(
                xseq)[-nfuture:]

        elif self.arch == "encoder_decoder":
            xseq = torch.cat((history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(
                tgt=xseq,
                memory=z,
            )
            # print('output:', output.shape)
            output = output[-nfuture:]

        output = self.final_layer(output)
        # Pytorch Transformer: [Sequence, Batch size, ...]
        motion_recon = output.permute(1, 0, 2)
        return motion_recon
    
    def forward(self, future_motion, history_motion, nfuture):
        bs, nfuture, nfeats = future_motion.shape

        x = torch.cat((history_motion, future_motion), dim=1)  # [bs, H+F, nfeats]
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)
        x = x.permute(1, 0, 2)  # [bs, nframes, h_dim] -> [nframes, bs, h_dim]
        
        if self.rep_mode == "bottleneck":
            z, commit_loss, metrics = self.bottleneck.forward(x) # (bs, nframes, latent_dim)
        elif self.rep_mode == "vae":
            # Switch sequence and batch_size because the input of
            # Pytorch Transformer is [Sequence, Batch size, ...]

            # Each batch has its own set of tokens
            dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

            # adding the embedding token for all sequences
            xseq = torch.cat((dist, x), 0)

            xseq = self.query_pos_encoder(xseq)
            dist = self.encoder(xseq)[:dist.shape[0]]  # [2*latent_size, bs, h_dim]
            dist = self.encoder_latent_proj(dist)  # [2*latent_size, bs, latent_dim]

            # content distribution
            mu = dist[0:self.latent_size, ...]
            logvar = dist[self.latent_size:, ...]
            z, dist = self.reparameterize(mu, logvar)
        
            bs = history_motion.shape[0]
            metrics = {}
            commit_loss = torch.zeros(1, device=z.device)
        
        z = self.decoder_latent_proj(z)  # [n, bs, latent_dim] => [n, bs, h_dim]
        queries = torch.zeros(nfuture, bs, self.h_dim, device=z.device)
        history_embedding = self.skel_embedding(history_motion).permute(1, 0, 2)  # [nhistory, bs, h_dim]

        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((z, history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(
                xseq)[-nfuture:]

        elif self.arch == "encoder_decoder":
            xseq = torch.cat((history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(
                tgt=xseq,
                memory=z,
            )
            # print('output:', output.shape)
            output = output[-nfuture:]

        output = self.final_layer(output)
        # Pytorch Transformer: [Sequence, Batch size, ...]
        motion_recon = output.permute(1, 0, 2)
        return motion_recon, commit_loss, metrics

class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_channels, h_dim, time_range):
        super().__init__()
        # 借鉴 Code 2 的结构
        intermediate = input_channels // 3
        padding = (time_range - 1) // 2
        
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, intermediate, time_range, stride=1, padding=padding),
            nn.LayerNorm([intermediate, time_range]), # 注意 LN 在 1D 上的维度
            nn.ELU(),
            nn.Conv1d(intermediate, h_dim, time_range, stride=1, padding=padding),
            nn.ELU()
        )

    def forward(self, x):
        # x: [bs, seq_len, nfeats] -> [bs, nfeats, seq_len]
        x = x.permute(0, 2, 1)
        x = self.net(x)
        # -> [bs, h_dim, seq_len] -> [seq_len, bs, h_dim]
        return x.permute(2, 0, 1)
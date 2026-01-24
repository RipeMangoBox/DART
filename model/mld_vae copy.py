import pdb
from functools import reduce
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, autocast, nn
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

    def encode(
            self,
            future_motion, history_motion,
            scale_latent: bool = False,
    ) -> Union[Tensor, Distribution]:
        device = future_motion.device
        bs, nfuture, nfeats = future_motion.shape
        nhistory = history_motion.shape[1]

        x = torch.cat((history_motion, future_motion), dim=1)  # [bs, H+F, nfeats]
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, h_dim]

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
        logvar = torch.clamp(logvar, min=-10, max=10)  # avoid numerical issues, otherwise denoiser rollout can break
        # if torch.isnan(mu).any() or torch.isinf(mu).any() or torch.isnan(logvar).any() or torch.isinf(logvar).any():
        #     pdb.set_trace()

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample() # (1, bs, latent_dim)
        if scale_latent:  # only used during denoiser training
            latent = latent / self.latent_std
        return latent, dist

    def decode(self, z: Tensor, history_motion, nfuture,
               scale_latent: bool = False,
               ):
        bs = history_motion.shape[0]
        if scale_latent:  # only used during denoiser training
            z = z * self.latent_std
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
        feats = output.permute(1, 0, 2)
        return feats

class AutoMldPae(nn.Module):
    def __init__(self,
                 nfeats: int,
                 latent_dim: tuple = [1, 256], # PAE模式下，latent_dim[-1] 应与 PAE 的 latent_dim 对应
                 h_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 time_range: int = 10, # 对应 PAE 的 window size
                 window: float = 1/3,   # 时间窗口长度（秒）
                 pae_combine_mode: str = "linear",
                 arch: str = "all_encoder",
                 position_embedding: str = "learned",
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 **kwargs) -> None:
        super().__init__()

        self.h_dim = h_dim
        self.time_range = time_range
        self.latent_dim = latent_dim[-1] # PAE 的分量数
        self.latent_size = latent_dim[0]
        self.arch = arch

        self.query_pos_encoder = build_position_encoding(self.h_dim, position_embedding=position_embedding)
        self.query_pos_decoder = build_position_encoding(self.h_dim, position_embedding=position_embedding)
        
        # --- 1. 引入 PAE 专有的物理参数 ---
        self.tpi = nn.Parameter(torch.tensor([2.0 * np.pi]), requires_grad=False)
        self.args = nn.Parameter(torch.linspace(-window/2, window/2, time_range), requires_grad=False)
        self.freqs = nn.Parameter(torch.fft.rfftfreq(time_range)[1:] * time_range / window, requires_grad=False)

        # --- 2. 编码/解码结构 ---
        if pae_combine_mode == "conv":
            self.skel_embedding = ConvFeatureExtractor(nfeats, h_dim, time_range)
        else:
            self.skel_embedding = nn.Linear(nfeats, h_dim)
            
        self.decoder_latent_proj = nn.Linear(self.latent_dim, self.h_dim)

        encoder_layer = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.h_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.encoder_latent_proj = nn.Linear(self.h_dim, self.latent_dim)

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers, decoder_norm)
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
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers, decoder_norm)
        else:
            raise ValueError("Not support architecture!")
        
        # Transformer 用于处理时间序列
        # 注意：PAE 的逻辑通常基于 Encoder 提取的特征 y 来计算相位
        encoder_layer = nn.TransformerEncoderLayer(d_model=h_dim, nhead=4, dim_feedforward=1024, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 将 Transformer 输出映射到 PAE 的 embedding 通道
        self.to_phase_latent = nn.Linear(h_dim, self.latent_dim)

        # PAE 关键：相位投影 (对应 PAE 的 self.fc)
        self.phase_fc = nn.ModuleList([nn.Linear(time_range, 2) for _ in range(self.latent_dim)])

        # --- 3. 后处理结构 ---
        self.final_layer = nn.Linear(self.latent_dim, nfeats)
        
        self.register_buffer('latent_mean', torch.tensor(0))
        self.register_buffer('latent_std', torch.tensor(1))
        
        self.latent_param_mean = kwargs.get('latent_param_mean', torch.tensor(0))
        self.latent_param_std = kwargs.get('latent_param_std', torch.tensor(1))

    def FFT(self, y, dim=2):
        """ 这里的 y 形状为 [bs, latent_dim, time_range] """
        # 1. 记录原始数据类型
        orig_dtype = y.dtype
        
        # 2. 将输入转换为 float32 进行计算
        # 这样可以绕过 cuFFT 对半精度必须是 2 的幂次方的限制
        y_float32 = y.to(torch.float32)
        
        # 3. 执行 FFT
        rfft = torch.fft.rfft(y_float32, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:] 
        power = spectrum**2

        # 频率 (Weighted average frequency)
        freq = torch.sum(self.freqs * power, dim=dim) / (torch.sum(power, dim=dim) + 1e-8)
        # 振幅
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range
        # 偏移 (DC Component)
        offset = rfft.real[:, :, 0] / self.time_range
        return freq, amp, offset

    # @autocast(enabled = False, device_type='cuda')
    def encode(self, future_motion, history_motion, scale_latent: bool = False):
        bs = future_motion.shape[0]
        # 合并历史与未来 [bs, seq, feats]
        x = torch.cat((history_motion, future_motion), dim=1) 
        
        # 1. 语义浓缩 (Linear) -> [bs, seq, h_dim]
        x = self.skel_embedding(x)
        
        # 2. Transformer 时间建模
        y = self.encoder(x) # [bs, seq, h_dim]
        
        # 3. 映射到 PAE 隐空间 [bs, seq, h_dim] -> [bs, latent_dim, seq]
        y = self.to_phase_latent(y).permute(0, 2, 1)
        
        latent = y #Save latent for returning
        
        # 4. 计算 PAE 参数 (FFT)
        f, a, b = self.FFT(y, dim=2) # [bs, emb_ch]
        
        # 5. 计算相位 (Atan2)
        p = torch.empty((bs, self.latent_dim), device=y.device)
        for i in range(self.latent_dim):
            v = self.phase_fc[i](y[:, i, :]) # 输入整个时间窗口
            p[:, i] = torch.atan2(v[:, 1], v[:, 0]) / self.tpi

        #Parameters
        p = p.unsqueeze(2)  # (bs, latent_dim, 1)
        f = f.unsqueeze(2)  # (bs, latent_dim, 1)
        a = a.unsqueeze(2)  # (bs, latent_dim, 1)
        b = b.unsqueeze(2)  # (bs, latent_dim, 1)
        params = [p, f, a, b]

        return params, latent

    def decode(self, params: List[Tensor], history_motion, nfuture, scale_latent: bool = False):
        '''
        params: List[Tensor], shape: [bs, latent_dim, 1]
        '''
        bs = history_motion.shape[0]
            
        """ 接收 PAE 参数进行正弦重建 """
        # p, f, a, b = params.permute(1, 2, 0).split(self.latent_dim, dim=1) # 4 * (bs, latent_dim, 1)
        p, f, a, b = params

        # 1. 信号重建 (PAE 核心公式)
        # y: [bs, latent_dim, time_range]
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b
        
        # 2. 映射回特征空间 [bs, emb, seq] -> [seq, bs, emb]
        y = y.permute(2, 0, 1)
        
        y = self.decoder_latent_proj(y)  # [latent_size, bs, latent_dim] => [latent_size, bs, h_dim]
        queries = torch.zeros(nfuture, bs, self.h_dim, device=y.device)
        history_embedding = self.skel_embedding(history_motion).permute(1, 0, 2)  # [nhistory, bs, h_dim]

        # Pass through the transformer decoder
        # with the latent vector for memory
        if self.arch == "all_encoder":
            xseq = torch.cat((y, history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(
                xseq)[-nfuture:]

        elif self.arch == "encoder_decoder":
            xseq = torch.cat((history_embedding, queries), dim=0)
            xseq = self.query_pos_decoder(xseq)
            output = self.decoder(
                tgt=xseq,
                memory=y,
            )
            # print('output:', output.shape)
            output = output[-nfuture:]

        output = self.final_layer(output)
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats

class AutoMldPae_Phase(AutoMldPae):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def encode_to_manifold(self, future_motion, history_motion, scale_latent: bool = False):
        params, _ = self.encode(future_motion, history_motion, scale_latent)
        p, f, a, b = params
        # p = (p + 0.5) % 1.0 - 0.5  # normalize p to [-0.5, 0.5]
        
        # # 增加信号约束：防止 encode 阶段由于 linear 层初始化问题输出天文数字
        # f = torch.clamp(f, 0, 20)
        # a = torch.clamp(a, 0, 10)
        # b = torch.clamp(b, -10, 10)
        
        sx = torch.sin(2 * torch.pi * p)
        sy = torch.cos(2 * torch.pi * p)
        
        # 强制将 (sx, sy) 投影到单位圆，防止 Diffusion 生成全 0 向量
        # 这能极大稳定 atan2 的梯度
        norm = torch.sqrt(sx**2 + sy**2 + 1e-8)
        sx, sy = sx / norm, sy / norm
    
        latent_parameterization = torch.cat([f, a, b], dim=1) # [bs, 3 * latent_dim]
        normed_latent_parameterization = (latent_parameterization - self.latent_param_mean) / (self.latent_param_std + 1e-8)
        manifold = torch.cat([sx, sy, normed_latent_parameterization], dim=1)  # (1, bs, 5 * latent_dim)
        assert torch.isnan(manifold).any() == False
        return manifold

    def decode_from_manifold(self, manifold, history_motion, nfuture, scale_latent: bool = False):
        '''
        manifold: [bs, T=1, 5 * latent_dim]
        '''
        assert torch.isnan(manifold).any() == False

        sx, sy, f, a, b = manifold.split(self.latent_dim, dim=2)

        # 关键补丁：强制单位圆投影
        # Diffusion 预测的 sx, sy 模长可能不为 1，甚至趋近于 0，导致 atan2 梯度爆炸
        phase_norm = torch.sqrt(sx**2 + sy**2 + 1e-8)
        sx, sy = sx / phase_norm, sy / phase_norm
        p = torch.atan2(sy, sx) / (2 * torch.pi)
        
        normed_latent_parameterization = torch.cat([f, a, b], dim=2) # [bs, 1, 3 * latent_dim]
        latent_parameterization = normed_latent_parameterization * self.latent_param_std + self.latent_param_mean
        f, a, b = latent_parameterization.permute(0, 2, 1).split(self.latent_dim, dim=1) # [bs, latent_dim, 1]
        
        # 物理限制：反标准化后也需要裁剪，防止 decode 进入正弦函数时崩溃
        # f = torch.clamp(f, 0, 20)
        
        params = [p.permute(0, 2, 1), f, a, b]
        return self.decode(params, history_motion, nfuture, scale_latent)

class AutoMldPae_P(AutoMldPae):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode_to_manifold(self, future_motion, history_motion, scale_latent: bool = False):
        params, _ = self.encode(future_motion, history_motion, scale_latent)
        p, f, a, b = params
        latent_parameterization = torch.cat([f, a, b], dim=1) # [bs, 3 * latent_dim]
        normed_latent_parameterization = (latent_parameterization - self.latent_param_mean) / (self.latent_param_std + 1e-8)
        assert torch.isnan(normed_latent_parameterization).any() == False
    
        manifold = torch.cat([p, normed_latent_parameterization], dim=1) # (1, bs,  4 *latent_dim)
        
        return manifold
    
    def decode_from_manifold(self, manifold: List[Tensor], history_motion, nfuture, scale_latent: bool = False):
        '''
        manifold: [bs, T=1, 4 * latent_dim]
        '''
        p, f, a, b = manifold.split(self.latent_dim, dim=2) # 4 * [bs, 1, latent_dim]
        normed_latent_parameterization = torch.cat([f, a, b], dim=2) # [bs, 1, 3 * latent_dim]
        latent_parameterization = normed_latent_parameterization * self.latent_param_std + self.latent_param_mean
        f, a, b = latent_parameterization.permute(0, 2, 1).split(self.latent_dim, dim=1) # [bs, latent_dim, 1]
        params = [p.permute(0, 2, 1), f, a, b]
        return self.decode(params, history_motion, nfuture, scale_latent)

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
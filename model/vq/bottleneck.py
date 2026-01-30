import torch
import torch.nn as nn

from model.vq.FSQ import FSQ
from model.vq.GroupedResidualFSQ import GroupedResidualFSQ
from model.vq.residual_vq import ResidualVQ

# def exists(val):
#     '''for debug'''
#     assert val is not None
#     return val is not None

# def default(val, d):
#     return val if exists(val) else d
        
class FSQ_BottleneckBlock(nn.Module):    
    def __init__(self, *, dim, levels, **kwargs):
        super().__init__()

        # dim = default(kwargs['dim'], 512)
        # levels = default(kwargs['levels'], [8,5,5,5])

        self.fsq = FSQ(
            dim = dim,
            levels = levels,
            **kwargs['Norm']
        )
        
    def encode(self, z):
        all_indices = self.fsq.encode(z)
        return all_indices
    
    def decode(self, emb_ind):
        quantize = self.fsq.decode(emb_ind)
        return quantize
    
    def forward(self, x): 
        quantize, embed_ind, metrics = self.fsq(x)
        commit_loss = torch.zeros(()).to(x.device)
        return embed_ind, quantize, commit_loss, metrics
    
    
class GRFSQ_BottleneckBlock(nn.Module):    
    def __init__(self, *, dim, levels, groups, num_quantizers, use_group_norm, group_norm_type, base_mask_rate=0.0, **kwargs):
        super().__init__()
        
        self.grfsq = GroupedResidualFSQ(
            dim = dim,
            levels = levels,
            groups = groups,
            num_quantizers = num_quantizers,
            channel_last = True,
            group_norm_type = group_norm_type,
            use_group_norm = use_group_norm,
            base_mask_rate = base_mask_rate,
            
            **kwargs['RFSQ']
        )
        
    def encode(self, z, encode_mode="all_quantizeds"):
        output = self.grfsq.encode(z, encode_mode=encode_mode)
        return output
    
    def decode(self, z, decode_mode='indices'):
        embs = self.grfsq.decode(z, decode_mode)
        return embs
    
    def forward(self, x): 
        quantized, all_indices, commit_loss, all_metrics = self.grfsq(x)
        
        flattened_metrics = {}
        for k, v in all_metrics.items():
            for group in range(v.size(0)):            # group level
                for quantizer in range(v.size(1)):    # quantizer level
                    grk = f'{k}_group{group}_quantizer{quantizer}'
                    flattened_metrics[grk] = v[group][quantizer]
                    
        return all_indices, quantized, commit_loss, flattened_metrics
    
class RVQ_BottleneckBlock(nn.Module):    
    def __init__(self, sample_codebook_temp=0.5, **kwargs):
        super().__init__()
        
        self.sample_codebook_temp = sample_codebook_temp
        self.model = ResidualVQ(
            **kwargs,
        )
        
    def encode(self, z, encode_mode="all_quantizeds"):
        output = self.model.encode(z, encode_mode=encode_mode)
        return output
    
    def decode(self, z, decode_mode='all_quantizeds'):
        output = self.model.decode(z, decode_mode=decode_mode)
        return output
    
    def forward(self, x): 
        quantized, all_indices, commit_loss, all_metrics = self.model.forward(x, sample_codebook_temp=self.sample_codebook_temp)
                    
        return all_indices, quantized, commit_loss, all_metrics

bottleneck_switch = {
    'FSQ_BottleneckBlock': FSQ_BottleneckBlock,
    'GRFSQ_BottleneckBlock': GRFSQ_BottleneckBlock,
    'RVQ_BottleneckBlock': RVQ_BottleneckBlock,
}
    
class Bottleneck(nn.Module):
    def __init__(self, bottleneckBlock="FSQ_BottleneckBlock", **kwargs):
        super().__init__()
        
        assert bottleneckBlock in bottleneck_switch, f"bottleneckBlock must be one of {bottleneck_switch.keys()}"
        self.bottleneck = bottleneck_switch[bottleneckBlock](**kwargs)
            
    def encode(self, xs, encode_mode="all_quantizeds"):
        output = self.bottleneck.encode(xs, encode_mode=encode_mode)
        return output

    def decode(self, zs, decode_mode='indices'):
        xs_quantised = self.bottleneck.decode(zs, decode_mode)
        return xs_quantised

    def forward(self, xs): # xs.shape (b, d, interval//4), '4' stands for twice downsamples
        indices, x_quantised, commit_loss, metric = self.bottleneck(xs)
        
        if not self.training:
            # Be extra paranoid and make sure the encoder weights can't
            # change from straight-through estimator
            x_quantised = x_quantised.detach()
        
        return x_quantised, commit_loss, metric

class NoBottleneck(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def encode(self, xs, encode_mode="all_quantizeds"):
        return xs

    def decode(self, zs, decode_mode='indices'):
        return zs

    def forward(self, xs):
        commit_loss = torch.zeros((), device=xs.device)
        metrics = [{}]
        return xs, commit_loss, metrics

if __name__ == '__main__':
    # from jukebox.utils.dist_utils import setup_dist_from_mpi
    # rank, local_rank, device = setup_dist_from_mpi(port=29600)
    # bottleneck = Bottleneck(256, 64, 0.99, 2).to(device)
    # bottleneck.check()
    
    print('hello world')

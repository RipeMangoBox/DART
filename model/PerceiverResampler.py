import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat

# ==============================================================================
# 核心组件：移植自 lucidrains/flamingo-pytorch
# 这是一个成熟的、久经考验的 Perceiver Resampler 实现
# ==============================================================================

def exists(val):
    return val is not None

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult, bias=False),
            nn.GELU(),
            nn.Linear(dim * mult, dim, bias=False)
        )

    def forward(self, x):
        return self.net(x)

class PerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        x: Music Features (Media)
        latents: Learnable Queries
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # kv_input 是将 latents 和 x 拼接，允许 latents 之间互相注意 (Self-Attention)
        # 同时注意 x (Cross-Attention)。这是 Perceiver 的标准做法。
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b t (h d) -> b h t d', h=h), (q, k, v))

        q = q * self.scale

        # Attention
        sim = einsum('... i d, ... j d -> ... i j', q, k)
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)', h=h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    """
    标准的 Perceiver Resampler 块
    用于将变长序列 x 压缩为固定长度的 latents
    """
    def __init__(self, dim, depth, dim_head=64, heads=8, num_latents=1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim)
            ]))

    def forward(self, x):
        # x shape: (B, T, D)
        b = x.shape[0]
        # 复制 learnable query 到 batch 维度
        latents = repeat(self.latents, 'n d -> b n d', b=b)

        for attn, ff in self.layers:
            # Attention: Latents 查询 x
            latents = attn(x, latents) + latents
            # FFN
            latents = ff(latents) + latents
            
        return latents

# ==============================================================================
# 适配器：将 Music-to-Dance 适配到 Text-to-Motion 接口
# ==============================================================================

class MusicToGlobalCondAdapter(nn.Module):
    def __init__(self, music_input_dim=35, global_cond_output_dim=512, depth=2, heads=8):
        """
        Args:
            music_input_dim: 原始音乐特征维度 (如 Librosa/MERT 维度)
            text_output_dim: 目标 Text Embedding 维度 (如 CLIP 512/768)
            depth: Perceiver 层数，通常 2 层足够
        """
        super().__init__()
        
        # 1. 维度投影: 将音乐维度映射到 Latent 维度
        self.music_proj = nn.Linear(music_input_dim, global_cond_output_dim)
        
        # 2. Perceiver Resampler: 压缩时序信息
        # num_latents=1 表示我们要把所有信息压缩成 1 个 Token (全局信息)
        self.resampler = PerceiverResampler(
            dim=global_cond_output_dim,
            depth=depth,
            heads=heads,
            dim_head=64,
            num_latents=1 
        )

    def forward(self, emb_text):
        """
        Args:
            emb_text: 实际是 music features, shape (B, T, music_input_dim)
        Returns:
            global_cond: shape (B, text_output_dim) -> 适配 (B, D) 要求
        """
        # 1. 投影维度 (B, T, 35) -> (B, T, 512)
        x = self.music_proj(emb_text)
        
        # 2. Attention Pooling (B, T, 512) -> (B, 1, 512)
        # 这一步模型会自动根据 Learnable Query 提取全局节奏/风格信息
        latents = self.resampler(x)
        
        # 3. Squeeze: (B, 1, 512) -> (B, 512)
        # 满足你要求的 (B, D) 格式
        return latents.squeeze(1)

# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == "__main__":
    # 假设你的 Motion Model 需要的 text_dim 是 512
    # 你的音乐特征维度是 35
    adapter = MusicToGlobalCondAdapter(music_input_dim=35, text_output_dim=512).cuda()
    
    # 模拟输入 (Batch=32, Time=180, Dim=35)
    # 这里的 emb_text 实际上是 music slice
    fake_music_batch = torch.randn(32, 180, 35).cuda()
    
    # 前向传播
    # 输出应为 (32, 512)
    global_cond = adapter(fake_music_batch)
    
    print(f"Input Shape: {fake_music_batch.shape}") # torch.Size([32, 180, 35])
    print(f"Output Shape: {global_cond.shape}")     # torch.Size([32, 512])
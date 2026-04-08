# source : @pankratozzi (kaggle)
# source : alexandre st-georges
# see great original article https://huggingface.co/blog/annotated-diffusion for details

# ── Imports ───────────────────────────────────────────────────────────────────
import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from helpers import exists, default


# ── Low-level primitives ──────────────────────────────────────────────────────

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# ── Time embedding ────────────────────────────────────────────────────────────

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ── Attention ─────────────────────────────────────────────────────────────────

class LinearAttention(nn.Module):
    """O(n) attention — used in encoder/decoder paths."""
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    """Full O(n²) attention — used in the bottleneck."""
    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention, self).__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# ── ResNet blocks ─────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


# ── U-Net ─────────────────────────────────────────────────────────────────────

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super(Unet, self).__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)  # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_class = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # encoder
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    block_class(dim_in, dim_in, time_emb_dim=time_dim),
                    block_class(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

        # bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_class(mid_dim, mid_dim, time_emb_dim=time_dim)

        # decoder
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList([
                    block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    block_class(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                ])
            )

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_class(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


class cResNetBlock(nn.Module):
    """ResNet block with FiLM conditioning from [c_t ; c'].

    Forward pass:
        Conv1 → Norm → ⊗γ → ⊕β → SiLU → Conv2 → Norm → SiLU → + skip

    MLP II projects cat([c_t, c']) → 2 * dim_out, then splits into γ and β.
    """

    def __init__(self, dim, dim_out, *, time_emb_dim, cond_dim, groups=8):
        super().__init__()
        # MLP II: SiLU → Linear([c_t ; c'] → 2 * dim_out)
        self.mlp_II = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim + cond_dim, dim_out * 2),
        )

        self.conv1 = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, dim_out)

        self.conv2 = WeightStandardizedConv2d(dim_out, dim_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, dim_out)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb, cond):
        """
        Args:
            x:        [B, dim, H, W]       — feature map
            time_emb: [B, time_emb_dim]    — time embedding c_t
            cond:     [B, cond_dim]        — condition c' from F_enc
        """
        h = self.conv1(x)
        h = self.norm1(h)

        # ── MLP II: γ, β from [c_t ; c'] ─────────────────────────────────
        film = self.mlp_II(torch.cat([time_emb, cond], dim=1))  # [B, dim_out*2]
        film = film[:, :, None, None]                            # broadcast to [B, dim_out*2, 1, 1]
        gamma, beta = film.chunk(2, dim=1)                       # each [B, dim_out, 1, 1]

        # ── ⊗γ → ⊕β → SiLU ───────────────────────────────────────────────
        h = h * (1 + gamma) + beta
        h = F.silu(h)

        # ── Conv2 → Norm → SiLU ───────────────────────────────────────────
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.res_conv(x)

class Fenc(nn.Module):
    """Lightweight CNN encoder F_enc: image → condition vector c'.

    Four strided conv layers halve spatial resolution each time, then
    global average pooling and a final linear project to cond_dim.
    """

    def __init__(self, channels=3, cond_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 4, stride=2, padding=1),   # H/2
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),          # H/4
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),         # H/8
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),        # H/16
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),                             # [B, 256, 1, 1]
            nn.Flatten(),                                        # [B, 256]
            nn.Linear(256, cond_dim),
        )

    def forward(self, x):
        return self.net(x)


class cUnet(nn.Module):
    """U-Net where every ResNet block is replaced by cResNetBlock.

    The condition vector c' is produced by an internal F_enc image encoder
    and injected (together with the time embedding c_t) via FiLM into
    every block of the encoder, bottleneck, and decoder.
    """

    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        cond_dim=256,
        resnet_block_groups=4,
    ):
        super().__init__()

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # ── Condition encoder F_enc ───────────────────────────────────────
        self.fenc = Fenc(channels=channels, cond_dim=cond_dim)

        # ── Time embedding MLP I ──────────────────────────────────────────
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        block_class = partial(
            cResNetBlock,
            time_emb_dim=time_dim,
            cond_dim=cond_dim,
            groups=resnet_block_groups,
        )

        # ── Encoder ───────────────────────────────────────────────────────
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_class(dim_in, dim_in),
                block_class(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
            ]))

        # ── Bottleneck ────────────────────────────────────────────────────
        mid_dim = dims[-1]
        self.mid_block1 = block_class(mid_dim, mid_dim)
        self.mid_attn   = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_class(mid_dim, mid_dim)

        # ── Decoder ───────────────────────────────────────────────────────
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_class(dim_out + dim_in, dim_out),
                block_class(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
            ]))

        self.cond_dim = cond_dim
        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_class(dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        # Learned null embedding for the CFG unconditional path (condition=None)
        self.null_cond = nn.Parameter(torch.zeros(1, cond_dim))

    def encode(self, x):
        """Encode a reference image to the condition vector c'. Shape: [B, cond_dim]."""
        return self.fenc(x)

    def forward(self, x, time, condition=None):
        """
        Args:
            x:         [B, C, H, W]   — noisy image x_t
            time:      [B]            — diffusion timesteps
            condition: [B, C, H, W]   — raw condition image (will be encoded by F_enc)
                    OR [B, cond_dim]  — already-encoded condition vector
                    OR None           — unconditional path (CFG null embedding used)
        """
        # ── Resolve condition to a [B, cond_dim] vector ──────────────────────
        if condition is not None:
            if condition.dim() == 4:               # raw image [B, C, H, W]
                cond = self.fenc(condition)        # → [B, cond_dim]
            else:                                  # already encoded [B, cond_dim]
                cond = condition
        else:
            # CFG unconditional path: broadcast learned null embedding
            cond = self.null_cond.expand(x.size(0), -1)   # [B, cond_dim]

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)   # c_t: [B, time_dim]

        h = []

        # ── Encoder ──────────────────────────────────────────────────────────
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, cond)    # cResNetBlock: FiLM(c_t, c')
            h.append(x)
            x = block2(x, t, cond)    # cResNetBlock: FiLM(c_t, c')
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # ── Bottleneck ────────────────────────────────────────────────────────
        x = self.mid_block1(x, t, cond)   # cResNetBlock: FiLM(c_t, c')
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, cond)   # cResNetBlock: FiLM(c_t, c')

        # ── Decoder ──────────────────────────────────────────────────────────
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, cond)    # cResNetBlock: FiLM(c_t, c')
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, cond)    # cResNetBlock: FiLM(c_t, c')
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t, cond)   # cResNetBlock: FiLM(c_t, c')
        return self.final_conv(x)
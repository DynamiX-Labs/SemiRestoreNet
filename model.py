"""
model.py — High-Performance Semiconductor Image Restoration & Super-Resolution Network.

Architecture:
    Input [B, 1, H, W] (Unclipped Float32 SEM Image)
      → Dual-Domain Feature Extraction:
          - Linear Path: Conv First (3×3, 64ch)
          - Homomorphic Log Path: SignedLog(x) → Conv Log (3×3, 64ch)
          - Dynamic Gated Fusion (GFM): Spatial-channel soft routing between Linear & Log
      → Stage 1: 8× RRDB (Residual-in-Residual Dense Blocks) [Features: F1]
      → Swin Transformer Block 1 (Window=8, Shifted-Window Self-Attention)
      → Stage 2: 8× RRDB [Features: F2]
      → Swin Transformer Block 2 (Window=16, Long-Range Periodic Array Self-Attention)
      → Stage 3: 7× RRDB [Total: 23 RRDB Blocks]
      → Anisotropic Directional CBAM (1×9 Strip + 9×1 Strip + 7×7 2D Defect Attention)
      → Conv Body (3×3) + Global Trunk Residual Skip
      → Cross-Stage Dense Highway: F_head = F_trunk + γ1·Proj1(F1) + γ2·Proj2(F2)
      → Restoration Head (PixelShuffle for 2× SR, Conv for 1× Denoising)
      → Output: y_hat = Up(x) + Delta_x (Global Base Image Residual)

Transfer Learning:
    Includes `load_pretrained_rrdb_weights` to transfer weights from Real-ESRGAN / ESRGAN
    checkpoints with RGB→Grayscale channel averaging for instant convergence.
"""

import math
import os
from pathlib import Path
from typing import Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# DropPath (Stochastic Depth)
# =============================================================================

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# =============================================================================
# Physics-Aware Homomorphic Signed Log Transformation & Gated Fusion
# =============================================================================

class SignedLogTransform(nn.Module):
    """Physics-aware signed log transformation for multiplicative speckle.
    
    Computes: y_log = sign(x) * log(1.0 + |x| / epsilon)
    
    Properties:
        - Preserves exact sign for dark/unclipped pixels (no sign loss).
        - Smooth, monotonic, zero-centered (f(0) = 0).
        - Converts multiplicative Gamma speckle noise into additive noise for CNNs.
        - Avoids NaN on unclipped negative input values.
    """
    def __init__(self, epsilon: float = 0.05):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        return sign * torch.log1p(abs_x / self.epsilon)


class DynamicGatedFusion(nn.Module):
    """Spatial-channel dynamic gated fusion between linear and log-domain streams.
    
    Computes learnable soft gate alpha(x) in [0, 1] per spatial-channel location:
        alpha = Sigmoid(Conv1x1(LeakyReLU(Conv3x3([F_lin, F_log]))))
        F_fused = alpha * F_log + (1 - alpha) * F_lin + Conv_proj([F_lin, F_log])
    """
    def __init__(self, num_feat: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
        
    def forward(self, feat_lin: torch.Tensor, feat_log: torch.Tensor) -> torch.Tensor:
        cat_feat = torch.cat([feat_lin, feat_log], dim=1)
        alpha = self.gate(cat_feat)
        gated = alpha * feat_log + (1.0 - alpha) * feat_lin
        return gated + self.proj(cat_feat)


# =============================================================================
# Residual Dense Block (RDB) & Residual-in-Residual Dense Block (RRDB)
# =============================================================================

class ResidualDenseBlock(nn.Module):
    """5-layer Residual Dense Block with growth channels."""
    
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block (3 RDBs + residual scaling)."""
    
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


# =============================================================================
# Shifted-Window Self-Attention (Swin Transformer)
# =============================================================================

class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention (W-MSA / SW-MSA)."""
    
    def __init__(self, dim: int = 64, window_size: int = 8, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        return out


class MLP(nn.Module):
    """Multi-Layer Perceptron for Transformer blocks."""
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 2
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SwinTransformerLayer(nn.Module):
    """Single Swin Transformer layer with cyclic shifting."""
    
    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 4,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))
        
        self._attn_mask = None
        self._mask_shape = None
    
    def _create_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        if self.shift_size == 0:
            return None
        if self._mask_shape == (H, W) and self._attn_mask is not None and self._attn_mask.device == device:
            return self._attn_mask
            
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = rearrange(
            img_mask, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c',
            p1=self.window_size, p2=self.window_size
        )
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        self._attn_mask = attn_mask
        self._mask_shape = (H, W)
        return self._attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, Hp, Wp = x.shape
        
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        shortcut = x_flat
        x_norm = self.norm1(x_flat)
        x_img = rearrange(x_norm, 'b (h w) c -> b h w c', h=Hp, w=Wp)
        
        if self.shift_size > 0:
            shifted_x = torch.roll(x_img, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_img
            
        x_windows = rearrange(
            shifted_x, 'b (h p1) (w p2) c -> (b h w) (p1 p2) c',
            p1=self.window_size, p2=self.window_size
        )
        
        mask = self._create_mask(Hp, Wp, x.device)
        attn_windows = self.attn(x_windows, mask=mask)
        
        shifted_x = rearrange(
            attn_windows, '(b h w) (p1 p2) c -> b (h p1) (w p2) c',
            h=Hp // self.window_size, w=Wp // self.window_size,
            p1=self.window_size, p2=self.window_size
        )
        
        if self.shift_size > 0:
            x_img = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_img = shifted_x
            
        x_flat = rearrange(x_img, 'b h w c -> b (h w) c')
        x_flat = shortcut + self.drop_path(x_flat)
        x_flat = x_flat + self.drop_path(self.mlp(self.norm2(x_flat)))
        
        out = rearrange(x_flat, 'b (h w) c -> b c h w', h=Hp, w=Wp)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
            
        return out


class SwinTransformerBlock(nn.Module):
    """Pair of W-MSA and SW-MSA layers."""
    
    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 4,
        window_size: int = 8,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.block1 = SwinTransformerLayer(
            dim, num_heads, window_size, shift_size=0, drop_path_rate=drop_path_rate
        )
        self.block2 = SwinTransformerLayer(
            dim, num_heads, window_size, shift_size=window_size // 2, drop_path_rate=drop_path_rate
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


# =============================================================================
# Anisotropic Directional Defect Attention Module (CBAM)
# =============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int = 64, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class AnisotropicSpatialAttention(nn.Module):
    """Directional Strip + 2D Spatial Attention for semiconductor line gratings.
    
    Combines:
        - 1x9 horizontal strip conv (horizontal gates/wordlines)
        - 9x1 vertical strip conv (vertical fins/bitlines)
        - 7x7 standard 2D spatial conv (point defect anomalies)
    """
    def __init__(self, kernel_size: int = 7, strip_kernel: int = 9):
        super().__init__()
        self.conv_2d = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.conv_h = nn.Conv2d(2, 1, (1, strip_kernel), padding=(0, strip_kernel // 2), bias=False)
        self.conv_v = nn.Conv2d(2, 1, (strip_kernel, 1), padding=(strip_kernel // 2, 0), bias=False)
        self.fuse = nn.Conv2d(3, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        map_2d = self.conv_2d(x_cat)
        map_h = self.conv_h(x_cat)
        map_v = self.conv_v(x_cat)
        
        fused = self.fuse(torch.cat([map_2d, map_h, map_v], dim=1))
        return self.sigmoid(fused)


class CBAM(nn.Module):
    """Directional Anisotropic CBAM for sharpening nanoscale defect boundaries."""
    def __init__(self, in_planes: int = 64, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = AnisotropicSpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# =============================================================================
# Restoration & Super-Resolution Head
# =============================================================================

class RestorationHead(nn.Module):
    """Restoration Head supporting 1x same-resolution and 2x super-resolution."""
    
    def __init__(self, num_feat: int = 64, out_channels: int = 1, upscale_factor: int = 1):
        super().__init__()
        self.upscale_factor = upscale_factor
        
        if upscale_factor == 2:
            self.head = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_feat, out_channels, 3, 1, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_feat, out_channels, 3, 1, 1),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# =============================================================================
# Full Model: SemiRestoreNet (23 RRDB + 2 Swin + CBAM + Dual-Domain + Highway)
# =============================================================================

class FullModel(nn.Module):
    """Deep Hybrid Backbone for Semiconductor Image Restoration & Super-Resolution.
    
    Args:
        in_channels: Input channels (default: 1 for grayscale SEM).
        num_feat: Intermediate feature channels (default: 64).
        num_grow_ch: RDB growth channels (default: 32).
        num_rrdb_blocks: Tuple of blocks per stage (default: (8, 8, 7) = 23 blocks).
        window_size: Base Swin window size (default: 8).
        upscale_factor: 1 for same-res denoising, 2 for 2x super-resolution.
        drop_path_rate: Stochastic depth rate (default: 0.1).
        use_log_domain: If True, enables dynamic gated homomorphic log-domain stream.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_feat: int = 64,
        num_grow_ch: int = 32,
        num_rrdb_blocks: tuple = (8, 8, 7),
        window_size: int = 8,
        upscale_factor: int = 1,
        drop_path_rate: float = 0.1,
        use_log_domain: bool = True,
    ):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.num_rrdb_blocks = num_rrdb_blocks
        self.use_log_domain = use_log_domain
        
        # 1. Shallow Feature Extraction (Linear Path)
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        
        # 1b. Dynamic Gated Homomorphic Log Stream (Signed Log for Speckle)
        if use_log_domain:
            self.log_transform = SignedLogTransform(epsilon=0.05)
            self.conv_log = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
            self.fusion = DynamicGatedFusion(num_feat)
        
        # 2. Stage 1: Dense Convolutions (Low-level edge features)
        self.stage1 = nn.ModuleList([
            RRDB(num_feat, num_grow_ch) for _ in range(num_rrdb_blocks[0])
        ])
        
        # 3. Swin Block 1 (Window=8)
        self.swin1 = SwinTransformerBlock(
            dim=num_feat, num_heads=4, window_size=window_size, drop_path_rate=drop_path_rate
        )
        
        # 4. Stage 2: Dense Convolutions (Mid-level grating array features)
        self.stage2 = nn.ModuleList([
            RRDB(num_feat, num_grow_ch) for _ in range(num_rrdb_blocks[1])
        ])
        
        # 5. Swin Block 2 (Window=16 for long-range periodic array regularities)
        swin2_window = min(window_size * 2, 16)
        self.swin2 = SwinTransformerBlock(
            dim=num_feat, num_heads=4, window_size=swin2_window, drop_path_rate=drop_path_rate
        )
        
        # 6. Stage 3: Dense Convolutions (High-level abstract features)
        self.stage3 = nn.ModuleList([
            RRDB(num_feat, num_grow_ch) for _ in range(num_rrdb_blocks[2])
        ])
        
        # 7. Directional Anisotropic Defect Attention (CBAM)
        self.cbam = CBAM(num_feat)
        
        # 8. Trunk Convolution
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # 8b. Cross-Stage Dense Highway (Multi-scale early skip connections to head)
        self.highway_proj1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.highway_proj2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.gamma1 = nn.Parameter(torch.tensor(0.1))
        self.gamma2 = nn.Parameter(torch.tensor(0.1))
        
        # 9. Reconstruction Head
        self.restoration_head = RestorationHead(
            num_feat=num_feat, out_channels=1, upscale_factor=upscale_factor
        )
    
    def forward(self, x: torch.Tensor, return_dict: bool = True):
        """
        Args:
            x: Input degraded image [B, 1, H, W] (Float32, unclipped).
            return_dict: If True, returns dict with 'restored'; else tensor.
            
        Returns:
            Dict or Tensor of restored image [B, 1, s*H, s*W].
        """
        # Shallow features with optional dynamic gated homomorphic fusion
        feat_lin = self.conv_first(x)
        if self.use_log_domain:
            feat_log = self.conv_log(self.log_transform(x))
            feat_first = self.fusion(feat_lin, feat_log)
        else:
            feat_first = feat_lin
            
        feat = feat_first
        
        # Stage 1 (Extract early shallow edge features)
        for block in self.stage1:
            feat = block(feat)
        feat_stage1 = feat
            
        # Swin 1
        feat = self.swin1(feat)
        
        # Stage 2 (Extract mid-level periodic features)
        for block in self.stage2:
            feat = block(feat)
        feat_stage2 = feat
            
        # Swin 2
        feat = self.swin2(feat)
        
        # Stage 3
        for block in self.stage3:
            feat = block(feat)
            
        # Defect Attention + Global Trunk Residual Skip
        feat = self.cbam(feat)
        feat_trunk = self.conv_body(feat) + feat_first
        
        # Cross-Stage Dense Highway Injection
        feat_head_in = (
            feat_trunk
            + self.gamma1 * self.highway_proj1(feat_stage1)
            + self.gamma2 * self.highway_proj2(feat_stage2)
        )
        
        # High-frequency residual
        residual = self.restoration_head(feat_head_in)
        
        # Global base image skip connection: y_hat = Up(x) + Delta_x
        if self.upscale_factor > 1:
            base_img = F.interpolate(
                x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False
            )
        else:
            base_img = x
            
        restored = base_img + residual
        
        if return_dict:
            return {'restored': restored}
        return restored
    
    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        """Extract intermediate stage features for Knowledge Distillation."""
        feat_lin = self.conv_first(x)
        if self.use_log_domain:
            feat_log = self.conv_log(self.log_transform(x))
            feat_first = self.fusion(feat_lin, feat_log)
        else:
            feat_first = feat_lin
            
        feat = feat_first
        features = {}
        
        for block in self.stage1:
            feat = block(feat)
        features['after_stage1'] = feat
        feat_stage1 = feat
        
        feat = self.swin1(feat)
        
        for block in self.stage2:
            feat = block(feat)
        features['after_stage2'] = feat
        feat_stage2 = feat
        
        feat = self.swin2(feat)
        
        for block in self.stage3:
            feat = block(feat)
        features['after_stage3'] = feat
        
        feat = self.cbam(feat)
        feat_trunk = self.conv_body(feat) + feat_first
        
        feat_head_in = (
            feat_trunk
            + self.gamma1 * self.highway_proj1(feat_stage1)
            + self.gamma2 * self.highway_proj2(feat_stage2)
        )
        
        residual = self.restoration_head(feat_head_in)
        base_img = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False) if self.upscale_factor > 1 else x
        features['restored'] = base_img + residual
        return features


# =============================================================================
# Pretrained RRDB Weight Transfer (ESRGAN / Real-ESRGAN -> SemiRestoreNet)
# =============================================================================

def load_pretrained_rrdb_weights(
    model: nn.Module,
    weights_path_or_dict: Union[str, Path, Dict[str, Any]],
    strict: bool = False,
    verbose: bool = True,
) -> dict:
    """Transfer weights from pretrained ESRGAN / Real-ESRGAN (RRDBNet) to SemiRestoreNet.
    
    Mapping Strategy:
        - `conv_first.weight`: shape [64, 3, 3, 3] -> Grayscale average [64, 1, 3, 3]
        - `conv_first.bias`: shape [64] -> direct transfer
        - `body.{0..7}` -> `stage1.{0..7}` (8 RRDB blocks)
        - `body.{8..15}` -> `stage2.{0..7}` (8 RRDB blocks)
        - `body.{16..22}` -> `stage3.{0..6}` (7 RRDB blocks) [Total 23 RRDBs]
        - `conv_body.{weight, bias}` -> direct transfer
    
    Newly added semiconductor-specific modules (Swin attention, Anisotropic CBAM,
    Highway projections, Gated Log Fusion, and Task Heads) remain initialized for domain fine-tuning.
    
    Args:
        model: FullModel instance to load weights into.
        weights_path_or_dict: Path to .pth/.pt file or raw state_dict dictionary.
        strict: If True, raises error on unmatched keys.
        verbose: If True, prints transfer summary.
        
    Returns:
        Dict with 'transferred_keys', 'missing_keys', 'unmapped_keys', 'transferred_params'.
    """
    if isinstance(weights_path_or_dict, (str, Path)):
        weights_path = Path(weights_path_or_dict)
        if not weights_path.exists():
            raise FileNotFoundError(f"Pretrained weights file not found: {weights_path}")
        raw_dict = torch.load(str(weights_path), map_location='cpu')
    else:
        raw_dict = weights_path_or_dict
        
    # Unwrap state dict if nested
    for wrapper_key in ['params_ema', 'params', 'model', 'net_g', 'state_dict']:
        if isinstance(raw_dict, dict) and wrapper_key in raw_dict:
            raw_dict = raw_dict[wrapper_key]
            
    model_state = model.state_dict()
    transferred_dict = {}
    transferred_keys = []
    unmapped_keys = []
    
    for k, v in raw_dict.items():
        clean_k = k
        # Strip common prefixes
        for prefix in ['module.', 'net_g.', 'model.']:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
                
        target_k = None
        
        # 1. First convolution (RGB -> Grayscale adaptation)
        if clean_k == 'conv_first.weight':
            target_k = 'conv_first.weight'
            if v.shape[1] == 3 and model_state[target_k].shape[1] == 1:
                # Average RGB channels: (W_R + W_G + W_B) / 3
                v = v.mean(dim=1, keepdim=True)
            elif v.shape != model_state[target_k].shape:
                continue
        elif clean_k == 'conv_first.bias':
            target_k = 'conv_first.bias'
            
        # 2. Trunk convolution
        elif clean_k.startswith('conv_body.'):
            target_k = clean_k
            
        # 3. 23 RRDB blocks mapping to stage1 (8), stage2 (8), stage3 (7)
        elif clean_k.startswith('body.'):
            parts = clean_k.split('.')
            try:
                block_idx = int(parts[1])
                sub_path = '.'.join(parts[2:])
                
                if 0 <= block_idx < 8:
                    target_k = f'stage1.{block_idx}.{sub_path}'
                elif 8 <= block_idx < 16:
                    target_k = f'stage2.{block_idx - 8}.{sub_path}'
                elif 16 <= block_idx < 23:
                    target_k = f'stage3.{block_idx - 16}.{sub_path}'
            except (ValueError, IndexError):
                target_k = None
                
        if target_k is not None and target_k in model_state:
            if model_state[target_k].shape == v.shape:
                transferred_dict[target_k] = v
                transferred_keys.append(target_k)
            else:
                unmapped_keys.append(f"{k} (shape mismatch: src {v.shape} vs dst {model_state[target_k].shape})")
        else:
            unmapped_keys.append(k)
            
    # Load into model
    missing_keys, unexpected_keys = model.load_state_dict(transferred_dict, strict=False)
    
    total_transferred_params = sum(p.numel() for p in transferred_dict.values())
    total_model_params = sum(p.numel() for p in model.parameters())
    transfer_pct = (total_transferred_params / max(total_model_params, 1)) * 100.0
    
    if verbose:
        print(f"[TransferLearning] Pretrained RRDB weights loaded successfully:")
        print(f"  - Transferred keys: {len(transferred_keys)} tensors ({total_transferred_params:,} parameters, {transfer_pct:.1f}% of model)")
        print(f"  - Untrained/Domain keys: {len(missing_keys)} (Swin Attention, CBAM, Highway, Heads, Gated Log Fusion)")
        print(f"  - Unmapped source keys: {len(unmapped_keys)}")
        
    return {
        'transferred_keys': transferred_keys,
        'missing_keys': missing_keys,
        'unmapped_keys': unmapped_keys,
        'transferred_params': total_transferred_params,
        'transfer_percentage': transfer_pct,
    }


# =============================================================================
# Factory Functions
# =============================================================================

def create_teacher_model(
    num_rrdb_blocks: tuple = (8, 8, 7),
    upscale_factor: int = 1,
    use_log_domain: bool = True,
    **kwargs,
) -> FullModel:
    """Create full 23-RRDB Teacher model (16.86M parameters)."""
    return FullModel(
        num_rrdb_blocks=num_rrdb_blocks,
        upscale_factor=upscale_factor,
        use_log_domain=use_log_domain,
        **kwargs
    )


def create_student_model(num_blocks: int = 8, upscale_factor: int = 1, use_log_domain: bool = True, **kwargs) -> FullModel:
    """Create compact Student model (e.g. 8 blocks: 3+3+2, 6.39M parameters)."""
    if num_blocks == 8:
        blocks = (3, 3, 2)
    elif num_blocks == 16:
        blocks = (6, 5, 5)
    else:
        b = num_blocks // 3
        blocks = (b, b, num_blocks - 2 * b)
        
    return FullModel(
        num_rrdb_blocks=blocks,
        upscale_factor=upscale_factor,
        use_log_domain=use_log_domain,
        **kwargs
    )

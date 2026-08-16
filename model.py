"""
model.py — High-Performance Semiconductor Image Restoration & Super-Resolution Network.

Architectural Innovations & Engineering Solutions:
---------------------------------------------------
1. Restormer-Style Multi-Dconv Head Transposed Attention (MDTA):
   - Computes cross-covariance across channels with linear spatial complexity O(HW C^2).
   - Achieves an unconstrained global receptive field to capture periodic transistor pitches 
     (FinFET arrays, DRAM capacitor cells) across the entire die without windowed boundary cuts.

2. Explicit Noise-Conditioned Gated Fusion Module (NoiseConditionedGFM):
   - Computes an analytical high-frequency Laplacian residual / noise-level map.
   - Explicitly conditions the soft routing alpha(x) in [0, 1] between linear and homomorphic 
     log-domain streams, relieving early convolutions from re-learning noise variance estimation.

3. Multi-Scale Manhattan Anisotropic Attention (MultiScaleManhattanAttention):
   - Integrates multi-scale orthogonal strip convolutions: fine-pitch (1x7, 7x1) and 
     coarse-pitch/wordlines (1x15, 15x1) alongside 7x7 spatial pooling to eliminate line collapse.

4. Decoupled Two-Stage Restoration Head (DecoupledRestorationHead):
   - Decouples native-resolution (1x) spatial phase denoising from sub-pixel (2x) PixelShuffle edge synthesis.

5. Structural Reparameterization (RepBlock):
   - Multi-branch (3x3 + 1x1 + Identity) during training -> algebraically collapsed into a single 
     standard 3x3 convolution at inference via `switch_to_deploy()`. Zero runtime speed penalty.

6. SignedLogTransform & Pretrained RRDB Backbone Transfer:
   - Preserves exact signs and handles negative detector floats without NaN crashes.
   - Retains 23-RRDB dense feature extraction transfer from Real-ESRGAN.
"""

import math
import os
from pathlib import Path
from typing import Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
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
# Physics-Aware Homomorphic Signed Log Transformation & Noise Estimation
# =============================================================================

class SignedLogTransform(nn.Module):
    """Physics-aware signed log transformation for multiplicative speckle.
    
    Computes: y_log = sign(x) * log(1.0 + |x| / epsilon)
    
    Properties:
        - Preserves exact sign for dark/unclipped pixels (no sign loss).
        - Smooth, monotonic, zero-centered (f(0) = 0).
        - Converts multiplicative Gamma speckle noise into additive noise for CNNs.
        - Avoids NaN on unclipped negative detector electronic offset floats.
    """
    def __init__(self, epsilon: float = 0.05):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        return sign * torch.log1p(abs_x / self.epsilon)


class NoiseEstimator(nn.Module):
    """Estimates local high-frequency noise variance and SNR map from input telemetry."""
    def __init__(self, in_channels: int = 1, out_feat: int = 16):
        super().__init__()
        laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian', laplacian)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_feat, out_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.conv2d(x, self.laplacian, padding=1)
        abs_res = torch.abs(residual)
        return self.net(abs_res)


class NoiseConditionedGFM(nn.Module):
    """Spatial-channel dynamic gated fusion conditioned on explicit local noise level.
    
    Computes soft gate alpha(x) in [0, 1] per spatial-channel location:
        alpha = Sigmoid(Conv1x1(LeakyReLU(Conv3x3([F_lin, F_log, F_noise]))))
        F_fused = alpha * F_log + (1 - alpha) * F_lin + Conv_proj([F_lin, F_log])
    """
    def __init__(self, num_feat: int = 64, noise_feat: int = 16):
        super().__init__()
        self.noise_feat = noise_feat
        self.noise_estimator = NoiseEstimator(in_channels=1, out_feat=noise_feat)
        self.gate = nn.Sequential(
            nn.Conv2d(num_feat * 2 + noise_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
        
    def forward(self, feat_lin: torch.Tensor, feat_log: torch.Tensor, raw_input: torch.Tensor = None) -> torch.Tensor:
        if raw_input is not None:
            feat_noise = self.noise_estimator(raw_input)
        else:
            b, _, h, w = feat_lin.shape
            feat_noise = torch.zeros(b, self.noise_feat, h, w, device=feat_lin.device, dtype=feat_lin.dtype)
            
        cat_gate = torch.cat([feat_lin, feat_log, feat_noise], dim=1)
        alpha = self.gate(cat_gate)
        cat_proj = torch.cat([feat_lin, feat_log], dim=1)
        gated = alpha * feat_log + (1.0 - alpha) * feat_lin
        return gated + self.proj(cat_proj)


class FiLMNoiseConditioner(nn.Module):
    """FiLM-style noise conditioning: predict noise level scalar, modulate features.
    
    Uses the NoiseEstimator feature map to:
    1. Predict a noise-level scalar (supervised with auxiliary MSE loss)
    2. Modulate main trunk features via Feature-wise Linear Modulation (FiLM):
       output = (1 + gamma) * features + beta
    """
    def __init__(self, noise_feat: int = 16, num_feat: int = 64):
        super().__init__()
        self.noise_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(noise_feat, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # noise_level is in [0, 1]
        )
        self.gamma_net = nn.Sequential(
            nn.Linear(1, num_feat),
            nn.Tanh(),  # scale modulation in [-1, 1]
        )
        self.beta_net = nn.Linear(1, num_feat)
    
    def forward(self, noise_features: torch.Tensor, main_features: torch.Tensor):
        """Args:
            noise_features: [B, noise_feat, H, W] from NoiseEstimator
            main_features: [B, num_feat, H, W] trunk features to modulate
        Returns:
            modulated_features: [B, num_feat, H, W]
            noise_level_pred: [B, 1] predicted noise level scalar
        """
        noise_level_pred = self.noise_predictor(noise_features)  # [B, 1]
        gamma = self.gamma_net(noise_level_pred).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta_net(noise_level_pred).unsqueeze(-1).unsqueeze(-1)
        modulated = (1.0 + gamma) * main_features + beta
        return modulated, noise_level_pred


# Backwards compatibility alias
DynamicGatedFusion = NoiseConditionedGFM


# =============================================================================
# Structural Reparameterization Block (RepBlock)
# =============================================================================

class RepBlock(nn.Module):
    """Structural Reparameterization Block (RepVGG-style for Fast Student Inference).
    
    Training Mode:
        - 3x3 Conv branch (rich local context)
        - 1x1 Conv branch (cross-channel mixing)
        - Identity skip branch (if in_channels == out_channels)
    Inference Mode (post switch_to_deploy()):
        - Single mathematically equivalent 3x3 Conv kernel.
        - Exactly 0% speed penalty and 0% memory overhead at runtime!
    """
    def __init__(self, in_channels: int = 64, out_channels: int = 64, act_layer=nn.LeakyReLU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_deployed = False
        
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)
        self.has_identity = (in_channels == out_channels)
        self.act = act_layer(negative_slope=0.2, inplace=True)
        self.rbr_reparam = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_deployed:
            return self.act(self.rbr_reparam(x))
        
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.has_identity:
            out = out + x
        return self.act(out)
        
    def switch_to_deploy(self):
        """Collapses multi-branch weights into a single standard 3x3 Conv2d layer."""
        if self.is_deployed:
            return
        
        w3 = self.conv3x3.weight.data
        b3 = self.conv3x3.bias.data if self.conv3x3.bias is not None else torch.zeros(self.out_channels, device=w3.device)
        
        # Pad 1x1 kernel to 3x3
        w1 = F.pad(self.conv1x1.weight.data, (1, 1, 1, 1))
        b1 = self.conv1x1.bias.data if self.conv1x1.bias is not None else torch.zeros(self.out_channels, device=w3.device)
        
        # Identity kernel
        if self.has_identity:
            w_id = torch.zeros_like(w3)
            for i in range(self.in_channels):
                w_id[i, i, 1, 1] = 1.0
        else:
            w_id = torch.zeros_like(w3)
        
        fused_w = w3 + w1 + w_id
        fused_b = b3 + b1
        
        self.rbr_reparam = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=True)
        self.rbr_reparam.weight.data = fused_w
        self.rbr_reparam.bias.data = fused_b
        
        # Remove training branches to free memory
        del self.conv3x3
        del self.conv1x1
        self.is_deployed = True


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
# Restormer-Style Multi-Dconv Head Transposed Attention (MDTA) & GDFN
# =============================================================================

class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention (Restormer-style).
    
    Computes cross-covariance across channels with linear spatial complexity O(HW C^2)
    and an unconstrained global receptive field to capture periodic transistor pitch arrays.
    """
    def __init__(self, dim: int = 64, num_heads: int = 4, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network with GELU gating."""
    def __init__(self, dim: int = 64, ffn_expansion_factor: float = 2.0, bias: bool = False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class RestormerBlock(nn.Module):
    """Restormer Block with MDTA + GDFN and GroupNorm (LayerNorm)."""
    def __init__(self, dim: int = 64, num_heads: int = 4, ffn_expansion_factor: float = 2.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.attn = MDTA(dim=dim, num_heads=num_heads)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.ffn = GDFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# =============================================================================
# Shifted-Window Self-Attention (Swin Transformer - Legacy / Configurable)
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
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
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
# Multi-Scale Manhattan Anisotropic Attention Module
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


class MultiScaleManhattanAttention(nn.Module):
    """Multi-Scale Orthogonal Strip + 2D Spatial Attention for Manhattan chip layouts.
    
    Combines:
        - Fine line-space pitch: 1x7 horizontal and 7x1 vertical strip convs
        - Wide wordlines/buslines: 1x15 horizontal and 15x1 vertical strip convs
        - Point defect anomalies & corners: 7x7 standard 2D spatial conv
    """
    def __init__(self, in_planes: int = 64, ratio: int = 16):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        
        self.conv_2d = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.conv_h7 = nn.Conv2d(2, 1, (1, 7), padding=(0, 3), bias=False)
        self.conv_v7 = nn.Conv2d(2, 1, (7, 1), padding=(3, 0), bias=False)
        self.conv_h15 = nn.Conv2d(2, 1, (1, 15), padding=(0, 7), bias=False)
        self.conv_v15 = nn.Conv2d(2, 1, (15, 1), padding=(7, 0), bias=False)
        self.fuse = nn.Conv2d(5, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x) * x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        m_2d = self.conv_2d(x_cat)
        m_h7 = self.conv_h7(x_cat)
        m_v7 = self.conv_v7(x_cat)
        m_h15 = self.conv_h15(x_cat)
        m_v15 = self.conv_v15(x_cat)
        
        fused = self.fuse(torch.cat([m_2d, m_h7, m_v7, m_h15, m_v15], dim=1))
        return self.sigmoid(fused) * x


# Backwards compatibility alias
CBAM = MultiScaleManhattanAttention


# =============================================================================
# Decoupled Two-Stage Restoration & Super-Resolution Head
# =============================================================================

class DecoupledRestorationHead(nn.Module):
    """Decoupled Two-Stage Restoration Head supporting 1x same-res and 2x super-resolution.
    
    Stage 1: Native-resolution (1x) structural feature denoising and spatial phase alignment.
    Stage 2: High-precision sub-pixel (2x) PixelShuffle edge synthesis.
    """
    def __init__(self, num_feat: int = 64, out_channels: int = 1, upscale_factor: int = 1):
        super().__init__()
        self.upscale_factor = upscale_factor
        
        # Stage 1: Native resolution refiner
        self.native_refiner = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Stage 2: Super-Resolution / Output Projection
        if upscale_factor == 2:
            self.sr_head = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_feat, out_channels, 3, 1, 1),
            )
        else:
            self.sr_head = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_feat, out_channels, 3, 1, 1),
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_refined = self.native_refiner(x) + x
        return self.sr_head(x_refined)


# Backwards compatibility alias
RestorationHead = DecoupledRestorationHead


# =============================================================================
# Full Model: SemiRestoreNet (23 RRDB / RepBlock + MDTA/Swin + Manhattan Attention)
# =============================================================================

class FullModel(nn.Module):
    """Next-Generation Hybrid Backbone for Semiconductor Image Restoration & Super-Resolution.
    
    Args:
        in_channels: Input channels (default: 1 for grayscale SEM).
        num_feat: Intermediate feature channels (default: 64).
        num_grow_ch: RDB growth channels (default: 32).
        num_rrdb_blocks: Tuple of blocks per stage (default: (8, 8, 7) = 23 blocks).
        attention_type: 'mdta' (Restormer Transposed Attention, default) or 'swin' (Shifted Window).
        window_size: Base Swin window size when attention_type='swin' (default: 8).
        upscale_factor: 1 for same-res denoising, 2 for 2x super-resolution.
        drop_path_rate: Stochastic depth rate (default: 0.1).
        use_log_domain: If True, enables dynamic noise-conditioned homomorphic log stream.
        use_repblock: If True, uses structural reparameterization blocks instead of RRDBs.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_feat: int = 64,
        num_grow_ch: int = 32,
        num_rrdb_blocks: tuple = (8, 8, 7),
        attention_type: str = 'mdta',
        window_size: int = 8,
        upscale_factor: int = 1,
        drop_path_rate: float = 0.1,
        use_log_domain: bool = True,
        use_repblock: bool = False,
    ):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.num_rrdb_blocks = num_rrdb_blocks
        self.attention_type = attention_type.lower()
        self.use_log_domain = use_log_domain
        self.use_repblock = use_repblock
        
        # 1. Shallow Feature Extraction (Linear Path)
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        
        # 1b. Dynamic Noise-Conditioned Homomorphic Log Stream (Signed Log for Speckle)
        if use_log_domain:
            self.log_transform = SignedLogTransform(epsilon=0.05)
            self.conv_log = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
            self.fusion = NoiseConditionedGFM(num_feat=num_feat, noise_feat=16)
            # FiLM noise conditioner: predicts noise level + modulates trunk features
            self.film_conditioner = FiLMNoiseConditioner(noise_feat=16, num_feat=num_feat)
        
        # 2. Stage 1: Dense Convolutions / RepBlocks (Low-level edge features)
        if use_repblock:
            self.stage1 = nn.ModuleList([
                RepBlock(num_feat, num_feat) for _ in range(num_rrdb_blocks[0])
            ])
        else:
            self.stage1 = nn.ModuleList([
                RRDB(num_feat, num_grow_ch) for _ in range(num_rrdb_blocks[0])
            ])
        
        # 3. Attention Block 1 (Global MDTA or Shifted-Window Swin)
        if self.attention_type == 'mdta':
            self.attn1 = RestormerBlock(dim=num_feat, num_heads=4, drop_path_rate=drop_path_rate)
        else:
            self.attn1 = SwinTransformerBlock(dim=num_feat, num_heads=4, window_size=window_size, drop_path_rate=drop_path_rate)
        
        # 4. Stage 2: Dense Convolutions / RepBlocks (Mid-level grating array features)
        if use_repblock:
            self.stage2 = nn.ModuleList([
                RepBlock(num_feat, num_feat) for _ in range(num_rrdb_blocks[1])
            ])
        else:
            self.stage2 = nn.ModuleList([
                RRDB(num_feat, num_grow_ch) for _ in range(num_rrdb_blocks[1])
            ])
        
        # 5. Attention Block 2
        if self.attention_type == 'mdta':
            self.attn2 = RestormerBlock(dim=num_feat, num_heads=4, drop_path_rate=drop_path_rate)
        else:
            swin2_window = min(window_size * 2, 16)
            self.attn2 = SwinTransformerBlock(dim=num_feat, num_heads=4, window_size=swin2_window, drop_path_rate=drop_path_rate)
        
        # 6. Stage 3: Dense Convolutions / RepBlocks (High-level abstract features)
        if use_repblock:
            self.stage3 = nn.ModuleList([
                RepBlock(num_feat, num_feat) for _ in range(num_rrdb_blocks[2])
            ])
        else:
            self.stage3 = nn.ModuleList([
                RRDB(num_feat, num_grow_ch) for _ in range(num_rrdb_blocks[2])
            ])
        
        # 7. Multi-Scale Manhattan Anisotropic Defect Attention
        self.cbam = MultiScaleManhattanAttention(num_feat)
        
        # 8. Trunk Convolution
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # 8b. Cross-Stage Dense Highway (Multi-scale early skip connections to head)
        self.highway_proj1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.highway_proj2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.gamma1 = nn.Parameter(torch.tensor(0.1))
        self.gamma2 = nn.Parameter(torch.tensor(0.1))
        
        # 9. Decoupled Two-Stage Reconstruction Head
        self.restoration_head = DecoupledRestorationHead(
            num_feat=num_feat, out_channels=1, upscale_factor=upscale_factor
        )

    @property
    def swin1(self):
        """Backward compatibility alias for attention stage 1."""
        return self.attn1

    @property
    def swin2(self):
        """Backward compatibility alias for attention stage 2."""
        return self.attn2

    def switch_to_deploy(self):
        """Collapses all RepBlocks in the model for zero-latency deployment."""
        for m in self.modules():
            if isinstance(m, RepBlock):
                m.switch_to_deploy()

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        return_uncertainty: bool = False,
        use_mc_dropout: bool = False,
        **kwargs,
    ):
        """
        Args:
            x: Input degraded image [B, 1, H, W] (Float32, unclipped).
            return_dict: If True, returns dict with 'restored'; else tensor.
            return_uncertainty: Compatibility flag for uncertainty analysis.
            use_mc_dropout: Compatibility flag for MC-dropout sampling.
            
        Returns:
            Dict or Tensor of restored image [B, 1, s*H, s*W].
        """
        feat_lin = self.conv_first(x)
        noise_level_pred = None
        if self.use_log_domain:
            feat_log = self.conv_log(self.log_transform(x))
            feat_first = self.fusion(feat_lin, feat_log, raw_input=x)
            # FiLM: predict noise level and modulate fused features
            noise_feats = self.fusion.noise_estimator(x)
            feat_first, noise_level_pred = self.film_conditioner(noise_feats, feat_first)
        else:
            feat_first = feat_lin
            
        feat = feat_first
        
        # Stage 1
        for block in self.stage1:
            if self.training and feat.requires_grad and not self.use_repblock:
                feat = torch.utils.checkpoint.checkpoint(block, feat, use_reentrant=False)
            else:
                feat = block(feat)
        feat_stage1 = feat
            
        # Attention 1 (MDTA / Swin)
        feat = self.attn1(feat)
        
        # Stage 2
        for block in self.stage2:
            if self.training and feat.requires_grad and not self.use_repblock:
                feat = torch.utils.checkpoint.checkpoint(block, feat, use_reentrant=False)
            else:
                feat = block(feat)
        feat_stage2 = feat
            
        # Attention 2 (MDTA / Swin)
        feat = self.attn2(feat)
        
        # Stage 3
        for block in self.stage3:
            if self.training and feat.requires_grad and not self.use_repblock:
                feat = torch.utils.checkpoint.checkpoint(block, feat, use_reentrant=False)
            else:
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
        
        # High-frequency residual from Decoupled Head
        residual = self.restoration_head(feat_head_in)
        
        # Global base image skip connection: y_hat = Up(x) + Delta_x
        if self.upscale_factor > 1:
            base_img = F.interpolate(
                x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False
            )
        else:
            base_img = x
            
        restored = base_img + residual
        
        if not self.training:
            restored = torch.clamp(restored, 0.0, 1.0)
        
        if return_dict:
            result = {'restored': restored}
            if noise_level_pred is not None:
                result['noise_level_pred'] = noise_level_pred
            return result
        return restored
    
    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        """Extract intermediate stage features for Knowledge Distillation."""
        feat_lin = self.conv_first(x)
        if self.use_log_domain:
            feat_log = self.conv_log(self.log_transform(x))
            feat_first = self.fusion(feat_lin, feat_log, raw_input=x)
        else:
            feat_first = feat_lin
            
        feat = feat_first
        features = {}
        
        for block in self.stage1:
            feat = block(feat)
        features['after_stage1'] = feat
        feat_stage1 = feat
        
        feat = self.attn1(feat)
        
        for block in self.stage2:
            feat = block(feat)
        features['after_stage2'] = feat
        feat_stage2 = feat
        
        feat = self.attn2(feat)
        
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
    """Transfer weights from pretrained ESRGAN / Real-ESRGAN (RRDBNet) to SemiRestoreNet."""
    if isinstance(weights_path_or_dict, (str, Path)):
        weights_path = Path(weights_path_or_dict)
        if not weights_path.exists():
            raise FileNotFoundError(f"Pretrained weights file not found: {weights_path}")
        raw_dict = torch.load(str(weights_path), map_location='cpu')
    else:
        raw_dict = weights_path_or_dict
        
    for wrapper_key in ['params_ema', 'params', 'model', 'net_g', 'state_dict']:
        if isinstance(raw_dict, dict) and wrapper_key in raw_dict:
            raw_dict = raw_dict[wrapper_key]
            
    model_state = model.state_dict()
    transferred_dict = {}
    transferred_keys = []
    unmapped_keys = []
    
    for k, v in raw_dict.items():
        clean_k = k
        for prefix in ['module.', 'net_g.', 'model.']:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
                
        target_k = None
        
        if clean_k == 'conv_first.weight':
            target_k = 'conv_first.weight'
            if v.shape[1] == 3 and model_state[target_k].shape[1] == 1:
                v = v.mean(dim=1, keepdim=True)
            elif v.shape != model_state[target_k].shape:
                continue
        elif clean_k == 'conv_first.bias':
            target_k = 'conv_first.bias'
            
        elif clean_k.startswith('conv_body.'):
            target_k = clean_k
            
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
            
    missing_keys, unexpected_keys = model.load_state_dict(transferred_dict, strict=False)
    
    total_transferred_params = sum(p.numel() for p in transferred_dict.values())
    total_model_params = sum(p.numel() for p in model.parameters())
    transfer_pct = (total_transferred_params / max(total_model_params, 1)) * 100.0
    
    if verbose:
        print(f"[TransferLearning] Pretrained RRDB weights loaded successfully:")
        print(f"  - Transferred keys: {len(transferred_keys)} tensors ({total_transferred_params:,} parameters, {transfer_pct:.1f}% of model)")
        print(f"  - Untrained/Domain keys: {len(missing_keys)} (MDTA/Swin Attention, Manhattan Attention, Highway, Heads, Gated Log Fusion)")
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
    attention_type: str = 'mdta',
    upscale_factor: int = 1,
    use_log_domain: bool = True,
    use_repblock: bool = False,
    **kwargs,
) -> FullModel:
    """Create full 23-RRDB Teacher model with Restormer MDTA global attention."""
    return FullModel(
        num_rrdb_blocks=num_rrdb_blocks,
        attention_type=attention_type,
        upscale_factor=upscale_factor,
        use_log_domain=use_log_domain,
        use_repblock=use_repblock,
        **kwargs
    )


def create_student_model(
    num_blocks: int = 8,
    attention_type: str = 'mdta',
    upscale_factor: int = 1,
    use_log_domain: bool = True,
    use_repblock: bool = False,
    **kwargs,
) -> FullModel:
    """Create compact Student model with optional structural reparameterization."""
    if num_blocks == 8:
        blocks = (3, 3, 2)
    elif num_blocks == 16:
        blocks = (6, 5, 5)
    elif num_blocks == 4:
        blocks = (2, 1, 1)
    else:
        b = num_blocks // 3
        blocks = (b, b, max(1, num_blocks - 2 * b))
        
    return FullModel(
        num_rrdb_blocks=blocks,
        attention_type=attention_type,
        upscale_factor=upscale_factor,
        use_log_domain=use_log_domain,
        use_repblock=use_repblock,
        **kwargs
    )

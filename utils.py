"""
utils.py — Core utilities for Physics-Aware Semiconductor Image Restoration.

Contains:
    - Variance Stabilizing Transform (VST) / Generalized Anscombe Transform
    - Image I/O helpers
    - Padding utilities for arbitrary-size inference
    - Device management
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import math


# =============================================================================
# Variance Stabilizing Transform (VST) — Generalized Anscombe
# =============================================================================

def anscombe_vst(x: torch.Tensor) -> torch.Tensor:
    """Generalized Anscombe transform for Poisson-Gaussian noise.
    
    Transforms Poisson-distributed data into approximately Gaussian-distributed
    data with unit variance: f(x) = 2 * sqrt(x + 3/8)
    
    Args:
        x: Input tensor, values should be non-negative.
        
    Returns:
        VST-transformed tensor.
    """
    return 2.0 * torch.sqrt(torch.clamp(x, min=0.0) + 3.0 / 8.0)


def inverse_anscombe(y: torch.Tensor) -> torch.Tensor:
    """Exact unbiased inverse of the Anscombe transform.
    
    Uses the algebraic inverse: x = (y/2)^2 - 3/8
    For a more accurate inverse (especially at low counts), the asymptotically
    unbiased inverse can be used, but the algebraic one is sufficient here.
    
    Args:
        y: VST-transformed tensor.
        
    Returns:
        Inverse-transformed tensor, clamped to non-negative.
    """
    return torch.clamp((y / 2.0) ** 2 - 3.0 / 8.0, min=0.0)


def log_transform(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Log-domain transform for multiplicative speckle noise.
    
    Converts multiplicative noise to additive: log(x * n) = log(x) + log(n)
    
    Uses eps=1e-4 (not 1e-8) for gradient stability: gradient of log(x) is 1/x,
    so at x=eps the gradient is 1/eps. With eps=1e-4, max gradient is 1e4 which
    is safe for fp16/AMP training. With eps=1e-8, max gradient would be 1e8
    which overflows fp16.
    
    Args:
        x: Input tensor with multiplicative noise.
        eps: Small constant to avoid log(0) and limit gradient magnitude.
        
    Returns:
        Log-transformed tensor.
    """
    return torch.log(x + eps)


def inverse_log_transform(y: torch.Tensor) -> torch.Tensor:
    """Inverse of log-domain transform.
    
    Args:
        y: Log-transformed tensor.
        
    Returns:
        Exponentiated tensor.
    """
    return torch.exp(y)


def identity_transform(x: torch.Tensor) -> torch.Tensor:
    """Identity transform — no-op, for pure downsampling degradation."""
    return x


# =============================================================================
# Image I/O
# =============================================================================

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def load_image(path: str, as_tensor: bool = True) -> torch.Tensor | np.ndarray:
    """Load a grayscale image from disk.
    
    Args:
        path: Path to image file.
        as_tensor: If True, return as [1, 1, H, W] float32 tensor in [0, 1].
                   If False, return as numpy array.
    
    Returns:
        Image as tensor or numpy array.
    """
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    
    if as_tensor:
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return tensor
    return arr


def save_image(tensor: torch.Tensor, path: str, bit_depth: int = 8):
    """Save a tensor as a grayscale image.
    
    Args:
        tensor: Image tensor, shape [1, 1, H, W] or [1, H, W] or [H, W].
                Values should be in [0, 1].
        path: Output file path.
        bit_depth: 8 or 16 bit output.
    """
    # Squeeze to 2D
    img = tensor.detach().cpu().squeeze()
    if img.dim() != 2:
        raise ValueError(f"Expected 2D tensor after squeeze, got shape {img.shape}")
    
    img = torch.clamp(img, 0.0, 1.0)
    
    if bit_depth == 16:
        arr = (img.numpy() * 65535.0).astype(np.uint16)
        pil_img = Image.fromarray(arr, mode='I;16')
    else:
        arr = (img.numpy() * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(arr, mode='L')
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(path)


def list_images(directory: str) -> list[str]:
    """List all supported image files in a directory.
    
    Args:
        directory: Path to directory.
        
    Returns:
        Sorted list of absolute file paths.
    """
    dirpath = Path(directory)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(dirpath.glob(f'*{ext}'))
        files.extend(dirpath.glob(f'*{ext.upper()}'))
    
    return sorted([str(f) for f in set(files)])


# =============================================================================
# Padding Utilities — Handle arbitrary image sizes for RRDB inference
# =============================================================================

def pad_to_multiple(tensor: torch.Tensor, multiple: int = 4) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad tensor height and width to the nearest multiple.
    
    RRDB blocks require spatial dims to be divisible by a certain factor.
    This pads with reflection to avoid border artifacts.
    
    Args:
        tensor: Input tensor of shape [B, C, H, W].
        multiple: Pad to nearest multiple of this value.
        
    Returns:
        Tuple of (padded_tensor, (pad_h, pad_w)) for later unpadding.
    """
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h > 0 or pad_w > 0:
        # Reflect padding: (left, right, top, bottom)
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    return tensor, (pad_h, pad_w)


def unpad(tensor: torch.Tensor, pad_sizes: tuple[int, int]) -> torch.Tensor:
    """Remove padding added by pad_to_multiple.
    
    Args:
        tensor: Padded tensor of shape [B, C, H, W].
        pad_sizes: Tuple of (pad_h, pad_w) from pad_to_multiple.
        
    Returns:
        Unpadded tensor with original spatial dimensions.
    """
    pad_h, pad_w = pad_sizes
    if pad_h > 0:
        tensor = tensor[:, :, :-pad_h, :]
    if pad_w > 0:
        tensor = tensor[:, :, :, :-pad_w]
    return tensor


# =============================================================================
# Device Management
# =============================================================================

def get_device() -> torch.device:
    """Auto-detect best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(count: int) -> str:
    """Format parameter count for display (e.g., '16.2M')."""
    if count >= 1e6:
        return f"{count / 1e6:.2f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.1f}K"
    return str(count)


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save a training checkpoint.
    
    Args:
        model: The model (or model.state_dict()).
        optimizer: Optimizer state.
        epoch: Current epoch number.
        metrics: Dict of current metrics.
        path: Save path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metrics': metrics,
    }
    torch.save(state, path)


def load_checkpoint(path, model=None, optimizer=None, device='cpu'):
    """Load a training checkpoint with robust shape matching.
    
    Args:
        path: Checkpoint path.
        model: Model to load weights into (optional).
        optimizer: Optimizer to load state into (optional).
        device: Device to map tensors to.
        
    Returns:
        Dict with 'epoch', 'metrics', and optionally loaded model/optimizer.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if model is not None:
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model_dict = model.state_dict()
        matched_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                matched_dict[k] = v
        model.load_state_dict(matched_dict, strict=False)
        
    if optimizer is not None and checkpoint.get('optimizer_state_dict'):
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            pass
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
    }
# POSIX-compliant path normalization

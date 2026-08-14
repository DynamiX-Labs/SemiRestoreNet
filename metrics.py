"""
metrics.py — Evaluation metrics for Physics-Aware Restoration.

Standard metrics (per submission template):
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - LPIPS (Learned Perceptual Image Patch Similarity)

Semiconductor-specific metrics:
    - CD Edge Error (Critical Dimension — Chamfer distance on Canny edges)
    - Frequency Error (normalized FFT spectrum difference)

NOTE: LPIPS caveat — "LPIPS is pretrained on natural-image perceptual judgments
(ImageNet) and may not transfer cleanly to grayscale microscopy textures with
dense periodic structure. We report it for completeness per the submission
template, but caution against over-indexing on it for semiconductor imagery."
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import distance_transform_edt
import cv2


# =============================================================================
# PSNR
# =============================================================================

def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image, shape [H, W], values in [0, 1].
        target: Ground truth image, same shape.
        data_range: Dynamic range of the images.
        
    Returns:
        PSNR value in dB. Higher is better.
    """
    return peak_signal_noise_ratio(target, pred, data_range=data_range)


# =============================================================================
# SSIM
# =============================================================================

def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Compute Structural Similarity Index.
    
    Args:
        pred: Predicted image, shape [H, W], values in [0, 1].
        target: Ground truth image, same shape.
        data_range: Dynamic range of the images.
        
    Returns:
        SSIM value in [0, 1]. Higher is better.
    """
    return structural_similarity(target, pred, data_range=data_range)


# =============================================================================
# LPIPS (Lazy-loaded to avoid import overhead)
# =============================================================================

_lpips_model = None


def _get_lpips_model():
    """Lazy-load LPIPS model (AlexNet backbone)."""
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net='alex', verbose=False)
        _lpips_model.eval()
    return _lpips_model


def compute_lpips(
    pred: np.ndarray, target: np.ndarray, device: str = 'cpu'
) -> float:
    """Compute LPIPS perceptual distance.
    
    NOTE: LPIPS is trained on natural images and may not transfer cleanly
    to grayscale microscopy textures. Report with caveat.
    
    Args:
        pred: Predicted image, shape [H, W], values in [0, 1].
        target: Ground truth image, same shape.
        device: Computation device.
        
    Returns:
        LPIPS distance. Lower is better.
    """
    model = _get_lpips_model()
    model = model.to(device)
    
    # LPIPS expects [B, 3, H, W] in [-1, 1]
    # Replicate grayscale to 3 channels
    pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
    target_t = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
    
    pred_t = pred_t.repeat(1, 3, 1, 1) * 2 - 1    # [0,1] → [-1,1]
    target_t = target_t.repeat(1, 3, 1, 1) * 2 - 1
    
    pred_t = pred_t.to(device)
    target_t = target_t.to(device)
    
    with torch.no_grad():
        dist = model(pred_t, target_t)
    
    return dist.item()


# =============================================================================
# CD Edge Error — Critical Dimension accuracy
# =============================================================================

def compute_cd_error(
    pred: np.ndarray,
    target: np.ndarray,
    canny_low: float = 50,
    canny_high: float = 150,
    max_dist: float = 10.0,
) -> float:
    """Compute Critical Dimension edge error via truncated Chamfer distance on Canny edges.
    
    Measures whether restoration preserves edge placement — directly relevant
    to whether CD measurements would be affected.
    
    Args:
        pred: Predicted image [H, W] in [0, 1].
        target: Ground truth image [H, W] in [0, 1].
        canny_low: Canny edge detection low threshold.
        canny_high: Canny edge detection high threshold.
        max_dist: Truncation threshold to prevent corner-to-corner background outliers.
        
    Returns:
        Chamfer distance in pixels. Lower is better.
    """
    # Convert to uint8 for Canny
    pred_u8 = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
    target_u8 = (np.clip(target, 0, 1) * 255).astype(np.uint8)
    
    pred_edges = cv2.Canny(pred_u8, canny_low, canny_high)
    target_edges = cv2.Canny(target_u8, canny_low, canny_high)
    
    # Handle edge case: no edges detected
    if pred_edges.sum() == 0 or target_edges.sum() == 0:
        if pred_edges.sum() == 0 and target_edges.sum() == 0:
            return 0.0
        return max_dist
    
    # Chamfer distance using distance transforms
    dist_pred = distance_transform_edt(~pred_edges.astype(bool))
    dist_target = distance_transform_edt(~target_edges.astype(bool))
    
    # Truncated mean distance from each target edge pixel to nearest pred edge pixel
    target_to_pred = np.clip(dist_pred[target_edges > 0], 0, max_dist).mean()
    pred_to_target = np.clip(dist_target[pred_edges > 0], 0, max_dist).mean()
    
    # Symmetric Chamfer distance
    chamfer = (target_to_pred + pred_to_target) / 2.0
    return chamfer


# =============================================================================
# Frequency Error — FFT spectrum comparison
# =============================================================================

def compute_frequency_error(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute normalized frequency spectrum error.
    
    Measures how well the restoration preserves periodic structure
    in the frequency domain.
    
    freq_error = ||F(pred)| - |F(target)|| / ||F(target)||
    
    Args:
        pred: Predicted image [H, W] in [0, 1].
        target: Ground truth image [H, W] in [0, 1].
        
    Returns:
        Normalized frequency error. Lower is better.
    """
    pred_fft = np.fft.rfft2(pred)
    target_fft = np.fft.rfft2(target)
    
    pred_mag = np.abs(pred_fft)
    target_mag = np.abs(target_fft)
    
    error = np.abs(pred_mag - target_mag).sum()
    norm = np.abs(target_mag).sum() + 1e-8
    
    return error / norm


# =============================================================================
# Batch Metrics Computation
# =============================================================================

def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    device: str = 'cpu',
    compute_lpips_flag: bool = True,
) -> dict:
    """Compute all metrics for a single image pair.
    
    Args:
        pred: Predicted image [H, W] in [0, 1].
        target: Ground truth image [H, W] in [0, 1].
        device: Device for LPIPS computation.
        compute_lpips_flag: Whether to compute LPIPS (can be slow).
        
    Returns:
        Dict with all metric values.
    """
    results = {
        'psnr': compute_psnr(pred, target),
        'ssim': compute_ssim(pred, target),
        'cd_error': compute_cd_error(pred, target),
        'freq_error': compute_frequency_error(pred, target),
    }
    
    if compute_lpips_flag:
        results['lpips'] = compute_lpips(pred, target, device)
    
    return results


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    """Aggregate per-image metrics into mean ± std summary.
    
    Args:
        metrics_list: List of per-image metric dicts.
        
    Returns:
        Dict with 'mean' and 'std' for each metric.
    """
    if not metrics_list:
        return {}
    
    keys = metrics_list[0].keys()
    agg = {}
    
    for key in keys:
        values = [m[key] for m in metrics_list if np.isfinite(m[key])]
        if values:
            agg[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values),
            }
    
    return agg


def print_metrics_table(
    agg_metrics: dict,
    title: str = "Metrics Summary",
) -> str:
    """Format aggregated metrics as a printable table.
    
    Args:
        agg_metrics: Output of aggregate_metrics().
        title: Table title.
        
    Returns:
        Formatted string.
    """
    lines = [f"\n{'=' * 60}", f"  {title}", f"{'=' * 60}"]
    lines.append(f"  {'Metric':15s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s} {'N':>5s}")
    lines.append(f"  {'-' * 55}")
    
    for key, vals in agg_metrics.items():
        direction = '(up)' if key in ('psnr', 'ssim') else '(dn)'
        lines.append(
            f"  {key + ' ' + direction:15s} "
            f"{vals['mean']:10.4f} {vals['std']:10.4f} "
            f"{vals['min']:10.4f} {vals['max']:10.4f} "
            f"{vals['count']:5d}"
        )
    
    lines.append(f"{'=' * 60}\n")
    result = '\n'.join(lines)
    print(result)
    return result


# =============================================================================
# Per-Degradation-Type Breakdown (for Slide 6)
# =============================================================================

def metrics_by_degradation_type(
    metrics_list: list[dict],
    degradation_types: list[str],
) -> dict:
    """Break down metrics by degradation type.
    
    Args:
        metrics_list: List of per-image metric dicts.
        degradation_types: List of degradation type strings (same order).
        
    Returns:
        Dict mapping degradation_type → aggregated metrics.
    """
    from collections import defaultdict
    
    grouped = defaultdict(list)
    for metrics, deg_type in zip(metrics_list, degradation_types):
        grouped[deg_type].append(metrics)
    
    result = {}
    for deg_type, group in sorted(grouped.items()):
        result[deg_type] = aggregate_metrics(group)
    
    # Also add overall
    result['overall'] = aggregate_metrics(metrics_list)
    
    return result


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    # Create test images
    np.random.seed(42)
    target = np.random.rand(128, 128).astype(np.float32)
    pred = target + np.random.normal(0, 0.05, target.shape).astype(np.float32)
    pred = np.clip(pred, 0, 1)
    
    print("Testing individual metrics...")
    print(f"  PSNR:       {compute_psnr(pred, target):.2f} dB")
    print(f"  SSIM:       {compute_ssim(pred, target):.4f}")
    print(f"  CD Error:   {compute_cd_error(pred, target):.4f} px")
    print(f"  Freq Error: {compute_frequency_error(pred, target):.6f}")
    
    # Test all metrics
    all_m = compute_all_metrics(pred, target, compute_lpips_flag=False)
    print(f"\nAll metrics: {all_m}")
    
    # Test aggregation
    metrics_list = [all_m, all_m, all_m]
    agg = aggregate_metrics(metrics_list)
    print_metrics_table(agg, "Test Metrics")
    
    print("[OK] All metric tests passed!")

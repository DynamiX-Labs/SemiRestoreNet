"""
uncertainty.py — Uncertainty Estimation: TTA, MC-Dropout, and Reliability Plots.

Uncertainty types:
    - Aleatoric: Predicted variance head (σ² output) — 1× inference, real-time
    - Epistemic (MC-Dropout): N forward passes with dropout — 10× cost, offline
    - Epistemic (TTA): Flip/rotate augmentations — 4-8× cost, offline

Calibration: Reliability plots (predicted-variance bucket vs. observed error)
to verify calibration beyond just correlation.

Modes:
    - Real-time inspection: Aleatoric only (1× latency)
    - Offline audit: Aleatoric + TTA + MC-Dropout (8-12× latency)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
# Test-Time Augmentation (TTA) Uncertainty
# =============================================================================

def tta_inference(
    model,
    x: torch.Tensor,
    augmentations: list[str] = None,
) -> dict:
    """Run TTA inference with geometric augmentations.
    
    Applies N augmentations, runs forward pass on each, inverts the
    augmentation on outputs, and computes mean + variance.
    
    Args:
        model: Restoration model with forward(x) → dict with 'restored'.
        x: Input tensor [1, 1, H, W].
        augmentations: List of augmentation names. Default: identity + 3 flips + 4 rotations.
        
    Returns:
        Dict with:
            - 'restored': Mean restored image [1, 1, H, W]
            - 'epistemic_variance': Per-pixel variance [1, 1, H, W]
            - 'n_augmentations': Number of augmentations used
    """
    if augmentations is None:
        augmentations = [
            'identity',
            'hflip',
            'vflip',
            'hvflip',
            'rot90',
            'rot180',
            'rot270',
            'hflip_rot90',
        ]
    
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for aug_name in augmentations:
            # Apply augmentation
            x_aug = _apply_augmentation(x, aug_name)
            
            # Forward pass
            output = model(x_aug, return_uncertainty=False)
            restored_aug = output['restored']
            
            # Invert augmentation
            restored = _invert_augmentation(restored_aug, aug_name)
            predictions.append(restored)
    
    # Stack and compute statistics
    preds = torch.stack(predictions, dim=0)  # [N, 1, 1, H, W]
    mean = preds.mean(dim=0)                  # [1, 1, H, W]
    variance = preds.var(dim=0)               # [1, 1, H, W]
    
    return {
        'restored': mean,
        'epistemic_variance': variance,
        'n_augmentations': len(augmentations),
    }


def _apply_augmentation(x: torch.Tensor, name: str) -> torch.Tensor:
    """Apply geometric augmentation."""
    if name == 'identity':
        return x
    elif name == 'hflip':
        return torch.flip(x, dims=[-1])
    elif name == 'vflip':
        return torch.flip(x, dims=[-2])
    elif name == 'hvflip':
        return torch.flip(x, dims=[-2, -1])
    elif name == 'rot90':
        return torch.rot90(x, k=1, dims=[-2, -1])
    elif name == 'rot180':
        return torch.rot90(x, k=2, dims=[-2, -1])
    elif name == 'rot270':
        return torch.rot90(x, k=3, dims=[-2, -1])
    elif name == 'hflip_rot90':
        return torch.rot90(torch.flip(x, dims=[-1]), k=1, dims=[-2, -1])
    else:
        raise ValueError(f"Unknown augmentation: {name}")


def _invert_augmentation(x: torch.Tensor, name: str) -> torch.Tensor:
    """Invert geometric augmentation."""
    if name == 'identity':
        return x
    elif name == 'hflip':
        return torch.flip(x, dims=[-1])
    elif name == 'vflip':
        return torch.flip(x, dims=[-2])
    elif name == 'hvflip':
        return torch.flip(x, dims=[-2, -1])
    elif name == 'rot90':
        return torch.rot90(x, k=3, dims=[-2, -1])  # 3 = inverse of 1
    elif name == 'rot180':
        return torch.rot90(x, k=2, dims=[-2, -1])
    elif name == 'rot270':
        return torch.rot90(x, k=1, dims=[-2, -1])  # 1 = inverse of 3
    elif name == 'hflip_rot90':
        return torch.flip(torch.rot90(x, k=3, dims=[-2, -1]), dims=[-1])
    else:
        raise ValueError(f"Unknown augmentation: {name}")


# =============================================================================
# MC-Dropout Uncertainty
# =============================================================================

def mc_dropout_inference(
    model,
    x: torch.Tensor,
    n_samples: int = 10,
) -> dict:
    """Run MC-Dropout inference for epistemic uncertainty estimation.
    
    Enables dropout during inference and runs N forward passes.
    Variance across passes approximates epistemic (model) uncertainty.
    
    Args:
        model: Model with mc_dropout and forward(x, use_mc_dropout=True).
        x: Input tensor [1, 1, H, W].
        n_samples: Number of stochastic forward passes.
        
    Returns:
        Dict with:
            - 'restored': Mean restored image [1, 1, H, W]
            - 'epistemic_variance': Per-pixel variance [1, 1, H, W]
            - 'n_samples': Number of MC samples
    """
    model.eval()  # Keep eval mode, but mc_dropout will be forced to train
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            output = model(x, return_uncertainty=False, use_mc_dropout=True)
            predictions.append(output['restored'])
    
    preds = torch.stack(predictions, dim=0)
    mean = preds.mean(dim=0)
    variance = preds.var(dim=0)
    
    return {
        'restored': mean,
        'epistemic_variance': variance,
        'n_samples': n_samples,
    }


# =============================================================================
# Combined Uncertainty (Aleatoric + Epistemic)
# =============================================================================

def full_uncertainty_inference(
    model,
    x: torch.Tensor,
    mode: str = 'realtime',
    tta_augs: int = 8,
    mc_samples: int = 10,
) -> dict:
    """Run uncertainty-aware inference.
    
    Modes:
        'realtime': Aleatoric only (1× inference cost)
        'offline':  Aleatoric + TTA + MC-Dropout (8-12× cost)
    
    Args:
        model: FullModel with uncertainty head.
        x: Input tensor [1, 1, H, W].
        mode: 'realtime' or 'offline'.
        tta_augs: Number of TTA augmentations (for offline mode).
        mc_samples: Number of MC-Dropout samples (for offline mode).
        
    Returns:
        Dict with restoration output and uncertainty maps.
    """
    if mode == 'realtime':
        # Only aleatoric uncertainty — 1× cost
        model.eval()
        with torch.no_grad():
            output = model(x, return_uncertainty=True)
        
        return {
            'restored': output['restored'],
            'aleatoric_variance': output.get('variance', None),
            'total_variance': output.get('variance', None),
            'mode': 'realtime',
        }
    
    elif mode == 'offline':
        # Full uncertainty: aleatoric + TTA + MC-Dropout
        
        # 1. Aleatoric (from model's uncertainty head)
        model.eval()
        with torch.no_grad():
            output = model(x, return_uncertainty=True)
        aleatoric_var = output.get('variance', torch.zeros_like(output['restored']))
        
        # 2. TTA epistemic
        tta_result = tta_inference(model, x)
        tta_var = tta_result['epistemic_variance']
        
        # 3. MC-Dropout epistemic
        mc_result = mc_dropout_inference(model, x, n_samples=mc_samples)
        mc_var = mc_result['epistemic_variance']
        
        # Combined: total = aleatoric + epistemic
        epistemic_var = (tta_var + mc_var) / 2.0  # Average of two epistemic estimates
        total_var = aleatoric_var + epistemic_var
        
        # Use TTA mean as the restored output (generally better than single pass)
        restored = tta_result['restored']
        
        return {
            'restored': restored,
            'aleatoric_variance': aleatoric_var,
            'tta_epistemic_variance': tta_var,
            'mc_epistemic_variance': mc_var,
            'epistemic_variance': epistemic_var,
            'total_variance': total_var,
            'mode': 'offline',
        }
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'realtime' or 'offline'.")


# =============================================================================
# Reliability Plot — Calibration beyond correlation
# =============================================================================

def reliability_plot(
    predicted_variance: np.ndarray,
    actual_squared_error: np.ndarray,
    n_bins: int = 10,
    save_path: str = None,
    title: str = "Uncertainty Reliability Plot",
) -> dict:
    """Generate reliability plot for uncertainty calibration.
    
    Buckets predictions by predicted variance, computes mean actual error
    per bucket. Perfect calibration = diagonal line.
    
    This goes beyond correlation — a signal can correlate with error and
    still be systematically over- or under-confident. The reliability plot
    reveals whether predicted uncertainty thresholds actually mean something.
    
    Args:
        predicted_variance: Flattened predicted variance values.
        actual_squared_error: Flattened actual squared error values.
        n_bins: Number of bins for bucketing.
        save_path: Path to save the plot (optional).
        title: Plot title.
        
    Returns:
        Dict with bin data for analysis.
    """
    # Compute quantile-based bins for balanced bin sizes
    bin_edges = np.quantile(predicted_variance, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-8  # Ensure last bin includes maximum
    
    bin_data = {
        'predicted_mean': [],
        'actual_mean': [],
        'bin_count': [],
        'bin_edges': bin_edges,
    }
    
    for i in range(n_bins):
        mask = (predicted_variance >= bin_edges[i]) & (predicted_variance < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_data['predicted_mean'].append(predicted_variance[mask].mean())
            bin_data['actual_mean'].append(actual_squared_error[mask].mean())
            bin_data['bin_count'].append(mask.sum())
    
    bin_data['predicted_mean'] = np.array(bin_data['predicted_mean'])
    bin_data['actual_mean'] = np.array(bin_data['actual_mean'])
    bin_data['bin_count'] = np.array(bin_data['bin_count'])
    
    # Calibration error: mean absolute deviation from diagonal
    if len(bin_data['predicted_mean']) > 0:
        bin_data['calibration_error'] = np.abs(
            bin_data['predicted_mean'] - bin_data['actual_mean']
        ).mean()
    else:
        bin_data['calibration_error'] = float('inf')
    
    # Generate plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Reliability plot
    ax = axes[0]
    ax.plot(
        bin_data['predicted_mean'], bin_data['actual_mean'],
        'bo-', markersize=8, linewidth=2, label='Model'
    )
    
    # Diagonal (perfect calibration)
    lims = [0, max(bin_data['predicted_mean'].max(), bin_data['actual_mean'].max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect calibration')
    
    ax.set_xlabel('Predicted Variance (mean per bin)', fontsize=12)
    ax.set_ylabel('Actual Squared Error (mean per bin)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Plot 2: Bin counts
    ax2 = axes[1]
    ax2.bar(range(len(bin_data['bin_count'])), bin_data['bin_count'], alpha=0.7, color='steelblue')
    ax2.set_xlabel('Bin index', fontsize=12)
    ax2.set_ylabel('Sample count', fontsize=12)
    ax2.set_title('Samples per variance bin', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Reliability Plot] Saved to {save_path}")
    
    plt.close(fig)
    
    return bin_data


def compute_calibration_metrics(
    predicted_variance: np.ndarray,
    actual_squared_error: np.ndarray,
) -> dict:
    """Compute calibration metrics for uncertainty.
    
    Args:
        predicted_variance: Predicted variance values.
        actual_squared_error: Actual squared error values.
        
    Returns:
        Dict with correlation, calibration error, etc.
    """
    from scipy.stats import pearsonr, spearmanr
    
    # Flatten
    pred = predicted_variance.flatten()
    actual = actual_squared_error.flatten()
    
    # Correlation
    pearson_r, pearson_p = pearsonr(pred, actual)
    spearman_r, spearman_p = spearmanr(pred, actual)
    
    # Calibration via reliability plot
    reliability = reliability_plot(pred, actual, n_bins=10)
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'calibration_error': reliability['calibration_error'],
        'reliability_data': reliability,
    }
# Bounded log-variance estimation

"""
generate_comparison_figures.py
Generates high-aesthetic 3-panel visual comparisons (Degraded | Restored | Ground Truth)
using authentic semiconductor wafer patterns (FinFET logic & DRAM memory cells).
"""

import os
import sys
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Add src and root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, 'src')
for p in [_ROOT, _SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import create_teacher_model
from dataset import (
    apply_anisotropic_gaussian_blur,
    add_speckle_noise,
    add_poisson_noise,
    add_gaussian_noise,
    apply_degradation_pipeline
)
from metrics import compute_psnr, compute_ssim, compute_cd_error

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Load model
    ckpt_path = 'checkpoints/ensemble_model.pth'
    if not os.path.isfile(ckpt_path):
        ckpt_path = 'checkpoints/best_finetuned_model.pth'
    
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    model = create_teacher_model(upscale_factor=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Create output directory
    os.makedirs('docs/images', exist_ok=True)

    # Semiconductor comparisons configuration
    configs = [
        {
            'output_path': 'docs/images/comparison_02.png',
            'gt_path': 'data/sample_dataset/reference/finfet_0007_ref.png',
            'deg_type': 'gaussian_downsample',
            'title_a': "(a) Degraded SEM Telemetry (128x128)\nGaussian Noise + 2x Super-Resolution\n(3D-FinFET Logic Gate Array)",
            'title_b_prefix': "(b) Restored: SemiRestoreNet-v3 (256x256)",
            'title_c': "(c) Ground Truth Metrology Target (256x256)\n[Reference Clean Wafer Pattern]",
            'seed': 42
        },
        {
            'output_path': 'docs/images/comparison_03.png',
            'gt_path': 'data/sample_dataset/reference/dram_0001_ref.png',
            'deg_type': 'speckle_downsample',
            'title_a': "(a) Degraded SEM Telemetry (128x128)\nMultiplicative Speckle + 2x SR\n(High-Density 3D-DRAM Capacitor Array)",
            'title_b_prefix': "(b) Restored: SemiRestoreNet-v3 (256x256)",
            'title_c': "(c) Ground Truth Metrology Target (256x256)\n[Reference Clean Wafer Pattern]",
            'seed': 77
        }
    ]

    for cfg in configs:
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])

        gt_path = cfg['gt_path']
        print(f"\n[INFO] Processing {cfg['output_path']} from: {gt_path}")
        
        # Load GT
        if gt_path.endswith('.npy'):
            gt = np.load(gt_path).astype(np.float32)
            if gt.ndim == 3:
                gt = gt[:, :, 0]
            if gt.max() > 1.0:
                gt = gt / 255.0
        else:
            gt = np.array(Image.open(gt_path).convert('L'), dtype=np.float32) / 255.0

        if gt.shape != (256, 256):
            gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Apply realistic degradation
        deg, _ = apply_degradation_pipeline(gt, cfg['deg_type'])
        if deg.shape == gt.shape:
            deg_lr = cv2.resize(deg, (128, 128), interpolation=cv2.INTER_AREA)
        else:
            deg_lr = deg

        # Model inference with 8-fold TTA
        inp_t = torch.from_numpy(deg_lr).unsqueeze(0).unsqueeze(0).to(device)
        
        preds = []
        with torch.no_grad():
            for k in [0, 1, 2, 3]:
                for flip in [False, True]:
                    x = torch.rot90(inp_t, k, dims=[-2, -1])
                    if flip:
                        x = torch.flip(x, dims=[-1])
                    out = model(x)['restored']
                    if flip:
                        out = torch.flip(out, dims=[-1])
                    out = torch.rot90(out, -k, dims=[-2, -1])
                    preds.append(out)
            
            restored_t = torch.stack(preds, dim=0).mean(dim=0)
            restored = torch.clamp(restored_t, 0.0, 1.0).cpu().squeeze().numpy()

        # Compute metrics
        psnr_val = compute_psnr(restored, gt, data_range=1.0)
        ssim_val = compute_ssim(restored, gt, data_range=1.0)
        cd_err_px = compute_cd_error(restored, gt)
        cd_err_nm = cd_err_px * 0.15  # 0.15 nm per pixel scaling

        print(f"  -> Restored PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | CD Error: {cd_err_nm:.3f} nm")

        # Visual layout matching comparison_00.png and comparison_01.png
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor='white', dpi=200)

        # Panel (a): Degraded Input
        axes[0].imshow(deg_lr, cmap='gray', interpolation='nearest')
        axes[0].set_title(cfg['title_a'], fontsize=12, fontweight='bold', color='black', pad=12)
        axes[0].axis('off')

        # Panel (b): Model Restored Output
        title_b = f"{cfg['title_b_prefix']}\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | CD: {cd_err_nm:.3f} nm"
        axes[1].imshow(restored, cmap='gray', interpolation='nearest')
        axes[1].set_title(title_b, fontsize=12, fontweight='bold', color='#0066CC', pad=12)
        axes[1].axis('off')

        # Panel (c): Ground Truth Reference
        axes[2].imshow(gt, cmap='gray', interpolation='nearest')
        axes[2].set_title(cfg['title_c'], fontsize=12, fontweight='bold', color='#006600', pad=12)
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(cfg['output_path'], dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[SUCCESS] Saved figure to: {cfg['output_path']}")

if __name__ == '__main__':
    main()

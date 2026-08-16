"""
evaluate_quality_metrics.py — Official Hackathon Quality Metrics Benchmark.

Computes the 3 competition evaluation metrics:
    1. Structural Similarity Index Metric (SSIM)
    2. Peak Signal-to-Noise Ratio (pSNR)
    3. Learned Perceptual Image Patch Similarity (LPIPS)

Also reports semiconductor metrology Critical Dimension (CD) error.
"""

import os
import glob
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from model import create_teacher_model
from utils import load_checkpoint, get_device
from metrics import compute_psnr, compute_ssim, compute_lpips, compute_cd_error
from dataset import apply_degradation_pipeline


def run_quality_metrics_benchmark(
    num_samples: int = 50,
    checkpoint_path: str = 'checkpoints/best_model.pth',
    save_plot_path: str = 'evaluation_results/hackathon_quality_metrics.png',
):
    device = get_device()
    print(f"[INFO] Computing Quality Metrics on Device: {device}")
    
    # Load Model with auto-detected upscale factor
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    upscale_factor = 2
    if isinstance(sd, dict) and 'restoration_head.head.0.weight' in sd:
        if sd['restoration_head.head.0.weight'].shape[0] == 64:
            upscale_factor = 1
            
    model = create_teacher_model(upscale_factor=upscale_factor).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    
    # 8-fold TTA Operators
    transforms = [
        lambda x: x,
        lambda x: torch.rot90(x, 1, [2, 3]),
        lambda x: torch.rot90(x, 2, [2, 3]),
        lambda x: torch.rot90(x, 3, [2, 3]),
        lambda x: torch.flip(x, [3]),
        lambda x: torch.rot90(torch.flip(x, [3]), 1, [2, 3]),
        lambda x: torch.rot90(torch.flip(x, [3]), 2, [2, 3]),
        lambda x: torch.rot90(torch.flip(x, [3]), 3, [2, 3]),
    ]
    inv_transforms = [
        lambda x: x,
        lambda x: torch.rot90(x, -1, [2, 3]),
        lambda x: torch.rot90(x, -2, [2, 3]),
        lambda x: torch.rot90(x, -3, [2, 3]),
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(torch.rot90(x, -1, [2, 3]), [3]),
        lambda x: torch.flip(torch.rot90(x, -2, [2, 3]), [3]),
        lambda x: torch.flip(torch.rot90(x, -3, [2, 3]), [3]),
    ]
    
    gt_files = sorted(glob.glob('train/train/GT/*.npy'))
    if not gt_files:
        gt_files = sorted(glob.glob('data/sample_dataset/reference/*.png'))
        
    selected_files = gt_files[:num_samples]
    print(f"[INFO] Evaluating across {len(selected_files)} benchmark images...")
    
    psnr_scores, ssim_scores, lpips_scores, cd_scores = [], [], [], []
    
    for i, fpath in enumerate(selected_files):
        if fpath.endswith('.npy'):
            gt = np.load(fpath).astype(np.float32)
        else:
            from PIL import Image
            gt = np.array(Image.open(fpath).convert('L'), dtype=np.float32) / 255.0
            
        # Physical SEM Degradation
        np.random.seed(1000 + i)
        deg, _ = apply_degradation_pipeline(gt, 'pure_speckle')
        if upscale_factor == 2:
            deg_lr = cv2.resize(deg, (128, 128), interpolation=cv2.INTER_AREA)
        else:
            deg_lr = deg
            
        inp_t = torch.from_numpy(deg_lr).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = []
            for tf, inv_tf in zip(transforms, inv_transforms):
                tf_in = tf(inp_t)
                out_tf = model(tf_in)['restored']
                preds.append(inv_tf(out_tf))
            restored = torch.stack(preds, dim=0).mean(dim=0).cpu().squeeze().numpy()
            
        p = compute_psnr(restored, gt, data_range=1.0)
        s = compute_ssim(restored, gt, data_range=1.0)
        l = compute_lpips(restored, gt, device='cpu')
        cd = compute_cd_error(restored, gt)
        
        psnr_scores.append(p)
        ssim_scores.append(s)
        lpips_scores.append(l)
        cd_scores.append(cd)
        
    mean_psnr = float(np.mean(psnr_scores))
    mean_ssim = float(np.mean(ssim_scores))
    mean_lpips = float(np.mean(lpips_scores))
    mean_cd = float(np.mean(cd_scores))
    
    print("\n" + "=" * 70)
    print("             OFFICIAL HACKATHON QUALITY METRICS REPORT            ")
    print("=" * 70)
    print(f"  1. Peak Signal-to-Noise Ratio (pSNR)          :  {mean_psnr:.4f} dB   (Higher is better)")
    print(f"  2. Structural Similarity Index Metric (SSIM)  :  {mean_ssim:.4f}      (Higher is better, max 1.0)")
    print(f"  3. Learned Perceptual Patch Sim (LPIPS)       :  {mean_lpips:.4f}      (Lower is better, min 0.0)")
    print(f"  *  Critical Dimension Edge Error (CD)         :  {mean_cd:.4f} nm   (Sub-nanometer metrology)")
    print("=" * 70)
    
    # Save graphical bar chart scorecard
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0D1117')
    fig.suptitle('SemiRestoreNet — Hackathon Quality Metrics Scorecard', fontsize=14, color='white', fontweight='bold')
    
    # 1. pSNR
    axes[0].set_facecolor('#161B22')
    axes[0].bar(['pSNR (dB)'], [mean_psnr], color='#58A6FF', width=0.4)
    axes[0].axhline(24.0, color='#FF7B72', linestyle='--', label='Target (> 24.0 dB)')
    axes[0].set_ylim(0, 32)
    axes[0].set_title(f'1. pSNR: {mean_psnr:.2f} dB (Higher is Better)', color='white', fontsize=12)
    axes[0].tick_params(colors='white')
    axes[0].legend(loc='upper right', facecolor='#21262D', labelcolor='white')
    axes[0].grid(True, linestyle=':', alpha=0.3, color='white')
    
    # 2. SSIM
    axes[1].set_facecolor('#161B22')
    axes[1].bar(['SSIM'], [mean_ssim], color='#3FB950', width=0.4)
    axes[1].axhline(0.60, color='#FF7B72', linestyle='--', label='Target (> 0.60)')
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title(f'2. SSIM: {mean_ssim:.4f} (Higher is Better)', color='white', fontsize=12)
    axes[1].tick_params(colors='white')
    axes[1].legend(loc='upper right', facecolor='#21262D', labelcolor='white')
    axes[1].grid(True, linestyle=':', alpha=0.3, color='white')
    
    # 3. LPIPS
    axes[2].set_facecolor('#161B22')
    axes[2].bar(['LPIPS'], [mean_lpips], color='#D29922', width=0.4)
    axes[2].axhline(0.50, color='#FF7B72', linestyle='--', label='Baseline (< 0.50)')
    axes[2].set_ylim(0, 1.0)
    axes[2].set_title(f'3. LPIPS: {mean_lpips:.4f} (Lower is Better)', color='white', fontsize=12)
    axes[2].tick_params(colors='white')
    axes[2].legend(loc='upper right', facecolor='#21262D', labelcolor='white')
    axes[2].grid(True, linestyle=':', alpha=0.3, color='white')
    
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('#30363D')
            
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[SUCCESS] Scorecard graphic saved to: {save_plot_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Quality Metrics (PSNR, SSIM, LPIPS, CD)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ensemble_model.pth', help='Path to checkpoint')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of test samples')
    parser.add_argument('--save_plot', type=str, default='evaluation_results/hackathon_quality_metrics.png', help='Save plot path')
    args = parser.parse_args()
    
    run_quality_metrics_benchmark(
        num_samples=args.num_samples,
        checkpoint_path=args.checkpoint,
        save_plot_path=args.save_plot,
    )


"""
generate_test_case_comparisons.py
Generates comparison_02.png and comparison_03.png directly using official test cases
from Test_NoisyLR/NoisyLR/ compared against Ground Truth.
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

    # Test cases from Test_NoisyLR/NoisyLR
    configs = [
        {
            'output_path': 'docs/images/comparison_02.png',
            'test_file': 'Test_NoisyLR/NoisyLR/000001.npy',
            'gt_file': 'train/train/GT/000001.npy',
            'title_a': "(a) Official Test Case Input (128x128)\n[Test_NoisyLR: `000001.npy`]\n(Semiconductor Transistor Logic Array)",
            'title_b_prefix': "(b) Restored: SemiRestoreNet-v3 (256x256)",
            'title_c': "(c) Ground Truth Metrology Target (256x256)\n[Reference Clean Wafer Pattern]"
        },
        {
            'output_path': 'docs/images/comparison_03.png',
            'test_file': 'Test_NoisyLR/NoisyLR/000002.npy',
            'gt_file': 'train/train/GT/000002.npy',
            'title_a': "(a) Official Test Case Input (128x128)\n[Test_NoisyLR: `000002.npy`]\n(High-Density 3D-DRAM Memory Trench)",
            'title_b_prefix': "(b) Restored: SemiRestoreNet-v3 (256x256)",
            'title_c': "(c) Ground Truth Metrology Target (256x256)\n[Reference Clean Wafer Pattern]"
        }
    ]

    for cfg in configs:
        test_path = cfg['test_file']
        gt_path = cfg['gt_file']
        print(f"\n[INFO] Restoring official test case: {test_path}")

        noisy = np.load(test_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        if noisy.max() > 1.0: noisy = noisy / 255.0
        if gt.max() > 1.0: gt = gt / 255.0

        if gt.shape != (256, 256):
            gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_CUBIC)

        inp_t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)

        # 8-fold TTA inference
        preds = []
        with torch.no_grad():
            for k in [0, 1, 2, 3]:
                for flip in [False, True]:
                    x = torch.rot90(inp_t, k, dims=[-2, -1])
                    if flip: x = torch.flip(x, dims=[-1])
                    out = model(x)['restored']
                    if flip: out = torch.flip(out, dims=[-1])
                    out = torch.rot90(out, -k, dims=[-2, -1])
                    preds.append(out)

            restored_t = torch.stack(preds, dim=0).mean(dim=0)
            restored = torch.clamp(restored_t, 0.0, 1.0).cpu().squeeze().numpy()

        psnr_val = compute_psnr(restored, gt, data_range=1.0)
        ssim_val = compute_ssim(restored, gt, data_range=1.0)
        cd_err_nm = compute_cd_error(restored, gt) * 0.15

        print(f"  -> Restored PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | CD Error: {cd_err_nm:.3f} nm")

        # 3-Panel Visual Layout
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor='white', dpi=200)

        # Panel (a): Test Case Input
        axes[0].imshow(noisy, cmap='gray', interpolation='nearest')
        axes[0].set_title(cfg['title_a'], fontsize=12, fontweight='bold', color='black', pad=12)
        axes[0].axis('off')

        # Panel (b): Model Restored Output
        title_b = f"{cfg['title_b_prefix']}\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | CD: {cd_err_nm:.3f} nm"
        axes[1].imshow(restored, cmap='gray', interpolation='nearest')
        axes[1].set_title(title_b, fontsize=12, fontweight='bold', color='#0066CC', pad=12)
        axes[1].axis('off')

        # Panel (c): Ground Truth Target
        axes[2].imshow(gt, cmap='gray', interpolation='nearest')
        axes[2].set_title(cfg['title_c'], fontsize=12, fontweight='bold', color='#006600', pad=12)
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(cfg['output_path'], dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[SUCCESS] Saved figure to: {cfg['output_path']}")

if __name__ == '__main__':
    main()

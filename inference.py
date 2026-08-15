"""
inference.py — Single-image inference with uncertainty visualization (dev tool).

NOT the submission script (use evaluate.py for that).
This is for development: visualizes restored image alongside uncertainty maps,
noise type classification, and domain routing weights.
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from model import create_teacher_model
from utils import load_image, save_image, pad_to_multiple, unpad, get_device, load_checkpoint
from uncertainty import full_uncertainty_inference


@torch.no_grad()
def infer_single_image(
    model,
    image_path: str,
    output_dir: str,
    device: torch.device,
    uncertainty_mode: str = 'realtime',
):
    """Run inference on a single image with visualization.
    
    Args:
        model: Loaded model.
        image_path: Path to input image.
        output_dir: Directory to save outputs.
        device: Computation device.
        uncertainty_mode: 'realtime' (single-pass) or 'offline' (8-fold TTA ensemble).
    """
    os.makedirs(output_dir, exist_ok=True)
    name = Path(image_path).stem
    
    # Load image
    img_tensor = load_image(image_path).to(device)  # [1, 1, H, W]
    print(f"Input: {image_path} | Shape: {img_tensor.shape}")
    
    # Check model upscale factor
    upscale_factor = getattr(model, 'upscale_factor', 2)
    
    # Pad input
    padded, pad_sizes = pad_to_multiple(img_tensor, 16)
    
    if uncertainty_mode == 'offline':
        # 8-fold TTA with epistemic uncertainty estimation
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
        preds = []
        for tf, inv_tf in zip(transforms, inv_transforms):
            x_tf = tf(img_tensor)
            p_tf, p_sizes = pad_to_multiple(x_tf, 16)
            out_tf = model(p_tf)['restored']
            unp_tf = unpad(out_tf, (p_sizes[0] * upscale_factor, p_sizes[1] * upscale_factor))
            preds.append(inv_tf(unp_tf))
        
        preds_stacked = torch.stack(preds, dim=0)
        restored = torch.clamp(preds_stacked.mean(dim=0), 0.0, 1.0)
        epistemic_var = preds_stacked.var(dim=0)
    else:
        # Single-pass forward
        output = model(padded)
        restored = unpad(output['restored'], (pad_sizes[0] * upscale_factor, pad_sizes[1] * upscale_factor))
        restored = torch.clamp(restored, 0.0, 1.0)
        epistemic_var = None
        
    # Save restored image
    restored_out_path = os.path.join(output_dir, f'{name}_restored.png')
    save_image(restored, restored_out_path)
    
    # ---- Visualization ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0D1117')
    fig.suptitle(f'SemiRestoreNet Inference Diagnostic: {Path(image_path).name}', fontsize=14, color='white', fontweight='bold')
    
    # 1. Input image
    inp_np = img_tensor.cpu().squeeze().numpy()
    axes[0].imshow(inp_np, cmap='gray')
    axes[0].set_title(f'1. SEM Input [{inp_np.shape[0]}x{inp_np.shape[1]}]', color='white', fontsize=12)
    axes[0].axis('off')
    
    # 2. Restored output
    res_np = restored.cpu().squeeze().numpy()
    axes[1].imshow(res_np, cmap='gray')
    axes[1].set_title(f'2. SemiRestoreNet Output [{res_np.shape[0]}x{res_np.shape[1]}] ({upscale_factor}x SR)', color='#58A6FF', fontsize=12)
    axes[1].axis('off')
    
    # 3. Uncertainty or High-Frequency Edge Map
    axes[2].set_facecolor('#161B22')
    if epistemic_var is not None:
        var_np = epistemic_var.cpu().squeeze().numpy()
        im = axes[2].imshow(var_np, cmap='inferno')
        axes[2].set_title('3. 8-Fold TTA Epistemic Uncertainty (σ²)', color='#FFA657', fontsize=12)
        axes[2].axis('off')
        cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    else:
        # 1D Line profile
        mid_y = res_np.shape[0] // 2
        inp_up = cv2.resize(inp_np, (res_np.shape[1], res_np.shape[0]), interpolation=cv2.INTER_CUBIC)
        axes[2].plot(inp_up[mid_y, :], color='#FF7B72', label='Input (Bicubic)', linestyle=':', alpha=0.7)
        axes[2].plot(res_np[mid_y, :], color='#58A6FF', label='SemiRestoreNet', linewidth=1.8)
        axes[2].set_title(f'3. Cross-Section Intensity Profile (Row Y={mid_y})', color='white', fontsize=12)
        axes[2].set_xlabel('Spatial Position (X)', color='white', fontsize=10)
        axes[2].set_ylabel('Electron Intensity', color='white', fontsize=10)
        axes[2].tick_params(colors='white')
        axes[2].grid(True, linestyle='--', alpha=0.2, color='white')
        axes[2].legend(loc='upper right', facecolor='#21262D', edgecolor='none', labelcolor='white')
        for spine in axes[2].spines.values():
            spine.set_color('#30363D')
            
    plt.tight_layout()
    viz_path = os.path.join(output_dir, f'{name}_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    
    print(f"[SUCCESS] Restored image saved to: {restored_out_path}")
    print(f"[SUCCESS] Diagnostic visual saved to: {viz_path}")


def main():
    parser = argparse.ArgumentParser(description='Single-image inference with uncertainty visualization')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='./inference_output')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--uncertainty', type=str, default='realtime',
                        choices=['realtime', 'offline'],
                        help='Uncertainty mode: realtime (1x single pass) or offline (8-fold TTA ensemble)')
    
    args = parser.parse_args()
    device = get_device()
    
    # Detect upscale_factor from checkpoint if available
    upscale_factor = 2
    if os.path.isfile(args.checkpoint):
        ckpt_data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        sd = ckpt_data.get('model_state_dict', ckpt_data.get('state_dict', ckpt_data))
        if isinstance(sd, dict) and 'restoration_head.head.0.weight' in sd:
            if sd['restoration_head.head.0.weight'].shape[0] == 64:
                upscale_factor = 1
                
    model = create_teacher_model(upscale_factor=upscale_factor).to(device)
    if os.path.isfile(args.checkpoint):
        load_checkpoint(args.checkpoint, model, device=device)
        print(f"[INFO] Loaded checkpoint {args.checkpoint} (upscale_factor={upscale_factor})")
    else:
        print(f"[WARNING] No checkpoint at {args.checkpoint}, using random weights")
    model.eval()
    
    infer_single_image(model, args.image, args.output_dir, device, args.uncertainty)


if __name__ == '__main__':
    import cv2
    main()

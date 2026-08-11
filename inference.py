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
        uncertainty_mode: 'realtime' (aleatoric only) or 'offline' (full uncertainty).
    """
    os.makedirs(output_dir, exist_ok=True)
    name = Path(image_path).stem
    
    # Load image
    img_tensor = load_image(image_path).to(device)  # [1, 1, H, W]
    print(f"Input: {image_path} | Shape: {img_tensor.shape}")
    
    # Pad
    padded, pad_sizes = pad_to_multiple(img_tensor, 16)
    
    # Run uncertainty-aware inference
    result = full_uncertainty_inference(model, padded, mode=uncertainty_mode)
    
    # Unpad
    restored = unpad(result['restored'], pad_sizes)
    restored = torch.clamp(restored, 0, 1)
    
    # Save restored image
    save_image(restored, os.path.join(output_dir, f'{name}_restored.png'))
    
    # Also get degradation info from a single forward pass
    model.eval()
    output = model(padded, return_uncertainty=True)
    
    # ---- Visualization ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Inference: {Path(image_path).name}', fontsize=16)
    
    # Input
    axes[0, 0].imshow(img_tensor.cpu().squeeze(), cmap='gray')
    axes[0, 0].set_title('Input (degraded)')
    axes[0, 0].axis('off')
    
    # Restored
    axes[0, 1].imshow(restored.cpu().squeeze(), cmap='gray')
    axes[0, 1].set_title('Restored')
    axes[0, 1].axis('off')
    
    # Noise map
    noise_map = unpad(output['noise_map'], pad_sizes)
    axes[0, 2].imshow(noise_map.cpu().squeeze(), cmap='hot')
    axes[0, 2].set_title(f"Noise Map (σ̂={output['noise_level'].item():.4f})")
    axes[0, 2].axis('off')
    
    # Domain routing
    routing = output['routing_weights'].cpu().squeeze().numpy()
    labels = ['Speckle\n(log)', 'Gaussian\n(VST)', 'Mixed\n(identity)']
    colors = ['#e94560', '#533483', '#0f3460']
    bars = axes[1, 0].bar(labels, routing, color=colors)
    axes[1, 0].set_title('Domain Routing Weights')
    axes[1, 0].set_ylim(0, 1)
    for bar, val in zip(bars, routing):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', fontsize=11)
    
    # Aleatoric uncertainty
    if result.get('aleatoric_variance') is not None:
        aleatoric = unpad(result['aleatoric_variance'], pad_sizes)
        im = axes[1, 1].imshow(aleatoric.cpu().squeeze(), cmap='inferno')
        axes[1, 1].set_title('Aleatoric Uncertainty (σ²)')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    else:
        axes[1, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20)
        axes[1, 1].set_title('Aleatoric Uncertainty')
        axes[1, 1].axis('off')
    
    # Total uncertainty (if offline mode)
    if result.get('total_variance') is not None and uncertainty_mode == 'offline':
        total_var = unpad(result['total_variance'], pad_sizes)
        im = axes[1, 2].imshow(total_var.cpu().squeeze(), cmap='inferno')
        axes[1, 2].set_title(f'Total Uncertainty ({uncertainty_mode})')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    else:
        axes[1, 2].text(0.5, 0.5, 'Realtime mode\n(aleatoric only)',
                        ha='center', va='center', fontsize=14)
        axes[1, 2].set_title('Epistemic Uncertainty')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, f'{name}_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {os.path.join(output_dir, f'{name}_restored.png')}")
    print(f"Saved: {viz_path}")
    print(f"Routing: Speckle={routing[0]:.3f}, Gaussian={routing[1]:.3f}, Mixed={routing[2]:.3f}")
    print(f"Noise level: σ̂={output['noise_level'].item():.4f}")
    print(f"Scale factor: ŝ={output['scale_factor'].item():.2f}")


def main():
    parser = argparse.ArgumentParser(description='Single-image inference with uncertainty visualization')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='./inference_output')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--uncertainty', type=str, default='realtime',
                        choices=['realtime', 'offline'],
                        help='Uncertainty mode: realtime (1×) or offline (8-12×)')
    
    args = parser.parse_args()
    device = get_device()
    
    # Load model
    model = create_teacher_model().to(device)
    if os.path.isfile(args.checkpoint):
        load_checkpoint(args.checkpoint, model, device=device)
    else:
        print(f"[WARNING] No checkpoint at {args.checkpoint}, using random weights")
    model.eval()
    
    infer_single_image(model, args.image, args.output_dir, device, args.uncertainty)


if __name__ == '__main__':
    main()

"""
train_finetune_high_psnr.py — Fast Fine-Tuning Pipeline for 30–32 dB Metrology PSNR.

Features:
    1. Pretrained Weight Transfer: Initializes from Stage-1/Real-ESRGAN checkpoint.
    2. Calibrated Physics Noise Distribution: Tuned for exact SEM metrology SNR.
    3. ModelEMA Shadow: Continuous parameter averaging (decay=0.9995) to eliminate SGD noise.
    4. Closed-Loop Metrology Loss: Spatially-weighted Charbonnier + Sobel Edge + dNCC + CD profile.
    5. Decoupled Two-Stage Head with Restormer MDTA Global Attention.
    6. Built-in 8-Fold TTA Validation with metric logging (PSNR, SSIM, CD error).
"""

import argparse
import os
import sys
import time
import math
import copy
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from model import create_teacher_model, load_pretrained_rrdb_weights
from losses import CombinedLoss
from dataset import DomainRandomizationDataset
from metrics import compute_psnr, compute_ssim, compute_cd_error
from utils import save_checkpoint, load_checkpoint, get_device, count_parameters, format_params


# =============================================================================
# Model Exponential Moving Average (EMA)
# =============================================================================

class ModelEMA:
    """Maintains moving average of model parameters for smooth, high-PSNR inference."""
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def update(self, model: nn.Module):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                if k in msd and v.dtype.is_floating_point:
                    v.copy_(self.decay * v + (1.0 - self.decay) * msd[k])


# =============================================================================
# 8-Fold Geometric TTA Inference Function
# =============================================================================

def tta_eval_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """8-Fold Geometric Test-Time Augmentation on a single input tensor."""
    preds = []
    # 4 rotations x 2 flips
    for k in [0, 1, 2, 3]:
        for flip in [False, True]:
            x_aug = torch.rot90(x, k, dims=[-2, -1])
            if flip:
                x_aug = torch.flip(x_aug, dims=[-1])
                
            with torch.no_grad():
                out_aug = model(x_aug, return_dict=False)
                
            if flip:
                out_aug = torch.flip(out_aug, dims=[-1])
            out_orig = torch.rot90(out_aug, -k, dims=[-2, -1])
            preds.append(out_orig)
            
    stacked = torch.stack(preds, dim=0)
    return torch.mean(stacked, dim=0)


# =============================================================================
# Validation Function with Multi-Metric Evaluation
# =============================================================================

def validate(model: nn.Module, val_loader: DataLoader, device: torch.device, use_tta: bool = False, max_samples: int = 50) -> dict:
    model.eval()
    psnr_list, ssim_list, cd_list = [], [], []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_samples and i >= max_samples:
                break
                
            degraded = batch['degraded'].to(device)
            clean = batch['clean'].to(device)
            
            if use_tta:
                pred = tta_eval_forward(model, degraded)
            else:
                pred = model(degraded, return_dict=False)
                
            pred_np = pred.squeeze().cpu().numpy()
            clean_np = clean.squeeze().cpu().numpy()
            
            psnr = compute_psnr(pred_np, clean_np)
            ssim = compute_ssim(pred_np, clean_np)
            cd_err = compute_cd_error(pred_np, clean_np)
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            if np.isfinite(cd_err):
                cd_list.append(cd_err)
                
    return {
        'psnr': float(np.mean(psnr_list)) if psnr_list else 0.0,
        'ssim': float(np.mean(ssim_list)) if ssim_list else 0.0,
        'cd_error': float(np.mean(cd_list)) if cd_list else 0.0,
    }


# =============================================================================
# Fast Fine-Tuning Execution Engine
# =============================================================================

def run_finetuning(
    epochs: int = 25,
    batch_size: int = 2,
    accumulation_steps: int = 8,
    lr: float = 8e-5,
    resume_checkpoint: str = "checkpoints/best_finetuned_model.pth",
    pretrained_esrgan: str = "checkpoints/RealESRGAN_x4plus.pth",
    save_dir: str = "checkpoints",
    max_train_steps: int = None,
    max_val_samples: int = 30,
):
    os.makedirs(save_dir, exist_ok=True)
    device = get_device()
    print(f"\n{'='*70}")
    print(f"SemiRestoreNet: Fast 30–32 dB PSNR Fine-Tuning Engine")
    print(f"Device: {device} | Total Epochs: {epochs} | Batch Size: {batch_size} (Acc={accumulation_steps})")
    if max_train_steps:
        print(f"Fast Step Cap: {max_train_steps} steps/epoch | Val Samples: {max_val_samples}")
    print(f"{'='*70}\n")
    
    # 1. Instantiate Next-Gen Model
    model = create_teacher_model(
        num_rrdb_blocks=(8, 8, 7),
        attention_type='mdta',
        upscale_factor=2,
        use_log_domain=True,
    ).to(device)
    
    # 2. Transfer Weights
    if os.path.isfile(resume_checkpoint):
        print(f"[Init] Loading weights from {resume_checkpoint}...")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt.get('state_dict', ckpt)
        if any(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        m_state = model.state_dict()
        matched = {k: v for k, v in state.items() if k in m_state and v.shape == m_state[k].shape}
        model.load_state_dict(matched, strict=False)
        print(f"  -> Successfully transferred {len(matched)} matching parameter tensors.")
    elif os.path.isfile(pretrained_esrgan):
        print(f"[Init] Transferring pretrained Real-ESRGAN weights from {pretrained_esrgan}...")
        load_pretrained_rrdb_weights(model, pretrained_esrgan, verbose=True)
        
    ema = ModelEMA(model, decay=0.999)  # Faster adaptation (was 0.9995)
    
    # 3. Setup Dataset
    train_dir = './train/train/GT' if Path('./train/train/GT').is_dir() else './data/sample_dataset/search'
    print(f"[Data] Training data source: {train_dir}")
    
    train_dataset = DomainRandomizationDataset(
        data_dir=train_dir,
        patch_size=128,
        mode='train',
        upscale_factor=2,
    )
    val_dataset = DomainRandomizationDataset(
        data_dir=train_dir,
        patch_size=None,
        mode='val',
        upscale_factor=2,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    
    # 4. Calibrated Metrology Loss Stack
    loss_fn = CombinedLoss(
        lambda_charb=1.0,
        lambda_ssim=0.15,
        lambda_edge=0.08,
        lambda_fft=0.01,
        lambda_fidelity=0.001,    # 10x reduced — was fighting denoising at 0.010
        lambda_metrology=0.005,   # 5x reduced — CD loss was overwhelming Charbonnier at 0.025
        edge_boost=5.0,
    ).to(device)
    
    # 5. Layer-Wise Optimizer
    backbone_params = []
    head_attn_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(p) for p in ['stage1', 'stage2', 'stage3', 'conv_body', 'conv_first']):
            backbone_params.append(param)
        else:
            head_attn_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1, 'name': 'pretrained_trunk'},
        {'params': head_attn_params, 'lr': lr * 2.5, 'name': 'mdta_and_heads'},  # 2.5x LR for new modules
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    
    # 6. Training Loop
    best_psnr = 0.0
    best_ema_psnr = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}]", leave=False)
        for step, batch in enumerate(pbar):
            if max_train_steps and step >= max_train_steps:
                break
                
            degraded = batch['degraded'].to(device)
            clean = batch['clean'].to(device)
            noise_level_gt = batch['noise_level'].to(device) if 'noise_level' in batch else None
            charging_applied = batch.get('charging_applied', None)
            if charging_applied is not None:
                charging_applied = charging_applied.float().to(device)
            
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                out_dict = model(degraded, return_dict=True)
                out = out_dict['restored']
                noise_level_pred = out_dict.get('noise_level_pred', None)
                losses = loss_fn(
                    pred=out, target=clean, degraded=degraded,
                    noise_level_pred=noise_level_pred,
                    noise_level_gt=noise_level_gt,
                    charging_applied=charging_applied,
                )
                loss = losses['total'] / accumulation_steps
                
            scaler.scale(loss).backward()
            total_loss += losses['total'].item()
            
            total_steps_in_epoch = min(max_train_steps, len(train_loader)) if max_train_steps else len(train_loader)
            if (step + 1) % accumulation_steps == 0 or (step + 1) == total_steps_in_epoch:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                
            pbar.set_postfix({'Loss': f"{losses['total'].item():.4f}"})
            
        scheduler.step()
        epoch_time = time.time() - t0
        actual_steps = max_train_steps if max_train_steps else len(train_loader)
        avg_train_loss = total_loss / max(actual_steps, 1)
        
        # Validation
        val_metrics = validate(model, val_loader, device, use_tta=False, max_samples=max_val_samples)
        ema_metrics = validate(ema.module, val_loader, device, use_tta=False, max_samples=max_val_samples)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val PSNR: {val_metrics['psnr']:.2f} dB (SSIM: {val_metrics['ssim']:.4f}, CD: {val_metrics['cd_error']:.3f}nm) | "
              f"EMA PSNR: {ema_metrics['psnr']:.2f} dB")
        
        # Save checkpoints
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
            }, os.path.join(save_dir, "best_finetuned_model.pth"))
            
        if ema_metrics['psnr'] > best_ema_psnr:
            best_ema_psnr = ema_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema.module.state_dict(),
                'metrics': ema_metrics,
            }, os.path.join(save_dir, "best_ema_model.pth"))
            
    print(f"\n{'='*70}")
    print(f"Fine-Tuning Completed!")
    print(f"  - Best Model PSNR:     {best_psnr:.2f} dB")
    print(f"  - Best EMA Model PSNR: {best_ema_psnr:.2f} dB")
    print(f"Checkpoints saved in `{save_dir}/`")
    print(f"{'='*70}\n")
    
    # 7. Final 8-Fold TTA Benchmark
    print("[Final Evaluation] Running 8-Fold Geometric TTA on Best EMA Checkpoint...")
    tta_metrics = validate(ema.module, val_loader, device, use_tta=True, max_samples=max_val_samples)
    print(f"\nFinal Metrology Evaluation (8-Fold TTA):")
    print(f"  - Benchmark PSNR: {tta_metrics['psnr']:.2f} dB")
    print(f"  - Benchmark SSIM: {tta_metrics['ssim']:.4f}")
    print(f"  - Critical Dimension (CD) Error: {tta_metrics['cd_error']:.3f} nm")
    
    # 8. Multi-Checkpoint Ensemble (Average best model + best EMA)
    best_ckpt = os.path.join(save_dir, "best_finetuned_model.pth")
    ema_ckpt = os.path.join(save_dir, "best_ema_model.pth")
    if os.path.isfile(best_ckpt) and os.path.isfile(ema_ckpt):
        print("\n[Ensemble] Averaging Best + EMA checkpoints...")
        model_a = create_teacher_model(
            num_rrdb_blocks=(8, 8, 7), attention_type='mdta',
            upscale_factor=2, use_log_domain=True,
        ).to(device)
        model_b = create_teacher_model(
            num_rrdb_blocks=(8, 8, 7), attention_type='mdta',
            upscale_factor=2, use_log_domain=True,
        ).to(device)
        sd_a = torch.load(best_ckpt, map_location=device, weights_only=False)['model_state_dict']
        sd_b = torch.load(ema_ckpt, map_location=device, weights_only=False)['model_state_dict']
        model_a.load_state_dict(sd_a)
        model_b.load_state_dict(sd_b)
        # Create averaged model
        avg_sd = {}
        for k in sd_a:
            avg_sd[k] = (sd_a[k].float() + sd_b[k].float()) / 2.0
        model_a.load_state_dict(avg_sd)
        model_a.eval()
        ensemble_metrics = validate(model_a, val_loader, device, use_tta=True, max_samples=max_val_samples)
        print(f"  - Ensemble PSNR: {ensemble_metrics['psnr']:.2f} dB")
        print(f"  - Ensemble SSIM: {ensemble_metrics['ssim']:.4f}")
        print(f"  - Ensemble CD Error: {ensemble_metrics['cd_error']:.3f} nm")
        torch.save({'model_state_dict': avg_sd, 'metrics': ensemble_metrics},
                   os.path.join(save_dir, "ensemble_model.pth"))
        print(f"  -> Saved to {save_dir}/ensemble_model.pth")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune SemiRestoreNet for 30–32 dB PSNR')
    parser.add_argument('--epochs', type=int, default=25, help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per iteration')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=8e-5, help='Learning rate')
    parser.add_argument('--resume', type=str, default='checkpoints/best_finetuned_model.pth', help='Resume checkpoint path')
    parser.add_argument('--max_train_steps', type=int, default=None, help='Max train steps per epoch for fast turnaround')
    parser.add_argument('--max_val_samples', type=int, default=30, help='Max validation samples')
    args = parser.parse_args()
    
    run_finetuning(
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        lr=args.lr,
        resume_checkpoint=args.resume,
        max_train_steps=args.max_train_steps,
        max_val_samples=args.max_val_samples,
    )

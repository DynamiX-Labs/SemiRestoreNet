"""
train.py — Main training pipeline for SemiRestoreNet.

Training Stability & Memory Faults Faced & Engineering Solutions History:
--------------------------------------------------------------------------
FAULT 1: OSError [WinError 1455] Paging File Too Small / CUDA OOM
- Initial Issue: Large batch sizes and unconstrained PyTorch CUDA memory allocator caused Windows pagefile exhaustion 
  and CUDA Out-Of-Memory crashes on 4GB GPUs.
- Solution Implemented: Set `os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'`, set batch_size = 2,
  and used 8-step gradient accumulation (effective batch size = 16) for zero-OOM memory stability.

FAULT 2: Destruction of Pretrained Weights During Early Epochs
- Initial Issue: Applying a high learning rate (2e-4) equally to the backbone and head destroyed pretrained Real-ESRGAN weights.
- Solution Implemented: Introduced layer-wise optimizer parameter groups with 0.1x LR backbone scaling factor 
  (backbone_lr = 1e-5, head_lr = 1e-4) and 3-epoch linear warmup.
"""

import argparse
import os
import sys
import time
import yaml
from pathlib import Path

# Configure PyTorch memory allocator to avoid fragmentation on Windows GPUs
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.amp
import numpy as np
from tqdm import tqdm

from model import create_teacher_model, create_student_model, load_pretrained_rrdb_weights
from losses import CombinedLoss
from dataset import DomainRandomizationDataset
from metrics import compute_psnr, compute_ssim, compute_cd_error, compute_frequency_error
from utils import save_checkpoint, load_checkpoint, get_device, count_parameters, format_params


# =============================================================================
# Learning Rate Scheduler with Warmup
# =============================================================================

class WarmupCosineScheduler:
    """Cosine annealing with linear warmup supporting multiple param groups."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * factor)


# =============================================================================
# Exponential Moving Average (EMA)
# =============================================================================

class EMA:
    """Exponential Moving Average of model weights."""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._init_shadow()
    
    def _init_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# =============================================================================
# Training & Validation Loops
# =============================================================================

def train_one_epoch(
    model, dataloader, loss_fn, optimizer, scaler, device, epoch,
    use_amp=True, accumulation_steps=1, ema=None,
) -> dict:
    """Train for a single epoch."""
    model.train()
    loss_accum = {}
    count = 0
    
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    for step_idx, batch in enumerate(pbar):
        degraded = batch['degraded'].to(device)
        clean = batch['clean'].to(device)
        
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(degraded)
            losses = loss_fn(pred=output['restored'], target=clean, degraded=degraded)
            scaled_loss = losses['total'] / accumulation_steps
        
        if not torch.isfinite(scaled_loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        
        if use_amp and device.type == 'cuda':
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        if (step_idx + 1) % accumulation_steps == 0 or (step_idx + 1) == len(dataloader):
            if use_amp and device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update()
        
        for key, val in losses.items():
            if key not in loss_accum:
                loss_accum[key] = 0.0
            loss_accum[key] += val.item()
        count += 1
        
        postfix = {
            'total': f"{losses['total'].item():.4f}",
            'charb': f"{losses['charb'].item():.4f}",
            'edge': f"{losses['edge'].item():.4f}",
        }
        if 'fidelity' in losses and losses['fidelity'].item() > 0:
            postfix['fid'] = f"{losses['fidelity'].item():.4f}"
        pbar.set_postfix(postfix)
    
    return {k: v / max(count, 1) for k, v in loss_accum.items()}


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, max_val_samples: int = None) -> dict:
    """Run validation and compute metrology metrics with optional sample limit for speed."""
    model.eval()
    loss_accum = {}
    metrics_list = []
    count = 0
    
    for idx, batch in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
        if max_val_samples is not None and idx >= max_val_samples:
            break
        degraded = batch['degraded'].to(device)
        clean = batch['clean'].to(device)
        
        output = model(degraded)
        losses = loss_fn(pred=output['restored'], target=clean, degraded=degraded)
        
        for key, val in losses.items():
            if key not in loss_accum:
                loss_accum[key] = 0.0
            loss_accum[key] += val.item()
        count += 1
        
        pred_np = output['restored'].cpu().squeeze().numpy()
        clean_np = clean.cpu().squeeze().numpy()
        
        psnr = compute_psnr(pred_np, clean_np)
        ssim = compute_ssim(pred_np, clean_np)
        cd_err = compute_cd_error(pred_np, clean_np)
        metrics_list.append({'psnr': psnr, 'ssim': ssim, 'cd_error': cd_err})
    
    avg_losses = {k: v / max(count, 1) for k, v in loss_accum.items()}
    avg_psnr = np.mean([m['psnr'] for m in metrics_list]) if metrics_list else 0.0
    avg_ssim = np.mean([m['ssim'] for m in metrics_list]) if metrics_list else 0.0
    cd_vals = [m['cd_error'] for m in metrics_list if np.isfinite(m['cd_error'])]
    avg_cd = np.mean(cd_vals) if cd_vals else 0.0
    
    return {
        **avg_losses,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'cd_error': avg_cd,
    }


# =============================================================================
# Main Training Entry Point
# =============================================================================

def train(config: dict):
    device = get_device()
    print(f"[Train] Device: {device}")
    
    use_log_domain = config.get('use_log_domain', True)
    model = create_teacher_model(
        num_feat=config.get('num_feat', 64),
        num_grow_ch=config.get('num_grow_ch', 32),
        num_rrdb_blocks=tuple(config.get('num_rrdb_blocks', [8, 8, 7])),
        window_size=config.get('window_size', 8),
        upscale_factor=config.get('upscale_factor', 1),
        drop_path_rate=config.get('drop_path_rate', 0.1),
        use_log_domain=use_log_domain,
    ).to(device)
    
    print(f"[Train] Model parameters: {format_params(count_parameters(model))} (Log-domain stream: {use_log_domain})")
    
    # 1. Pretrained Weight Transfer (ESRGAN / Real-ESRGAN RRDB Trunk)
    pretrained_path = config.get('pretrained_weights')
    if pretrained_path:
        print(f"[Train] Transferring pretrained RRDB weights from {pretrained_path}...")
        load_pretrained_rrdb_weights(model, pretrained_path, verbose=True)
    
    # 2. Checkpoint Resume
    start_epoch = 0
    best_psnr = 0.0
    if config.get('resume_checkpoint'):
        ckpt_info = load_checkpoint(config['resume_checkpoint'], model, device=device)
        start_epoch = ckpt_info['epoch'] + 1
        best_psnr = ckpt_info['metrics'].get('psnr', 0.0)
        print(f"[Train] Resumed from epoch {start_epoch}")
    
    train_dir = config.get('train_data_dir')
    if not train_dir:
        if Path('./train/train/GT').is_dir():
            train_dir = './train/train/GT'
        elif Path('./data/sample_dataset/search').is_dir():
            train_dir = './data/sample_dataset/search'
        else:
            train_dir = './data'
            
    val_dir = config.get('val_data_dir')
    if not val_dir:
        val_dir = train_dir

    upscale_factor = config.get('upscale_factor', 1)
    train_dataset = DomainRandomizationDataset(
        data_dir=train_dir,
        patch_size=config.get('patch_size', 128),
        mode='train',
        upscale_factor=upscale_factor,
    )
    val_dataset = DomainRandomizationDataset(
        data_dir=val_dir,
        patch_size=None,
        mode='val',
        upscale_factor=upscale_factor,
    )
    
    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=True,
    )
    
    loss_fn = CombinedLoss(
        lambda_charb=config.get('lambda_charb', 1.0),
        lambda_ssim=config.get('lambda_ssim', 0.1),
        lambda_edge=config.get('lambda_edge', 0.05),
        lambda_fft=config.get('lambda_fft', 0.01),
        lambda_fidelity=config.get('lambda_fidelity', 0.05),
        edge_boost=config.get('edge_boost', 3.0),
        fft_cap=config.get('fft_cap', 2.0),
        enable_fft=config.get('enable_fft', True),
    ).to(device)
    
    # 3. Layer-Wise Learning Rate Optimizer
    base_lr = config.get('learning_rate', 2e-4)
    lr_backbone_scale = config.get('lr_backbone_scale', 0.2 if pretrained_path else 1.0)
    
    backbone_params = []
    head_attention_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Pretrained trunk: stage1, stage2, stage3, conv_body, conv_first
        if any(name.startswith(p) for p in ['stage1', 'stage2', 'stage3', 'conv_body', 'conv_first']):
            backbone_params.append(param)
        else:
            head_attention_params.append(param)
            
    param_groups = [
        {'params': backbone_params, 'lr': base_lr * lr_backbone_scale, 'name': 'backbone_trunk'},
        {'params': head_attention_params, 'lr': base_lr, 'name': 'attention_and_heads'},
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.get('weight_decay', 1e-4),
        betas=(0.9, 0.999),
    )
    print(f"[Train] Optimizer: AdamW with {len(param_groups)} param groups (Trunk LR: {base_lr * lr_backbone_scale:.2e}, Heads LR: {base_lr:.2e})")
    
    total_epochs = config.get('total_epochs', 200)
    if start_epoch >= total_epochs:
        # Incremental fine-tuning: add target epochs to start_epoch
        total_epochs = start_epoch + total_epochs
        
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.get('warmup_epochs', 5),
        total_epochs=total_epochs,
        min_lr=config.get('min_lr', 1e-7),
    )
    
    use_amp = config.get('use_amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    ema_decay = config.get('ema_decay', 0.999)
    ema = EMA(model, decay=ema_decay)
    
    log_dir = config.get('log_dir', './runs')
    writer = SummaryWriter(log_dir=log_dir)
    
    ckpt_dir = config.get('checkpoint_dir', './checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"\n[Train] Starting training for {total_epochs} epochs")
    
    for epoch in range(start_epoch, total_epochs):
        scheduler.step(epoch)
        current_head_lr = optimizer.param_groups[1]['lr']
        
        train_losses = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, epoch,
            use_amp, accumulation_steps=config.get('accumulation_steps', 1), ema=ema,
        )
        
        for key, val in train_losses.items():
            writer.add_scalar(f'train/{key}', val, epoch)
        writer.add_scalar('train/lr', current_head_lr, epoch)
        
        val_interval = config.get('val_interval', 1)
        if (epoch + 1) % val_interval == 0 or epoch == total_epochs - 1:
            ema.apply_shadow()
            # Fast validation on 100 images for speed during epochs, full dataset on final epoch
            val_limit = None if epoch == total_epochs - 1 else config.get('max_val_samples', 100)
            val_results = validate(model, val_loader, loss_fn, device, max_val_samples=val_limit)
            ema.restore()
            
            for key, val in val_results.items():
                writer.add_scalar(f'val/{key}', val, epoch)
                
            psnr = val_results['psnr']
            ssim = val_results['ssim']
            cd_err = val_results['cd_error']
            print(f"Epoch {epoch:3d} | Train Loss: {train_losses['total']:.4f} | Val PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | CD Error: {cd_err:.3f} px (~{cd_err * 0.15:.3f} nm)")
            
            if psnr > best_psnr:
                best_psnr = psnr
                ema.apply_shadow()
                save_checkpoint(
                    model, optimizer, epoch, val_results,
                    os.path.join(ckpt_dir, 'best_model.pth')
                )
                ema.restore()
                print(f"  --> Saved new best model (PSNR: {best_psnr:.2f} dB)")
                
    writer.close()
    print("\n[Train] Complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SemiRestoreNet')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained RRDB weights (.pth)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from (.pth)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    if args.pretrained_weights:
        cfg['pretrained_weights'] = args.pretrained_weights
    if args.resume:
        cfg['resume_checkpoint'] = args.resume
        
    train(cfg)
# Gradient accumulation counter

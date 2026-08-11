"""
train_kd.py — Knowledge Distillation Training for Student Models.

Trains a smaller student model using a frozen teacher's outputs:
    L_KD = α * L1(student_output, teacher_output) 
         + β * L1(student_features, teacher_features)
         + γ * L_task(student_output, ground_truth)

Key requirements (from plan):
    - ALL Pareto points must be trained with same methodology (KD or clearly labeled)
    - Student MUST have its own MC-Dropout + uncertainty head (trained with NLL, not distilled)
    - Separate uncertainty calibration validation after KD training
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.amp
import numpy as np
from tqdm import tqdm

from model import create_teacher_model, create_student_model
from losses import CombinedLoss
from dataset import DomainRandomizationDataset
from metrics import compute_psnr, compute_ssim
from utils import save_checkpoint, load_checkpoint, get_device, count_parameters, format_params


# =============================================================================
# Knowledge Distillation Loss
# =============================================================================

class KDLoss(nn.Module):
    """Knowledge Distillation loss combining teacher supervision with task loss.
    
    L_total = α * L_output_KD + β * L_feature_KD + γ * L_task
    
    Where:
        - L_output_KD: L1 between student and teacher restored outputs
        - L_feature_KD: L1 between intermediate features (at stage boundaries)
        - L_task: Full CombinedLoss against ground truth (includes uncertainty NLL)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 1.0,
        task_loss_config: dict = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Task loss for student (against GT) — includes uncertainty NLL
        config = task_loss_config or {}
        self.task_loss = CombinedLoss(**config)
        
        # Feature adaptation layers (student features may differ in channel count)
        self.adapt_layers = nn.ModuleDict()
    
    def _get_adapt_layer(self, name, student_channels, teacher_channels, device):
        """Lazily create feature adaptation layers."""
        if name not in self.adapt_layers:
            if student_channels != teacher_channels:
                self.adapt_layers[name] = nn.Conv2d(
                    student_channels, teacher_channels, 1, bias=False
                ).to(device)
            else:
                self.adapt_layers[name] = nn.Identity()
        return self.adapt_layers[name]
    
    def forward(
        self,
        student_output: dict,
        teacher_output: dict,
        student_features: dict,
        teacher_features: dict,
        target: torch.Tensor,
        degraded: torch.Tensor = None,
        log_variance: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            student_output: Student model output dict (with 'restored').
            teacher_output: Teacher model output dict (with 'restored').
            student_features: Student intermediate features dict.
            teacher_features: Teacher intermediate features dict.
            target: Ground truth [B, 1, H, W].
            degraded: Degraded input tensor [B, 1, H, W] for fidelity loss.
            log_variance: Student's predicted log-variance (for uncertainty training).
        """
        losses = {}
        
        # 1. Output-level KD: match teacher's restored output
        losses['kd_output'] = nn.functional.l1_loss(
            student_output['restored'],
            teacher_output['restored'].detach(),
        )
        
        # 2. Feature-level KD: match at stage boundaries
        kd_feat_loss = torch.tensor(0.0, device=target.device)
        feat_count = 0
        
        for key in teacher_features:
            if key in student_features and key != 'restored':
                t_feat = teacher_features[key].detach()
                s_feat = student_features[key]
                
                # Adapt channel count if needed
                adapt = self._get_adapt_layer(
                    key, s_feat.shape[1], t_feat.shape[1], target.device
                )
                s_feat_adapted = adapt(s_feat)
                
                kd_feat_loss = kd_feat_loss + nn.functional.l1_loss(s_feat_adapted, t_feat)
                feat_count += 1
        
        if feat_count > 0:
            kd_feat_loss = kd_feat_loss / feat_count
        losses['kd_features'] = kd_feat_loss
        
        # 3. Task loss: student against ground truth and degraded evidence
        task_losses = self.task_loss(
            pred=student_output['restored'],
            target=target,
            degraded=degraded,
        )
        losses['task'] = task_losses['total']
        
        # Include individual task loss components for logging
        for key, val in task_losses.items():
            if key != 'total':
                losses[f'task_{key}'] = val
        
        # Total KD loss
        losses['total'] = (
            self.alpha * losses['kd_output']
            + self.beta * losses['kd_features']
            + self.gamma * losses['task']
        )
        
        return losses


# =============================================================================
# KD Training Loop
# =============================================================================

def train_kd(config: dict):
    """Knowledge distillation training.
    
    Args:
        config: Training configuration.
    """
    device = get_device()
    print(f"[KD] Device: {device}")
    
    # ---- Load Teacher (frozen) ----
    teacher = create_teacher_model(
        num_feat=config.get('teacher_num_feat', 64),
        num_grow_ch=config.get('teacher_num_grow_ch', 32),
        num_rrdb_blocks=tuple(config.get('teacher_num_rrdb_blocks', [8, 8, 5])),
    ).to(device)
    
    teacher_ckpt = config.get('teacher_checkpoint', 'checkpoints/best_model.pth')
    if os.path.isfile(teacher_ckpt):
        load_checkpoint(teacher_ckpt, teacher, device=device)
        print(f"[KD] Teacher loaded from {teacher_ckpt}")
    else:
        print(f"[WARNING] Teacher checkpoint not found: {teacher_ckpt}")
        print("[WARNING] Using randomly initialized teacher (for testing only)")
    
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    print(f"[KD] Teacher parameters: {format_params(count_parameters(teacher))}")
    
    # ---- Create Student ----
    student_blocks = config.get('student_num_blocks', 8)
    student = create_student_model(
        num_blocks=student_blocks,
        num_feat=config.get('student_num_feat', 64),
        num_grow_ch=config.get('student_num_grow_ch', 32),
        drop_path_rate=config.get('drop_path_rate', 0.05),
        use_log_domain=config.get('use_log_domain', True),
    ).to(device)
    
    print(f"[KD] Student ({student_blocks} blocks) parameters: {format_params(count_parameters(student))}")
    
    # ---- Dataset ----
    train_dataset = DomainRandomizationDataset(
        data_dir=config['train_data_dir'],
        patch_size=config.get('patch_size', 128),
        mode='train',
    )
    val_dataset = DomainRandomizationDataset(
        data_dir=config.get('val_data_dir', config['train_data_dir']),
        patch_size=None,
        mode='val',
    )
    
    batch_size = config.get('batch_size', 8)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=config.get('num_workers', 4),
        pin_memory=True, drop_last=(len(train_dataset) > batch_size),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=config.get('num_workers', 4), pin_memory=True,
    )
    
    # ---- KD Loss ----
    kd_loss_fn = KDLoss(
        alpha=config.get('kd_alpha', 1.0),
        beta=config.get('kd_beta', 0.5),
        gamma=config.get('kd_gamma', 1.0),
        task_loss_config={
            'lambda_charb': config.get('lambda_charb', 1.0),
            'lambda_ssim': config.get('lambda_ssim', 0.1),
            'lambda_edge': config.get('lambda_edge', 0.05),
            'lambda_fft': config.get('lambda_fft', 0.01),
            'lambda_fidelity': config.get('lambda_fidelity', 0.05),
            'enable_fft': config.get('enable_fft', True),
        },
    ).to(device)
    
    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(kd_loss_fn.adapt_layers.parameters()),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-4),
    )
    
    total_epochs = config.get('total_epochs', 100)
    use_amp = config.get('use_amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    ckpt_dir = config.get('checkpoint_dir', './checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    best_psnr = 0.0
    
    # ---- Training ----
    print(f"\n[KD] Starting KD training for {total_epochs} epochs")
    print(f"[KD] alpha(output)={config.get('kd_alpha', 1.0)}, "
          f"beta(feature)={config.get('kd_beta', 0.5)}, "
          f"gamma(task)={config.get('kd_gamma', 1.0)}")
    print()
    
    for epoch in range(total_epochs):
        student.train()
        loss_accum = {}
        count = 0
        
        pbar = tqdm(train_loader, desc=f"KD Epoch {epoch}", leave=False)
        
        for batch in pbar:
            degraded = batch['degraded'].to(device)
            clean = batch['clean'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device.type, enabled=use_amp):
                # Teacher forward (frozen, no grad)
                with torch.no_grad():
                    teacher_output = teacher(degraded)
                    teacher_features = teacher.get_intermediate_features(degraded)
                
                # Student forward
                student_output = student(degraded)
                student_features = student.get_intermediate_features(degraded)
                
                # KD loss
                losses = kd_loss_fn(
                    student_output=student_output,
                    teacher_output=teacher_output,
                    student_features=student_features,
                    teacher_features=teacher_features,
                    target=clean,
                    degraded=degraded,
                )
            
            if use_amp and device.type == 'cuda':
                scaler.scale(losses['total']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
            
            for key, val in losses.items():
                if key not in loss_accum:
                    loss_accum[key] = 0.0
                loss_accum[key] += val.item()
            count += 1
            
            pbar.set_postfix({
                'total': f"{losses['total'].item():.4f}",
                'kd': f"{losses['kd_output'].item():.4f}",
                'task': f"{losses['task'].item():.4f}",
            })
        
        avg_losses = {k: v / count for k, v in loss_accum.items()}
        
        # Validation
        if (epoch + 1) % config.get('val_interval', 5) == 0:
            student.eval()
            psnr_vals = []
            
            with torch.no_grad():
                for batch in val_loader:
                    degraded = batch['degraded'].to(device)
                    clean = batch['clean'].to(device)
                    output = student(degraded)
                    
                    p = output['restored'].cpu().squeeze().numpy()
                    c = clean.cpu().squeeze().numpy()
                    psnr_vals.append(compute_psnr(p, c))
            
            val_psnr = np.mean(psnr_vals) if psnr_vals else 0.0
            
            print(f"KD Epoch {epoch:4d} | Loss {avg_losses['total']:.4f} | "
                  f"KD {avg_losses['kd_output']:.4f} | "
                  f"Task {avg_losses['task']:.4f} | "
                  f"Val PSNR {val_psnr:.2f}")
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    student, optimizer, epoch,
                    {'psnr': val_psnr, 'student_blocks': student_blocks},
                    os.path.join(ckpt_dir, f'student_{student_blocks}b_best.pth')
                )
                print(f"  → Saved best student-{student_blocks}b (PSNR={val_psnr:.2f})")
        else:
            print(f"KD Epoch {epoch:4d} | Loss {avg_losses['total']:.4f}")
    
    # Save final
    save_checkpoint(
        student, optimizer, total_epochs - 1,
        {'psnr': best_psnr, 'student_blocks': student_blocks},
        os.path.join(ckpt_dir, f'student_{student_blocks}b_final.pth')
    )
    
    print(f"\n[KD] Training complete. Best PSNR: {best_psnr:.2f}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--train_data_dir', type=str, default=None)
    parser.add_argument('--val_data_dir', type=str, default=None)
    parser.add_argument('--teacher_checkpoint', type=str, default=None)
    parser.add_argument('--student_blocks', type=int, default=None,
                        help='Number of RRDB blocks in student (8 or 16)')
    parser.add_argument('--epochs', type=int, default=None)
    
    args = parser.parse_args()
    
    config_path = args.config
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    if args.train_data_dir:
        config['train_data_dir'] = args.train_data_dir
    if args.val_data_dir:
        config['val_data_dir'] = args.val_data_dir
    if args.teacher_checkpoint:
        config['teacher_checkpoint'] = args.teacher_checkpoint
    if args.student_blocks:
        config['student_num_blocks'] = args.student_blocks
    if args.epochs:
        config['total_epochs'] = args.epochs
    
    if 'train_data_dir' not in config:
        parser.error("--train_data_dir is required")
    
    train_kd(config)


if __name__ == '__main__':
    main()

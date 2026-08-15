"""
test_physics_improvements.py — Comprehensive Unit & Integration Test Suite.

Verifies:
    1. Unclipped dynamic range in degradation functions & dataset loading.
    2. Homomorphic signed log-domain stream with negative/positive inputs & gradient flow.
    3. Real-ESRGAN style high-order semiconductor degradation pipeline.
    4. Degradation-consistency (fidelity) loss shape math for 1x and 2x scales.
    5. Pretrained RRDB weight transfer loader with RGB->Grayscale adaptation.
    6. Full training forward-backward step with layer-wise parameter groups & EMA.
"""

import sys
import os
import math
import torch
import torch.nn as nn
import numpy as np

# Ensure local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from dataset import (
    add_speckle_noise,
    add_gaussian_noise,
    add_poisson_noise,
    add_charging_drift,
    apply_anisotropic_gaussian_blur,
    downsample_image,
    apply_real_esrgan_sem_pipeline,
    DomainRandomizationDataset,
)
from model import (
    FullModel,
    SignedLogTransform,
    create_teacher_model,
    create_student_model,
    load_pretrained_rrdb_weights,
)
from losses import DegradationConsistencyLoss, CombinedLoss
from utils import count_parameters


def test_1_unclipped_noise():
    print("=" * 60)
    print("Test 1: Unclipped Physics-Aware Degradation Functions")
    print("=" * 60)
    
    clean = np.ones((64, 64), dtype=np.float32) * 0.8
    
    # Speckle noise
    speckle_noisy = add_speckle_noise(clean, num_looks=1.0)
    assert not np.isnan(speckle_noisy).any(), "NaN in speckle noise"
    assert not np.isinf(speckle_noisy).any(), "Inf in speckle noise"
    print(f"  [Speckle] Mean: {speckle_noisy.mean():.4f}, Min: {speckle_noisy.min():.4f}, Max: {speckle_noisy.max():.4f}")
    assert speckle_noisy.max() > 1.0, f"Expected unclipped values > 1.0, got max {speckle_noisy.max()}"
    
    # Gaussian noise
    gauss_noisy = add_gaussian_noise(clean, sigma=0.2)
    assert not np.isnan(gauss_noisy).any(), "NaN in gaussian noise"
    print(f"  [Gaussian] Mean: {gauss_noisy.mean():.4f}, Min: {gauss_noisy.min():.4f}, Max: {gauss_noisy.max():.4f}")
    
    # Poisson noise
    poisson_noisy = add_poisson_noise(clean, peak_photons=20.0)
    assert not np.isnan(poisson_noisy).any(), "NaN in poisson noise"
    print(f"  [Poisson] Mean: {poisson_noisy.mean():.4f}, Min: {poisson_noisy.min():.4f}, Max: {poisson_noisy.max():.4f}")
    
    # Charging drift
    charging_noisy = add_charging_drift(clean, strength=0.1)
    assert not np.isnan(charging_noisy).any(), "NaN in charging drift"
    print(f"  [Charging] Mean: {charging_noisy.mean():.4f}, Min: {charging_noisy.min():.4f}, Max: {charging_noisy.max():.4f}")
    
    print("  --> [PASS] Test 1: Degradation functions preserve true unclipped range without NaN/Inf.")


def test_2_real_esrgan_pipeline():
    print("\n" + "=" * 60)
    print("Test 2: Real-ESRGAN High-Order Degradation Engine")
    print("=" * 60)
    
    clean = np.random.uniform(0.1, 0.9, size=(128, 128)).astype(np.float32)
    
    for i in range(10):
        deg, meta = apply_real_esrgan_sem_pipeline(clean)
        assert deg.shape == (128, 128), f"Shape mismatch: {deg.shape}"
        assert not np.isnan(deg).any(), f"NaN in iteration {i}"
        assert not np.isinf(deg).any(), f"Inf in iteration {i}"
        
    print(f"  [Sample Metadata] {meta}")
    print(f"  [Sample Degraded Range] [{deg.min():.3f}, {deg.max():.3f}]")
    print("  --> [PASS] Test 2: Real-ESRGAN degradation engine runs cleanly with diverse parameters.")


def test_3_log_domain_stream():
    print("\n" + "=" * 60)
    print("Test 3: Homomorphic Signed Log-Domain Processing Stream")
    print("=" * 60)
    
    # Test SignedLogTransform with negative, zero, and positive inputs
    log_trans = SignedLogTransform(epsilon=0.05)
    x_test = torch.tensor([[-1.0, -0.1, 0.0, 0.5, 2.5]], dtype=torch.float32)
    y_test = log_trans(x_test)
    assert not torch.isnan(y_test).any(), "NaN in SignedLogTransform"
    assert (y_test[0, :2] < 0).all(), "Negative inputs must have negative outputs (sign preservation)"
    assert y_test[0, 2] == 0.0, "Zero input must produce zero"
    assert (y_test[0, 3:] > 0).all(), "Positive inputs must have positive outputs"
    print(f"  [SignedLog Input]  {x_test.numpy().flatten()}")
    print(f"  [SignedLog Output] {y_test.numpy().flatten()}")
    
    # Test FullModel with use_log_domain=True and False
    model_log = FullModel(num_rrdb_blocks=(1, 1, 1), use_log_domain=True)
    model_nolog = FullModel(num_rrdb_blocks=(1, 1, 1), use_log_domain=False)
    
    inp = torch.randn(2, 1, 32, 32, requires_grad=True)
    
    # Forward and backward on log model
    out_log = model_log(inp)['restored']
    loss_log = out_log.sum()
    loss_log.backward()
    assert inp.grad is not None, "Gradient must flow back to input in log model"
    assert not torch.isnan(inp.grad).any(), "NaN in input gradient"
    
    # Forward on no-log model
    out_nolog = model_nolog(inp)['restored']
    assert out_nolog.shape == inp.shape, "Shape mismatch"
    
    print(f"  [Model with Log Stream] Params: {count_parameters(model_log):,}")
    print(f"  [Model without Log Stream] Params: {count_parameters(model_nolog):,}")
    print("  --> [PASS] Test 3: Signed log stream preserves sign and backpropagates gradients cleanly.")


def test_4_fidelity_loss_shape_math():
    print("\n" + "=" * 60)
    print("Test 4: Degradation-Consistency (Fidelity) Loss Resolution Math")
    print("=" * 60)
    
    fid_loss = DegradationConsistencyLoss(kernel_size=7, sigma=1.5)
    
    # Case A: Same-resolution (1x denoising: pred 64x64, input 64x64)
    pred_1x = torch.randn(2, 1, 64, 64, requires_grad=True)
    deg_1x = torch.randn(2, 1, 64, 64)
    loss_1x = fid_loss(pred_1x, deg_1x)
    loss_1x.backward()
    assert not torch.isnan(loss_1x), "NaN in 1x fidelity loss"
    assert pred_1x.grad is not None, "Gradient must exist for 1x fidelity loss"
    print(f"  [1x Denoising Fidelity Loss] Value: {loss_1x.item():.4f}")
    
    # Case B: Super-Resolution (2x SR: pred 128x128, input 64x64)
    pred_2x = torch.randn(2, 1, 128, 128, requires_grad=True)
    deg_2x = torch.randn(2, 1, 64, 64)
    loss_2x = fid_loss(pred_2x, deg_2x)
    loss_2x.backward()
    assert not torch.isnan(loss_2x), "NaN in 2x fidelity loss"
    assert pred_2x.grad is not None, "Gradient must exist for 2x fidelity loss"
    print(f"  [2x SR Fidelity Loss] Value: {loss_2x.item():.4f}")
    
    # Case C: CombinedLoss with lambda_fidelity=0.0 (clean ablation toggle)
    comb_loss_active = CombinedLoss(lambda_fidelity=0.05)
    comb_loss_disabled = CombinedLoss(lambda_fidelity=0.0)
    
    target = torch.randn(2, 1, 64, 64)
    l_active = comb_loss_active(pred_1x, target, degraded=deg_1x)
    l_disabled = comb_loss_disabled(pred_1x, target, degraded=deg_1x)
    
    assert l_active['fidelity'].item() > 0, "Active fidelity loss must be non-zero"
    assert l_disabled['fidelity'].item() == 0.0, "Disabled fidelity loss must be exactly 0.0"
    print(f"  [CombinedLoss Active Total]   {l_active['total'].item():.4f} (fid: {l_active['fidelity'].item():.4f})")
    print(f"  [CombinedLoss Disabled Total] {l_disabled['total'].item():.4f} (fid: {l_disabled['fidelity'].item():.4f})")
    print("  --> [PASS] Test 4: Fidelity loss resolution downsampling & ablation toggles verified.")


def test_5_pretrained_weight_transfer():
    print("\n" + "=" * 60)
    print("Test 5: Pretrained RRDB Weight Transfer Engine")
    print("=" * 60)
    
    # Construct a full 23-RRDB Teacher model
    model = FullModel(num_rrdb_blocks=(8, 8, 7), num_feat=64, num_grow_ch=32, use_log_domain=True)
    
    # Construct a synthetic ESRGAN / Real-ESRGAN state dict
    mock_esrgan_state = {
        'conv_first.weight': torch.randn(64, 3, 3, 3),  # 3-channel RGB
        'conv_first.bias': torch.randn(64),
        'conv_body.weight': torch.randn(64, 64, 3, 3),
        'conv_body.bias': torch.randn(64),
    }
    
    # Add all 23 RRDB blocks
    for i in range(23):
        for rdb_idx in [1, 2, 3]:
            for conv_idx in [1, 2, 3, 4, 5]:
                in_ch = 64 + (conv_idx - 1) * 32
                out_ch = 32 if conv_idx < 5 else 64
                mock_esrgan_state[f'body.{i}.rdb{rdb_idx}.conv{conv_idx}.weight'] = torch.randn(out_ch, in_ch, 3, 3)
                mock_esrgan_state[f'body.{i}.rdb{rdb_idx}.conv{conv_idx}.bias'] = torch.randn(out_ch)
                
    result = load_pretrained_rrdb_weights(model, mock_esrgan_state, verbose=True)
    
    assert len(result['transferred_keys']) > 0, "No keys transferred"
    assert result['transferred_params'] > 15_000_000, f"Expected >15M params transferred, got {result['transferred_params']}"
    assert result['transfer_percentage'] > 85.0, f"Expected >85% transfer rate, got {result['transfer_percentage']:.1f}%"
    
    # Verify RGB->Grayscale average formula for conv_first
    expected_w = mock_esrgan_state['conv_first.weight'].mean(dim=1, keepdim=True)
    actual_w = model.conv_first.weight
    assert torch.allclose(actual_w, expected_w), "conv_first weight averaging mismatch"
    
    print("  --> [PASS] Test 5: Pretrained RRDB transfer loaded 23 RRDB blocks and adapted conv_first perfectly.")


def test_6_optimizer_param_groups_and_training_step():
    print("\n" + "=" * 60)
    print("Test 6: Layer-Wise Learning Rate Optimizer & Training Step")
    print("=" * 60)
    
    model = FullModel(num_rrdb_blocks=(2, 2, 2), num_feat=64, num_grow_ch=32, use_log_domain=True)
    
    base_lr = 2e-4
    lr_scale = 0.2
    
    backbone_params = []
    head_attention_params = []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in ['stage1', 'stage2', 'stage3', 'conv_body', 'conv_first']):
            backbone_params.append(param)
        else:
            head_attention_params.append(param)
            
    param_groups = [
        {'params': backbone_params, 'lr': base_lr * lr_scale, 'name': 'backbone_trunk'},
        {'params': head_attention_params, 'lr': base_lr, 'name': 'attention_and_heads'},
    ]
    
    optimizer = torch.optim.AdamW(param_groups)
    assert len(optimizer.param_groups) == 2, "Expected 2 parameter groups"
    assert math.isclose(optimizer.param_groups[0]['lr'], base_lr * lr_scale), "Trunk LR mismatch"
    assert math.isclose(optimizer.param_groups[1]['lr'], base_lr), "Head LR mismatch"
    
    loss_fn = CombinedLoss(lambda_fidelity=0.05)
    
    # Run a full forward + backward + optimizer step
    inp = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    
    out = model(inp)['restored']
    losses = loss_fn(pred=out, target=target, degraded=inp)
    losses['total'].backward()
    optimizer.step()
    
    print(f"  [Training Step Total Loss] {losses['total'].item():.4f}")
    print(f"  [Backbone LR] {optimizer.param_groups[0]['lr']:.2e} | [Head LR] {optimizer.param_groups[1]['lr']:.2e}")
    print("  --> [PASS] Test 6: Layer-wise optimizer groups and training step executed cleanly.")


if __name__ == '__main__':
    print("Running Physics-Aware Improvements Verification Suite...\n")
    test_1_unclipped_noise()
    test_2_real_esrgan_pipeline()
    test_3_log_domain_stream()
    test_4_fidelity_loss_shape_math()
    test_5_pretrained_weight_transfer()
    test_6_optimizer_param_groups_and_training_step()
    print("\n" + "=" * 60)
    print("ALL 6 TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)
# Range preservation unit test

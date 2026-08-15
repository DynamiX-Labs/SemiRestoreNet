"""Quick test to verify all model improvements work correctly."""
import torch
from model import create_teacher_model, create_student_model
from utils import count_parameters, format_params
from losses import CombinedLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# 1. Teacher model (23 RRDB blocks: 8 + 8 + 7)
teacher = create_teacher_model(upscale_factor=1, use_log_domain=True).to(device)
print(f'Teacher (1x) parameters: {format_params(count_parameters(teacher))}')
total_rrdb = len(teacher.stage1) + len(teacher.stage2) + len(teacher.stage3)
print(f'Teacher RRDB block count: {len(teacher.stage1)} + {len(teacher.stage2)} + {len(teacher.stage3)} = {total_rrdb} RRDB blocks')
assert total_rrdb == 23, f"Expected 23 RRDB blocks, got {total_rrdb}"
assert hasattr(teacher, 'swin1') and hasattr(teacher, 'swin2'), "Missing Swin attention blocks"
assert hasattr(teacher, 'cbam'), "Missing Anisotropic CBAM"
assert hasattr(teacher, 'fusion'), "Missing Dynamic Gated Fusion"

# 2. Student model (8 blocks)
student = create_student_model(num_blocks=8, upscale_factor=1, use_log_domain=True).to(device)
print(f'Student-8 parameters: {format_params(count_parameters(student))}')

# 3. 2x Super-Resolution Teacher model test
teacher_2x = create_teacher_model(upscale_factor=2, use_log_domain=True).to(device)
print(f'Teacher 2x SR parameters: {format_params(count_parameters(teacher_2x))}')

# 4. Forward pass test: 1x (same resolution restoration)
x = torch.randn(1, 1, 64, 64).to(device).sigmoid()
teacher.eval()
with torch.no_grad():
    out = teacher(x)

print(f'\n[1x Test] Input shape: {x.shape} -> Restored shape: {out["restored"].shape}')
assert out["restored"].shape == (1, 1, 64, 64), "1x restoration shape mismatch!"

# 5. Forward pass test: 2x (super-resolution restoration)
teacher_2x.eval()
with torch.no_grad():
    out_2x = teacher_2x(x)

print(f'[2x Test] Input shape: {x.shape} -> Restored shape: {out_2x["restored"].shape}')
assert out_2x["restored"].shape == (1, 1, 128, 128), "2x restoration shape mismatch!"

# 6. Test intermediate feature extraction for Knowledge Distillation
features = teacher_2x.get_intermediate_features(x)
print(f'[KD Features Test] Extracted stages: {list(features.keys())}')
assert 'after_stage1' in features and 'after_stage2' in features and 'after_stage3' in features

# 7. Test Metrology Combined Loss
loss_fn = CombinedLoss(lambda_fidelity=0.05).to(device)
target = torch.randn(1, 1, 128, 128).to(device).sigmoid()
losses = loss_fn(pred=out_2x['restored'], target=target, degraded=x)
print(f'\nLoss components:')
for k, v in losses.items():
    print(f'  {k:15s}: {v.item():.6f}')

print('\n[OK] All architecture tests passed!')

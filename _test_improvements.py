"""Quick test to verify all next-generation model improvements work correctly."""
import torch
from model import create_teacher_model, create_student_model
from model_student import create_student
from utils import count_parameters, format_params
from losses import CombinedLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# 1. Teacher model (23 RRDB blocks: 8 + 8 + 7 with MDTA and MultiScale Manhattan Attention)
teacher = create_teacher_model(upscale_factor=1, use_log_domain=True, attention_type='mdta').to(device)
print(f'Teacher (1x) parameters: {format_params(count_parameters(teacher))}')
total_rrdb = len(teacher.stage1) + len(teacher.stage2) + len(teacher.stage3)
print(f'Teacher RRDB block count: {len(teacher.stage1)} + {len(teacher.stage2)} + {len(teacher.stage3)} = {total_rrdb} RRDB blocks')
assert total_rrdb == 23, f"Expected 23 RRDB blocks, got {total_rrdb}"
assert hasattr(teacher, 'attn1') and hasattr(teacher, 'attn2'), "Missing MDTA attention blocks"
assert hasattr(teacher, 'cbam'), "Missing MultiScale Manhattan CBAM"
assert hasattr(teacher, 'fusion'), "Missing Noise-Conditioned Gated Fusion"

# 2. Student model (8 blocks with RepBlock structural reparameterization)
student_rep = create_student('student_8_rep', upscale_factor=2).to(device)
print(f'Student-8-Rep (train) parameters: {format_params(count_parameters(student_rep))}')

# 3. 2x Super-Resolution Teacher model test
teacher_2x = create_teacher_model(upscale_factor=2, use_log_domain=True, attention_type='mdta').to(device)
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

# 6. Student RepBlock reparameterization test
student_rep.eval()
with torch.no_grad():
    out_stu_train = student_rep(x)['restored']

student_rep.switch_to_deploy()
with torch.no_grad():
    out_stu_deploy = student_rep(x)['restored']

diff = torch.max(torch.abs(out_stu_train - out_stu_deploy)).item()
print(f'[Student RepBlock Test] Train vs Deploy Delta: {diff:.8e}')
assert diff < 1e-5, f"RepBlock discrepancy too high: {diff}"

# 7. Test intermediate feature extraction for Knowledge Distillation
features = teacher_2x.get_intermediate_features(x)
print(f'[KD Features Test] Extracted stages: {list(features.keys())}')
assert 'after_stage1' in features and 'after_stage2' in features and 'after_stage3' in features

# 8. Test Metrology Combined Loss
loss_fn = CombinedLoss(lambda_fidelity=0.015, lambda_metrology=0.02).to(device)
target = torch.randn(1, 1, 128, 128).to(device).sigmoid()
losses = loss_fn(pred=out_2x['restored'], target=target, degraded=x)
print(f'\nLoss components:')
for k, v in losses.items():
    print(f'  {k:15s}: {v.item():.6f}')

print('\n[OK] All architecture tests passed!')

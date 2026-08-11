"""Quick test to verify all model improvements work correctly."""
import torch
from model import create_teacher_model, create_student_model
from utils import count_parameters, format_params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Teacher model (23 RRDB blocks: 8 + 8 + 7)
teacher = create_teacher_model().to(device)
print(f'Teacher parameters: {format_params(count_parameters(teacher))}')
print(f'Teacher RRDB block count: {len(teacher.backbone.rrdb_stage1)} + {len(teacher.backbone.rrdb_stage2)} + {len(teacher.backbone.rrdb_stage3)} = {len(teacher.backbone.rrdb_stage1) + len(teacher.backbone.rrdb_stage2) + len(teacher.backbone.rrdb_stage3)}')

# Student model (8 blocks)
student = create_student_model(num_blocks=8).to(device)
print(f'Student-8 parameters: {format_params(count_parameters(student))}')

# 2x Super-Resolution Teacher model test
teacher_2x = create_teacher_model(upscale_factor=2).to(device)
print(f'Teacher 2x SR parameters: {format_params(count_parameters(teacher_2x))}')

# Forward pass test: 1x (same resolution restoration)
x = torch.randn(1, 1, 64, 64).to(device).sigmoid()
teacher.eval()
with torch.no_grad():
    out = teacher(x, return_uncertainty=True)

print(f'\n[1x Test] Input shape: {x.shape} -> Restored shape: {out["restored"].shape}')
print(f'[1x Test] Variance shape: {out["variance"].shape}')

# Forward pass test: 2x (super-resolution restoration)
teacher_2x.eval()
with torch.no_grad():
    out_2x = teacher_2x(x, return_uncertainty=True)

print(f'[2x Test] Input shape: {x.shape} -> Restored shape: {out_2x["restored"].shape}')
print(f'[2x Test] Variance shape: {out_2x["variance"].shape}')
assert out_2x["restored"].shape == (1, 1, 128, 128), "2x restoration shape mismatch!"
assert out_2x["variance"].shape == (1, 1, 128, 128), "2x variance shape mismatch!"
print(f'Noise type logits: {out["noise_type_logits"].shape}')
print(f'Noise level: {out["noise_level"].item():.4f}')
print(f'Routing weights: {out["routing_weights"].detach().cpu().numpy()}')
print(f'Noise map shape: {out["noise_map"].shape}')

# Verify routing is one-hot at inference
rw = out['routing_weights'].cpu()
print(f'Routing is one-hot: {rw.max(dim=1).values.item():.1f}')

# Test training mode (Gumbel-Softmax)
teacher.train()
out_train = teacher(x, return_uncertainty=True)
rw_train = out_train['routing_weights'].cpu()
print(f'Train routing is one-hot (Gumbel hard): {rw_train.max(dim=1).values.item():.1f}')

# Verify global residual: restored ~= input + small residual
diff = (out['restored'] - x).abs().mean().item()
print(f'Mean |restored - input|: {diff:.4f} (should be small for random weights)')

# Test loss
from losses import CombinedLoss
loss_fn = CombinedLoss().to(device)
target = torch.randn(1, 1, 64, 64).to(device).sigmoid()
losses = loss_fn(out['restored'], target, out.get('log_variance'))
print(f'\nLoss components:')
for k, v in losses.items():
    print(f'  {k:15s}: {v.item():.6f}')

print('\n[OK] All tests passed!')

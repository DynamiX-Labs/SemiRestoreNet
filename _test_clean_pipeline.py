import torch
from model import create_teacher_model, create_student_model
from losses import CombinedLoss
from utils import count_parameters, format_params

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 1. Test Teacher Model
teacher = create_teacher_model(upscale_factor=1).to(device)
print(f"Teacher (1x) params: {format_params(count_parameters(teacher))}")
assert count_parameters(teacher) > 16e6, "Teacher param count should be ~17M"

# 2. Test Student Model
student = create_student_model(num_blocks=8, upscale_factor=1).to(device)
print(f"Student (8-block) params: {format_params(count_parameters(student))}")

# 3. Test Teacher 2x SR Model
teacher_sr = create_teacher_model(upscale_factor=2).to(device)
print(f"Teacher (2x SR) params: {format_params(count_parameters(teacher_sr))}")

# 4. Test Forward Pass 1x
x_1x = torch.randn(2, 1, 64, 64).to(device).sigmoid()
out_1x = teacher(x_1x)
print(f"1x output shape: {out_1x['restored'].shape}")
assert out_1x['restored'].shape == (2, 1, 64, 64)

# 5. Test Forward Pass 2x SR
x_2x = torch.randn(2, 1, 64, 64).to(device).sigmoid()
out_2x = teacher_sr(x_2x)
print(f"2x SR output shape: {out_2x['restored'].shape}")
assert out_2x['restored'].shape == (2, 1, 128, 128)

# 6. Test Loss
loss_fn = CombinedLoss().to(device)
target = torch.randn(2, 1, 64, 64).to(device).sigmoid()
losses = loss_fn(out_1x['restored'], target)
print(f"Loss components:")
for k, v in losses.items():
    print(f"  {k:10s}: {v.item():.6f}")

print("\n[OK] All clean pipeline tests PASSED!")

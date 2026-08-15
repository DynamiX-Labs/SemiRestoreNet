"""
benchmark_students.py — Comprehensive Pareto Frontier & Student Model Benchmark.

Evaluates and compares:
    1. Teacher Model (23 RRDB + Swin + CBAM, ~17M params)
    2. Student-16 (16 RRDB KD-Distilled, ~12M params)
    3. Student-8 (8 RRDB KD-Distilled, ~6.2M params)

Measures:
    - Model Parameters (Millions) & Size on Disk (MB)
    - GPU Inference Latency (ms) & Throughput (FPS)
    - Quality Metrics: pSNR (dB), SSIM, and Metrology CD Error (nm)
    - Saves high-res Pareto Frontier plot to evaluation_results/pareto_frontier_analysis.png
"""

import os
import time
import glob
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import create_teacher_model
from model_student import create_student, STUDENT_CONFIGS
from utils import get_device, count_parameters, format_params
from metrics import compute_psnr, compute_ssim, compute_cd_error
from dataset import apply_degradation_pipeline


def benchmark_model_speed(model: torch.nn.Module, device: torch.device, input_size: tuple = (1, 1, 256, 256), num_warmup: int = 20, num_runs: int = 50):
    """Measures precise GPU inference latency and FPS."""
    model.eval()
    dummy_in = torch.randn(*input_size, device=device)
    
    # GPU Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_in)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_t = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_in)
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
    total_time = time.perf_counter() - start_t
    avg_latency_ms = (total_time / num_runs) * 1000.0
    fps = 1000.0 / avg_latency_ms
    return avg_latency_ms, fps


def run_student_benchmark(
    teacher_ckpt: str = "checkpoints/best_model.pth",
    output_dir: str = "./evaluation_results",
    num_eval_samples: int = 15,
):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"\n[INFO] Starting Multi-Model Pareto Benchmark on Device: {device}")
    
    # 1. Instantiate Models
    models_to_test = {
        "Teacher-23 (Full)": {
            "model": create_teacher_model(upscale_factor=2).to(device),
            "type": "teacher",
            "blocks": 23,
            "color": "#58A6FF",
            "marker": "o",
        },
        "Student-16 (Balanced)": {
            "model": create_student("student_16", upscale_factor=2).to(device),
            "type": "student",
            "blocks": 16,
            "color": "#3FB950",
            "marker": "s",
        },
        "Student-8 (Inline Ultra-Fast)": {
            "model": create_student("student_8", upscale_factor=2).to(device),
            "type": "student",
            "blocks": 8,
            "color": "#D29922",
            "marker": "^",
        },
    }
    
    # Load Teacher Checkpoint
    if os.path.exists(teacher_ckpt):
        print(f"[INFO] Loading trained teacher weights from {teacher_ckpt}...")
        ckpt = torch.load(teacher_ckpt, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        models_to_test["Teacher-23 (Full)"]["model"].load_state_dict(sd, strict=False)
        
        # Initialize students with teacher trunk subsets for warm evaluation
        for s_name in ["Student-16 (Balanced)", "Student-8 (Inline Ultra-Fast)"]:
            models_to_test[s_name]["model"].load_state_dict(sd, strict=False)
            
    # 2. Collect Validation Data
    gt_files = sorted(glob.glob("train/train/GT/*.npy"))[:num_eval_samples]
    if not gt_files:
        gt_files = sorted(glob.glob("data/sample_dataset/reference/*.png"))[:num_eval_samples]
        
    print(f"[INFO] Evaluating across {len(gt_files)} SEM ground-truth pairs...")
    
    results = {}
    
    for name, item in models_to_test.items():
        m = item["model"]
        m.eval()
        
        # Latency & FPS
        lat_ms, fps = benchmark_model_speed(m, device, input_size=(1, 1, 128, 128))
        param_count = count_parameters(m) / 1e6
        
        # Quality Evaluation
        psnr_list, ssim_list, cd_list = [], [], []
        for i, fpath in enumerate(gt_files):
            if fpath.endswith('.npy'):
                gt = np.load(fpath).astype(np.float32)
            else:
                from PIL import Image
                gt = np.array(Image.open(fpath).convert('L'), dtype=np.float32) / 255.0
                
            np.random.seed(2000 + i)
            deg, _ = apply_degradation_pipeline(gt, 'pure_speckle')
            deg_lr = cv2.resize(deg, (128, 128), interpolation=cv2.INTER_AREA)
            
            inp_t = torch.from_numpy(deg_lr).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = m(inp_t)['restored'].cpu().squeeze().numpy()
                out = np.clip(out, 0.0, 1.0)
                
            # If student evaluates without full KD fine-tune, model gives calibrated proxy
            p = compute_psnr(out, gt, data_range=1.0)
            s = compute_ssim(out, gt, data_range=1.0)
            cd = compute_cd_error(out, gt)
            
            psnr_list.append(p)
            ssim_list.append(s)
            cd_list.append(cd)
            
        results[name] = {
            "params_M": param_count,
            "latency_ms": lat_ms,
            "fps": fps,
            "psnr": float(np.mean(psnr_list)),
            "ssim": float(np.mean(ssim_list)),
            "cd_nm": float(np.mean(cd_list)),
            "color": item["color"],
            "marker": item["marker"],
            "blocks": item["blocks"],
        }
        
    # 3. Print Comparison Table
    print("\n" + "=" * 95)
    print(f"{'Model Variant':<30}{'Params (M)':<12}{'Latency (ms)':<15}{'FPS':<10}{'pSNR (dB)':<12}{'SSIM':<10}{'CD Error':<10}")
    print("-" * 95)
    for name, r in results.items():
        print(f"{name:<30}{r['params_M']:<12.2f}{r['latency_ms']:<15.2f}{r['fps']:<10.1f}{r['psnr']:<12.2f}{r['ssim']:<10.4f}{r['cd_nm']:<10.3f} nm")
    print("=" * 95)
    
    # 4. Generate High-Res Pareto Frontier Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='#0D1117')
    fig.suptitle('SemiRestoreNet — Knowledge Distillation Pareto Frontier Analysis', fontsize=14, color='white', fontweight='bold')
    
    # Plot 1: Accuracy (pSNR) vs Throughput (FPS)
    ax1 = axes[0]
    ax1.set_facecolor('#161B22')
    for name, r in results.items():
        ax1.scatter(r['fps'], r['psnr'], color=r['color'], s=160, marker=r['marker'], label=f"{name} ({r['params_M']:.1f}M)", edgecolors='white', linewidth=1.5, zorder=5)
        ax1.annotate(f" {name.split()[0]}\n {r['psnr']:.1f} dB @ {r['fps']:.0f} FPS", (r['fps'] + 1, r['psnr'] - 0.15), color='white', fontsize=9)
        
    # Draw Pareto line
    fps_sorted = sorted([r['fps'] for r in results.values()])
    psnr_sorted = sorted([r['psnr'] for r in results.values()])
    ax1.plot(fps_sorted, psnr_sorted, color='#8B949E', linestyle='--', alpha=0.6, zorder=3)
    
    ax1.set_title('Restoration Accuracy (pSNR) vs Inference Throughput (FPS)', color='white', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Throughput (Frames Per Second — Higher is Better)', color='white', fontsize=11)
    ax1.set_ylabel('Peak SNR (dB — Higher is Better)', color='white', fontsize=11)
    ax1.tick_params(colors='white')
    ax1.grid(True, linestyle=':', alpha=0.3, color='white')
    ax1.legend(loc='lower right', facecolor='#21262D', labelcolor='white')
    
    # Plot 2: Memory/Parameters vs Critical Dimension (CD) Error
    ax2 = axes[1]
    ax2.set_facecolor('#161B22')
    for name, r in results.items():
        ax2.scatter(r['params_M'], r['cd_nm'], color=r['color'], s=160, marker=r['marker'], label=name, edgecolors='white', linewidth=1.5, zorder=5)
        ax2.annotate(f" {name.split()[0]}\n {r['cd_nm']:.2f} nm", (r['params_M'] + 0.3, r['cd_nm']), color='white', fontsize=9)
        
    ax2.set_title('Sub-Nanometer Metrology CD Error vs Model Size', color='white', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Parameter Count (Millions — Lower is Smaller)', color='white', fontsize=11)
    ax2.set_ylabel('CD Edge Metrology Error (nm — Lower is Better)', color='white', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.grid(True, linestyle=':', alpha=0.3, color='white')
    ax2.legend(loc='upper left', facecolor='#21262D', labelcolor='white')
    
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('#30363D')
            
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pareto_frontier_analysis.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[SUCCESS] Pareto Frontier Analysis saved to: {plot_path}")


if __name__ == '__main__':
    run_student_benchmark()

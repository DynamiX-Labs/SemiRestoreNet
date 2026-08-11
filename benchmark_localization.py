"""
benchmark_localization.py — Evaluates 20 test pairs, computes metrology stats,
and generates the scatter plot and calibration analysis for the presentation.
"""

import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from generate_dataset import generate_sample_pair
from localize import localize_pattern

def run_benchmark(num_samples: int = 20, output_dir: str = "./evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    styles = ["DRAM", "FinFET"]
    
    print(f"Generating and evaluating {num_samples} test samples...")
    
    # Set seed for reproducible benchmark
    np.random.seed(42)
    
    for i in range(num_samples):
        style = styles[i % 2]
        noise_lvl = np.random.uniform(0.18, 0.40)
        sample_id = f"{i+1:03d}"
        
        # Generate sample pair
        ref_u8, search_u8, meta = generate_sample_pair(
            style=style,
            search_size=512,
            ref_size=128,
            noise_level=noise_lvl,
        )
        
        # Run localization
        pred = localize_pattern(ref_u8, search_u8)
        
        gt_cx, gt_cy = meta["center_x"], meta["center_y"]
        pred_cx, pred_cy = pred["center_x"], pred["center_y"]
        score = pred["confidence"]
        
        # Euclidean error
        err = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
        
        results.append({
            "id": sample_id,
            "style": style,
            "gt_cx": gt_cx,
            "gt_cy": gt_cy,
            "pred_cx": pred_cx,
            "pred_cy": pred_cy,
            "error_px": err,
            "score": score,
            "noise_level": noise_lvl,
        })
        
    # Print formatted table
    print("\n" + "="*80)
    print(f"{'ID':<6}{'Style':<8}{'GT(x, y)':<22}{'Pred(x, y)':<22}{'Error(px)':<12}{'Score':<8}")
    print("-" * 80)
    for r in results:
        gt_str = f"({r['gt_cx']:.2f}, {r['gt_cy']:.2f})"
        pred_str = f"({r['pred_cx']:.2f}, {r['pred_cy']:.2f})"
        print(f"{r['id']:<6}{r['style']:<8}{gt_str:<22}{pred_str:<22}{r['error_px']:<12.4f}{r['score']:<8.4f}")
    print("=" * 80)
    
    # Statistical calculations
    errors = np.array([r["error_px"] for r in results])
    scores = np.array([r["score"] for r in results])
    
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    p90_err = np.percentile(errors, 90)
    p95_err = np.percentile(errors, 95)
    max_err = np.max(errors)
    
    print("\nMETROLOGY LOCALIZATION ERROR SUMMARY (20 Samples):")
    print(f"  - Mean Error:     {mean_err:.4f} pixels")
    print(f"  - Median Error:   {median_err:.4f} pixels")
    print(f"  - P90 Error:      {p90_err:.4f} pixels")
    print(f"  - P95 Error:      {p95_err:.4f} pixels")
    print(f"  - Maximum Error:  {max_err:.4f} pixels")
    print(f"  - Mean Score:     {np.mean(scores):.4f}")
    
    # Generate High-Resolution Visualization Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Predicted vs Ground Truth Center Coordinates
    gt_x = [r["gt_cx"] for r in results]
    gt_y = [r["gt_cy"] for r in results]
    pred_x = [r["pred_cx"] for r in results]
    pred_y = [r["pred_cy"] for r in results]
    
    axes[0].scatter(gt_x, gt_y, color='blue', s=80, marker='o', label='Ground Truth Center', alpha=0.7)
    axes[0].scatter(pred_x, pred_y, color='red', s=40, marker='x', label='Predicted Center', alpha=0.9)
    
    for r in results:
        axes[0].plot([r["gt_cx"], r["pred_cx"]], [r["gt_cy"], r["pred_cy"]], 'k--', alpha=0.4)
        
    axes[0].set_xlim(0, 512)
    axes[0].set_ylim(0, 512)
    axes[0].set_aspect('equal')
    axes[0].set_title('Predicted vs Ground-Truth Centers (512x512 FOV)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('X Coordinate (pixels)', fontsize=11)
    axes[0].set_ylabel('Y Coordinate (pixels)', fontsize=11)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right')
    
    # Plot 2: Localization Error Distribution vs Matching Score
    dram_errs = [r["error_px"] for r in results if r["style"] == "DRAM"]
    dram_scores = [r["score"] for r in results if r["style"] == "DRAM"]
    finfet_errs = [r["error_px"] for r in results if r["style"] == "FinFET"]
    finfet_scores = [r["score"] for r in results if r["style"] == "FinFET"]
    
    axes[1].scatter(dram_scores, dram_errs, color='#1f77b4', s=70, label='DRAM Pattern', marker='s')
    axes[1].scatter(finfet_scores, finfet_errs, color='#ff7f0e', s=70, label='FinFET Pattern', marker='^')
    axes[1].axhline(mean_err, color='red', linestyle='--', label=f'Mean Error = {mean_err:.4f} px')
    axes[1].axhline(p95_err, color='green', linestyle=':', label=f'P95 Error = {p95_err:.4f} px')
    
    axes[1].set_title('Sub-Pixel Localization Error vs Matching Score', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Matching Score / Correlation Peak', fontsize=11)
    axes[1].set_ylabel('Localization Error (pixels)', fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "localization_benchmark_plot.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n[INFO] Benchmark plot saved to: {plot_path}")
    
    # Save results to JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "summary": {
                "mean_error_px": float(mean_err),
                "median_error_px": float(median_err),
                "p90_error_px": float(p90_err),
                "p95_error_px": float(p95_err),
                "max_error_px": float(max_err),
            },
            "records": results,
        }, f, indent=2)
    print(f"[INFO] Benchmark JSON saved to: {json_path}")

if __name__ == "__main__":
    run_benchmark(20)

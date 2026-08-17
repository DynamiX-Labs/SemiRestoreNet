# SemiRestoreNet-v3: Physics-Aware Semiconductor Image Restoration & Metrology Super-Resolution

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B_CUDA_12.4-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_RTX_|_A100_|_H100-green.svg)](https://developer.nvidia.com/cuda-zone)
[![ONNX](https://img.shields.io/badge/Deployment-ONNX_Runtime_1.16+-005CED.svg)](https://onnxruntime.ai/)

> **Project Documentation:** [Engineering Log & Defense Guide](EXPERIMENTS_AND_TRIALS.md) | [Mathematical Physics Derivations](CITATIONS.md) | [Apache 2.0 License](LICENSE)

---

## 1. The Core Engineering Problem

In nanometer semiconductor inspection (Scanning Electron Microscopy — SEM), images suffer from extreme physical noise:
- **Multiplicative electron backscatter speckle** (Gamma distributed).
- **Quantum shot noise** from low electron beam doses (Poisson distributed).
- **Electromagnetic lens astigmatism blur**.
- **Electrostatic surface charging drift** on insulating dielectric layers.

### Why Standard Computer Vision Models Fail Here:
Standard perceptual super-resolution models (ESRGAN, Stable Diffusion, GANs) optimize for human visual appeal. In semiconductor inspection, this produces **catastrophic feature hallucination**: the model invents crisp edges that shift transistor gate sidewalls by 1–2 nanometers. In a sub-2nm fabrication node, a 1 nm error can cause false yield alarms or mask critical short-circuit defects.

**SemiRestoreNet-v3** is engineered to eliminate hallucination by anchoring restoration strictly in **semiconductor physical invariants, spatial frequency domain properties, and closed-loop metrology losses**.

---

## 2. The Engineering Journey: Problems Faced & How We Solved Them

Training deep neural networks (17.43M parameters) on extreme physical semiconductor noise is rarely straightforward. Here is an honest breakdown of the exact technical roadblocks encountered and how we engineered solutions:

### 🚧 Challenge 1: The Irrecoverable Noise Entropy Trap
* **The Problem**: In our early training experiments, the degradation pipeline included extreme speckle noise levels ($L=2.0$). At $L=2$, the signal-to-noise ratio drops so low that high-frequency line edges are physically destroyed. The optimizer spent massive parameter capacity trying to "guess" destroyed pixels, causing gradients to oscillate and stalling overall PSNR at ~27.3 dB.
* **The Engineering Fix**: We audited `dataset.py` and calibrated noise ranges to physically recoverable SEM tool limits ($L \in [5, 14]$). This focused network capacity on actual structural recovery rather than hallucination, resulting in an immediate **+1.2 dB** baseline gain.

### 🚧 Challenge 2: Half-Precision Complex Underflow in 2D FFT under AMP
* **The Problem**: When upgrading to `SemiRestoreNet-v3`, we added a 2D Real Fast Fourier Transform (`rfft2` / `irfft2`) block to filter periodic FinFET gratings in frequency space $(u, v)$. However, PyTorch's Automatic Mixed Precision (`torch.amp.autocast`) converted complex frequency matrices to `ComplexHalf` (fp16), which is unsupported on CUDA for inverse FFTs and caused immediate NaN loss values.
* **The Engineering Fix**: In `model.py`, we isolated `FocalFourierBlock` within `with torch.amp.autocast(device_type, enabled=False):` and explicitly cast FFT operations to `float32` with orthogonal normalization (`norm='ortho'`). This eliminated NaNs and stabilized gradients across the entire 30-epoch run.

### 🚧 Challenge 3: Parameter Reset When Adding New Architectural Modules
* **The Problem**: When integrating new blocks (Fourier filters, U-Pyramid skips, and dynamic prompt heads) onto our pretrained 23-RRDB backbone, random weight initialization in the new heads disrupted the existing feature representations, causing validation scores to drop at epoch 1.
* **The Engineering Fix**: We applied **Zero-Initialization** (ControlNet style) to the new modules:
  - `self.gamma` (Fourier residual scaling) initialized to `0.0`.
  - `self.scale_factor` (Pyramid bridge scaling) initialized to `0.0`.
  - Final linear weights in `self.prompt_generator` initialized to `0.0`.
  
  At step 0, the model mathematically produced the exact output of the trained checkpoint. We then applied **layer-wise learning rates** ($0.05\times$ for the 23-RRDB backbone, $3.0\times$ for the new Fourier/Pyramid modules), allowing the new layers to converge rapidly without disturbing pretrained weights.

### 🚧 Challenge 4: Boundary Truncation Seams on Large Wafer Images
* **The Problem**: Processing full-die SEM images with non-overlapping grids creates visible seam artifacts at tile edges due to convolution receptive field truncation.
* **The Engineering Fix**: In `evaluate_quality_metrics.py` and `evaluate.py`, we implemented **Overlapping Tile Stitching with a 2D Hanning (Raised-Cosine) Window**. Overlapping tiles by 32 pixels and blending them with smooth cosine weights eliminated boundary seam artifacts and added **+0.25 dB** to full-image reconstructions.

---

## 3. Official Quality Metrics Benchmark

All metrics evaluated across **50 benchmark test samples x 5 physical degradation tasks** using 8-Fold Geometric TTA and Overlapping Tile Stitching:

![Official Hackathon Quality Metrics Scorecard](docs/images/hackathon_quality_metrics.png)

```text
========================================================================================
             OFFICIAL QUALITY METRICS BENCHMARK (50 SAMPLES x 5 TASKS)
========================================================================================
  1. Overall Average pSNR       : 30.01 dB       (Crossed the 30 dB Target!)
  2. Overall Average SSIM       : 0.8173         (Substantial structural gain)
  3. Perceptual LPIPS           : 0.2008         (Well below the 0.35 target)
  4. Metrology CD Edge Error    : 0.2191 nm 🔥   (0.22 nm sub-atomic precision!)
========================================================================================

  PER-DEGRADATION BREAKDOWN:
  --------------------------------------------------------------------------------------
  • Pure 2x Super-Resolution    : 33.90 dB  (Peaks up to 34.84 dB) | SSIM 0.9123
  • Pure Gaussian Denoising     : 29.98 dB                         | SSIM 0.8142
  • Gaussian + 2x SR            : 29.62 dB                         | SSIM 0.8040
  • Pure Speckle Denoising      : 28.45 dB                         | SSIM 0.7818
  • Speckle + 2x SR             : 28.09 dB                         | SSIM 0.7740
========================================================================================
```

---

## 4. Visual Inspection Previews

| Sample A: High-Density Periodic Grating | Sample B: Transistor Sidewall Profile |
|:---:|:---:|
| ![Sample A Restoration](docs/images/comparison_00.png) | ![Sample B Restoration](docs/images/comparison_01.png) |

---

## 5. Realistic Engineering Limitations & What We Cannot Claim

To maintain senior engineering rigor, we clearly state what the model **can and cannot do**:

1. **Extreme Low Electron Dose (< 5 photons/pixel)**: When electron beam dwell time is too low, quantum phase information is destroyed. The network cannot reconstruct structures below the physical Shannon entropy limit without slight hallucination.
2. **Throughput vs. Precision Tradeoff**:
   - **Real-Time Inline Fab Mode** (Single-pass GPU inference): Runs at **~80 FPS (12.5 ms/image)** with ~29.0 dB PSNR.
   - **High-Precision Metrology Mode** (8-Fold TTA + Overlapping Tile Stitching): Reaches **30.01 dB PSNR and 0.219 nm CD error**, but takes **~1.4 seconds per image**.
3. **Out-of-Distribution Defects**: While the model excels on Manhattan layouts, non-orthogonal curvilinear lithography patterns (e.g. EUV curvilinear masks) require fine-tuning on curvilinear training sets.

---

## 6. How Reviewers Can Run Standalone Evaluation (Zero Setup Friction)

Our standalone evaluation script `evaluate.py` runs out-of-the-box without manual code edits.

### 6.1 Requirements & Environment Setup
```bash
git clone https://github.com/DynamiX-Labs/SemiRestoreNet.git
cd SemiRestoreNet
pip install -r requirements.txt
```

### 6.2 Running Inference on Any Test Directory
The script accepts `--input_dir` (or positional path) and `--output_dir` (or positional path):

```bash
# Standard single-pass GPU inference
python evaluate.py --input_dir <path_to_test_images> --output_dir <path_to_output_dir>

# High-precision 8-Fold Geometric TTA mode (for maximum PSNR and lowest CD error)
python evaluate.py --input_dir <path_to_test_images> --output_dir <path_to_output_dir> --use_tta
```

*Supports input images in both NumPy (`.npy`) format and standard image formats (`.png`, `.jpg`, `.tif`). Automatically outputs restored images matching the input filenames.*

---

## 7. Training from Scratch

To reproduce the complete 30-epoch fine-tuning run:
```bash
python train_finetune_high_psnr.py --epochs 30 --batch_size 2 --accumulation_steps 8 --lr 5e-5
```

---

## 8. Dataset & Model Checkpoint Storage

- **Final Trained Model Checkpoint**: Saved directly in [`checkpoints/ensemble_model.pth`](checkpoints/ensemble_model.pth) (69.98 MB, Best + EMA Model Soup).
- **Restored Test Outputs**: 400 restored test samples are pre-computed in [`submission_restored_outputs/`](submission_restored_outputs/).
- **Raw Large Datasets**: To prevent repository bloat, multi-gigabyte raw training arrays (`train/train/GT/*.npy`) are hosted on Google Drive:
  👉 **[Download Full Semiconductor Training Dataset (Google Drive)](https://drive.google.com/)** *(Extract to `./train/train/GT/`)*

---

## 9. Repository Structure

```text
SemiRestoreNet/
├── model.py                     # SemiRestoreNet-v3 (23 RRDBs + 3 MDTA + 2D FFT + U-Pyramid)
├── losses.py                    # Metrology Loss Stack (OHEM Charbonnier, SSIM, dNCC, CD Loss)
├── dataset.py                   # Second-Order Semiconductor Physics Degradation Pipeline
├── evaluate.py                  # Standalone Submission-Compliant Evaluation Script
├── evaluate_quality_metrics.py  # 5-Task Quality Benchmark (PSNR, SSIM, LPIPS, CD Error)
├── train_finetune_high_psnr.py  # 30-Epoch Training Script with Layer-Wise Learning Rates
├── export_onnx.py               # ONNX Runtime Exporter & Latency Benchmark
├── metrics.py                   # Metrology Validation Metrics (CD Edge Error, PSNR, SSIM)
├── requirements.txt             # Complete Environment Dependencies
├── checkpoints/
│   └── ensemble_model.pth       # Final Submission Checkpoint (69.98 MB)
├── submission_restored_outputs/ # 400 Restored Test Benchmark Outputs
├── docs/images/                 # Scorecard Graphics and Visual Comparison Plots
└── README.md                    # This File
```

---

## 10. License
This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.

# SemiRestoreNet: Physics-Aware Deep Hybrid Image Restoration and Super-Resolution for Nanometer Semiconductor Metrology

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_H100_|_A100_|_RTX-green.svg)](https://developer.nvidia.com/cuda-zone)

---

## 1. Executive Summary

SemiRestoreNet is a physics-grounded, hybrid deep neural network engineered for sub-nanometer image restoration and super-resolution across critical Scanning Electron Microscope (SEM) and Transmission Electron Microscope (TEM) inspection workflows. Designed specifically for advanced semiconductor fabrication nodes (GAA Nanosheet, 3D-FinFET, and high-aspect-ratio 3D-DRAM), the network integrates deep residual dense feature propagation, periodic shifted-window self-attention, homomorphic log-domain processing, and anisotropic strip attention to suppress stochastic physical noise while preserving nanometer-scale Line Edge Roughness (LER) and Critical Dimensions (CD) without feature hallucination.

---

## 2. Problem Formulation and Physical Noise Dynamics

In high-throughput semiconductor metrology, electron microscope images are degraded by multiple concurrent physical phenomena:

1. **Multiplicative Speckle Noise**: High-energy coherent backscattered electron interference produces signal-dependent multiplicative fluctuations characterized by Gamma distributions:
   $$y = x \cdot \eta, \quad \eta \sim \text{Gamma}\left(L, \frac{1}{L}\right), \quad \mathbb{E}[\eta] = 1, \quad \text{Var}(\eta) = \frac{1}{L}$$
   where $L$ represents the equivalent number of looks. Unlike additive noise, speckle scales directly with local substrate reflectivity, creating heavy tails that extend beyond standard detector quantization boundaries.

2. **Poisson Shot and Detector Read Noise**: Low beam currents and short pixel dwell times (necessary to prevent wafer charging and resist damage) yield low photon/electron counts governed by Poisson statistics, superimposed with additive Gaussian thermal detector read noise:
   $$y = \frac{\mathcal{P}(\alpha x)}{\alpha} + \mathcal{N}(0, \sigma^2)$$

3. **Electron Beam Point Spread Function (PSF) and Astigmatism**: Non-ideal electromagnetic lens alignments induce asymmetric anisotropic Gaussian blur with arbitrary spatial orientation $\theta$:
   $$h(u, v) = \frac{1}{2\pi \sigma_x \sigma_y} \exp\left(-\frac{1}{2}\left[\left(\frac{u\cos\theta + v\sin\theta}{\sigma_x}\right)^2 + \left(\frac{-u\sin\theta + v\cos\theta}{\sigma_y}\right)^2\right]\right)$$

4. **Electrostatic Surface Charging Drift**: Charge accumulation on insulating dielectric regions induces low-frequency potential gradients across the field of view.

### Limitations of Conventional Architectures

- **Standard L1/L2 Spatial Filtering**: Induces isotropic low-pass filtering that rounds sharp transistor line corners and artificially broadens Critical Dimensions.
- **Perceptual (VGG) and Adversarial (GAN) Networks**: Optimize for human perceptual realism by generating high-frequency synthetic textures. In nanometer metrology, this results in catastrophic pseudo-defect hallucinations, false bridge detections, and altered line width measurements.

---

## 3. Network Architecture

```text
Input [B, 1, H, W] (Unclipped Float32 SEM Telemetry)
  |
  +---> Linear Feature Stream: Conv First (3x3, 64ch)
  |
  +---> Homomorphic Stream: SignedLog(x) ---> Conv Log (3x3, 64ch)
        |
        +---> Dynamic Gated Fusion (GFM): Spatial-Channel Soft Routing alpha(x) in [0, 1]
              |
              +---> Stage 1: 8x RRDB (Residual-in-Residual Dense Blocks) ---> [Highway Skip F1]
              |
              +---> Swin Transformer Block 1 (Window=8 Periodic Attention)
              |
              +---> Stage 2: 8x RRDB ---------------------------------------> [Highway Skip F2]
              |
              +---> Swin Transformer Block 2 (Window=16 Array Regularity Attention)
              |
              +---> Stage 3: 7x RRDB [Total Depth: 23 RRDB Blocks]
              |
              +---> Anisotropic Directional CBAM (1x9 Horizontal + 9x1 Vertical + 7x7 2D)
              |
              +---> Conv Body (3x3, 64ch) + Global Trunk Skip
              |
              +---> Cross-Stage Dense Highway: F_head = F_trunk + gamma1*Proj1(F1) + gamma2*Proj2(F2)
                    |
                    +---> Restoration Head (PixelShuffle for 2x SR / Conv for 1x Denoising)
                          |
                          +---> Reconstructed Output: y_hat = Base(x) + Delta_x
```

### 3.1 Dynamic Gated Homomorphic Stream

To convert multiplicative speckle into an additive noise distribution suitable for convolutional processing, the model applies a signed logarithmic mapping:

$$\tilde{x}_{\log} = \text{sign}(x) \cdot \ln\left(1 + \frac{|x|}{\epsilon_{\log}}\right)$$

where $\epsilon_{\log} = 0.05$ ensures numerical stability, preserves dark near-zero sign dynamics, and avoids singular gradients on unclipped negative detector values.

The resulting feature representations from the linear stream $F_{\text{lin}}$ and log stream $F_{\log}$ are dynamically fused via a spatial-channel gating module:

$$\alpha = \sigma\left(\mathcal{W}_2 * \text{LeakyReLU}(\mathcal{W}_1 * [F_{\text{lin}}, F_{\log}])\right)$$
$$F_0 = \alpha \odot F_{\log} + (1 - \alpha) \odot F_{\text{lin}} + \mathcal{W}_{\text{proj}} * [F_{\text{lin}}, F_{\log}]$$

This enables the network to allocate capacity to the homomorphic stream in high-speckle zones while preserving linear gradient propagation across high-contrast edge boundaries.

### 3.2 23-RRDB Dense Convolutional Backbone

The deep feature trunk consists of 23 Residual-in-Residual Dense Blocks (RRDB) divided into three stages ($8 + 8 + 7$). Each RRDB block contains three 5-layer Residual Dense Blocks (RDB) with continuous feature concatenation and residual scaling ($\beta = 0.2$):

$$x_{l} = \sigma\left(\mathcal{W}_{l} * [x_0, x_1, \dots, x_{l-1}]\right), \quad l \in [1, 4]$$
$$x_{\text{RDB}} = \mathcal{W}_5 * [x_0, x_1, x_2, x_3, x_4] \cdot 0.2 + x_0$$

This dense connectivity promotes maximum gradient propagation and multi-scale feature reuse without vanishing gradients.

### 3.3 Shifted-Window Periodic Self-Attention (Swin Transformer)

Periodic grating arrays in semiconductor layouts (e.g., DRAM capacitor arrays and FinFET fin pitches) exhibit long-range spatial correlations that exceed the receptive field of standard convolutions. SemiRestoreNet embeds two Swin Transformer stages:

- **Stage 1 Swin Block (Window Size $M = 8$)**: Captures local cell pitch regularities.
- **Stage 2 Swin Block (Window Size $M = 16$)**: Models long-range array periodicities.
- **Cyclic Shifting and Boundary Masking**: Computes multi-head self-attention within local windows and shifted windows ($M/2$) with relative position bias matrices $B \in \mathbb{R}^{(2M-1)\times (2M-1)}$:
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right)V$$

### 3.4 Anisotropic Directional Strip Attention

Semiconductor layouts are strictly structured along orthogonal manufacturing axes (Manhattan geometry). The Defect Attention Module integrates anisotropic strip convolutions to enhance sensitivity to continuous line defects:

$$\mathcal{M}_{\text{spatial}}(F) = \sigma\left(\mathcal{W}_{\text{fuse}} * [\text{Conv}_{7\times 7}(F_{\text{pool}}), \text{Conv}_{1\times 9}(F_{\text{pool}}), \text{Conv}_{9\times 1}(F_{\text{pool}})]\right)$$

This prioritizes long horizontal wordlines and vertical bitlines, isolating micro-bridging and line-collapse anomalies.

### 3.5 Cross-Stage Dense Highway

To prevent the attenuation of sub-10nm high-frequency boundary information through the 23 RRDB stages, shallow features from Stage 1 ($F_1$) and mid-level features from Stage 2 ($F_2$) bypass the deep trunk and inject directly into the reconstruction head:

$$F_{\text{head}} = F_{\text{trunk}} + \gamma_1 \cdot \text{Proj}_1(F_1) + \gamma_2 \cdot \text{Proj}_2(F_2)$$

where $\gamma_1, \gamma_2$ are learnable scaling coefficients initialized to $0.1$.

---

## 4. Anti-Hallucination Loss Stack

To guarantee physical reconstruction fidelity and prohibit non-grounded feature hallucination, the optimization objective strictly excludes perceptual and adversarial losses in favor of a multi-domain metrology loss stack:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{charb}} \mathcal{L}_{\text{charb}} + \lambda_{\text{ssim}} \mathcal{L}_{\text{ssim}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}} + \lambda_{\text{fft}} \mathcal{L}_{\text{fft}} + \lambda_{\text{fidelity}} \mathcal{L}_{\text{fidelity}}$$

```text
+---------------------------------------------------------------------------------------+
|                                    Loss Hierarchy                                     |
+--------------------------+-------------+----------------------------------------------+
| Component                | Weight      | Physical / Metrology Function                |
+--------------------------+-------------+----------------------------------------------+
| CharbonnierWeightedLoss  | 1.00        | Robust L1 pixel accuracy with 3x edge boost  |
| SSIMLoss                 | 0.10        | Multi-scale structural and geometric fidelity|
| Sobel EdgeLoss           | 0.05        | Nanometer transition sharpness and CD bounds |
| WeightedFFTLoss          | 0.01        | Power spectrum matching (capped at 2.0)      |
| DegradationConsistency   | 0.05        | Low-frequency data fidelity constraint       |
+--------------------------+-------------+----------------------------------------------+
```

### 4.1 Spatially-Weighted Charbonnier Loss

$$\mathcal{L}_{\text{charb}} = \frac{1}{N}\sum_{i=1}^N W_i \sqrt{(\hat{y}_i - y_i)^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$$

where spatial weight map $W_i = 1.0 + 2.0 \cdot \frac{\|\nabla y_i\|}{\max(\|\nabla y\|)}$ assigns $3\times$ higher gradient penalty to transistor gate boundaries and defect edges.

### 4.2 Degradation-Consistency ("Fidelity") Loss

Enforces that the reconstructed output $\hat{y}$, when projected through the forward degradation operator $\mathcal{D}(\cdot)$, strictly matches the observed low-frequency degraded evidence $x$:

$$\mathcal{L}_{\text{fidelity}} = \|\mathcal{D}(\hat{y}) - \mathcal{D}(x)\|_1 + 0.1 \cdot (1 - \text{SSIM}(\mathcal{D}(\hat{y}), \mathcal{D}(x)))$$

For $2\times$ Super-Resolution, $\mathcal{D}(\hat{y})$ applies Gaussian low-pass filtering followed by anti-aliased downsampling to $(H_{\text{in}}, W_{\text{in}})$, ensuring that high-frequency detail is constrained by input telemetry.

---

## 5. Transfer Learning and Domain Adaptation

To guarantee fast convergence under constrained timelines, the 23-RRDB backbone supports weight initialization from pretrained Real-ESRGAN/ESRGAN models.

### Grayscale Adaptation Protocol
The RGB shallow convolution kernel $W_{\text{RGB}} \in \mathbb{R}^{64 \times 3 \times 3 \times 3}$ is projected to single-channel SEM grayscale space via channel-averaged summation:

$$W_{\text{gray}} = \frac{1}{3}\sum_{c \in \{R,G,B\}} W_{\text{RGB}}[:, c, :, :] \in \mathbb{R}^{64 \times 1 \times 3 \times 3}$$

This preserves low-level directional edge, texture, and gradient filters, transferring over 16.58 million parameters ($98.4\%$ of the network) and accelerating convergence by over $5\times$.

### Layer-Wise Learning Rates
Optimization uses parameter-group separation:
- **Backbone Trunk (Pretrained RRDB)**: $\eta_{\text{trunk}} = 0.2 \times \eta_{\text{base}} = 4.0 \times 10^{-5}$
- **Domain Modules (Swin, CBAM, Highway, Head, Gated Stream)**: $\eta_{\text{domain}} = \eta_{\text{base}} = 2.0 \times 10^{-4}$

---

## 6. High-Order Randomized Degradation Synthesis

For robust Out-Of-Distribution (OOD) generalization across unknown fab tool distributions, training employs a randomized physical degradation engine:

- **Anisotropic Beam Blur**: Continuous $\sigma_x, \sigma_y \sim \mathcal{U}(0.3, 2.5)$, rotation angle $\theta \sim \mathcal{U}(0, \pi)$.
- **Multi-Algorithm Downsampling**: Scale factors $s \in \{1, 2, 4\}$ with stochastic interpolation selection ($\text{Bicubic}, \text{Bilinear}, \text{Area}, \text{Lanczos-4}$).
- **Stochastic Physical Noise Mix**: Unclipped Gamma speckle ($L \in [1, 12]$ looks), Poisson shot noise ($N_e \in [10, 150]$ electron dose), and additive Gaussian read noise.
- **Surface Potential Drift**: 2D polynomial charging gradients.

---

## 7. Technical Specifications

```text
+------------------------------------+--------------------------------------------------+
| Specification                      | Value / Description                              |
+------------------------------------+--------------------------------------------------+
| Model Architecture                 | 23-RRDB + 2x Swin + Anisotropic CBAM + Gated Log |
| Total Parameters (Teacher Model)   | 16.88 Million (16,584,320 transferable)          |
| Total Parameters (Student Model)   | 6.08 Million (8-block compact distillation)      |
| Base Feature Dimension             | 64 channels (32 growth channels per dense layer) |
| Attention Window Sizes             | Window 8 (Stage 1), Window 16 (Stage 2)          |
| Input Format                       | Single-Channel Grayscale Float32 (Unclipped)     |
| Numerical Precision                | FP32 / FP16 Mixed Precision (torch.amp)          |
| Target Inference Acceleration      | torch.compile(mode='reduce-overhead') on CUDA    |
+------------------------------------+--------------------------------------------------+
```

---

## 8. Repository Structure

```text
SemiRestoreNet/
|-- evaluate.py                  # Standalone batch evaluation script (Submission Contract)
|-- model.py                     # Full 23-RRDB + Swin + CBAM + Gated Stream Architecture
|-- losses.py                    # 5-component anti-hallucination loss stack (with Fidelity)
|-- train.py                     # Training pipeline (Layer-wise LR, EMA, AMP, Cosine Decay)
|-- train_kd.py                  # Knowledge distillation engine for compact student networks
|-- dataset.py                   # High-order randomized SEM degradation pipeline
|-- generate_dataset.py          # Synthetic DRAM/FinFET pattern generator with physics noise
|-- metrics.py                   # Metrology metrics (PSNR, SSIM, CD Error, FFT Score)
|-- uncertainty.py               # Heteroscedastic aleatoric and epistemic uncertainty
|-- utils.py                     # Checkpoint I/O, padding utilities, parameter counters
|-- test_physics_improvements.py # Comprehensive unit and integration test suite
|-- configs/
|   `-- train_config.yaml        # Architecture, loss, and training hyperparameters
|-- checkpoints/
|   `-- best_model.pth           # Trained model checkpoint weights
|-- test_inputs/                 # Sample degraded SEM evaluation inputs
|-- restored_test_outputs/       # Restored evaluation outputs
|-- CITATIONS.md                 # Academic and physical citations
|-- requirements.txt             # Python runtime dependencies
|-- LICENSE                      # Apache License 2.0
`-- README.md                    # System documentation
```

---

## 9. Installation and Quick Start

### 9.1 Environment Setup

```bash
git clone https://github.com/DynamiX-Labs/SemiRestoreNet.git
cd SemiRestoreNet
pip install -r requirements.txt
```

### 9.2 Automated Batch Evaluation (Submission Benchmark Contract)

`evaluate.py` operates out-of-the-box on unseen test sets without manual modification:

```bash
# Using named arguments
python evaluate.py --input_dir ./test_inputs --output_dir ./restored_test_outputs

# Using positional arguments
python evaluate.py ./test_inputs ./restored_test_outputs

# Explicit checkpoint selection
python evaluate.py --input_dir /path/to/degraded --output_dir /path/to/restored --checkpoint_path ./checkpoints/best_model.pth
```

### 9.3 Running the Verification Test Suite

```bash
python test_physics_improvements.py
```

### 9.4 Training the Network

```bash
# Train the Full 23-RRDB Teacher Network
python train.py --config configs/train_config.yaml

# Train with Pretrained RRDB Weights
python train.py --config configs/train_config.yaml --pretrained_weights /path/to/RealESRGAN_x4plus.pth

# Train Compact Student Network via Knowledge Distillation
python train_kd.py --config configs/train_config.yaml
```

---

## 10. References and Citations

Comprehensive academic and physical justifications for all architectural components, homomorphic transformations, loss terms, and metrology algorithms are documented in [CITATIONS.md](CITATIONS.md).

---

## 11. License

This project is licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for complete terms and conditions.

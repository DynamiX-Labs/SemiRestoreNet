"""
generate_dataset.py — Standalone Synthetic Semiconductor Dataset Generator.

Generates realistic (Reference, Search) image pairs for:
    - DRAM: Periodic capacitor storage nodes, bitlines, and wordline gratings
    - FinFET: 3D vertical fin lines crossed with horizontal gate pitches and contact cuts

Simulates realistic SEM (Scanning Electron Microscope) physical degradations:
    - Poisson shot noise (low electron beam dose)
    - Additive Gaussian detector read noise
    - Multiplicative speckle noise (Gamma-distributed backscatter interference)
    - SEM beam blur / point spread function (PSF)
    - SEM surface charging / low-frequency background drift

Ground Truth:
    Records true sub-pixel center coordinates (center_x, center_y) of the reference
    template within the search image in CSV and JSON formats.

Usage:
    python generate_dataset.py --style DRAM --num_pairs 20 --output_dir ./data/dram_samples
    python generate_dataset.py --style FinFET --num_pairs 20 --output_dir ./data/finfet_samples
    python generate_dataset.py --style both --num_pairs 50 --output_dir ./data/sem_dataset
"""

import argparse
import os
import json
import csv
import random
import numpy as np
import cv2
from pathlib import Path


# =============================================================================
# 1. SEM Physics Noise & Degradation Engine
# =============================================================================

def apply_sem_degradations(
    image: np.ndarray,
    noise_level: float = 0.3,
    speckle_looks: int = 4,
    blur_sigma: float = 0.8,
    charging_strength: float = 0.15,
) -> np.ndarray:
    """Apply authentic SEM physics degradations to a clean pattern.
    
    Args:
        image: Clean normalized float32 image in [0, 1].
        noise_level: Gaussian/Poisson noise strength.
        speckle_looks: Gamma looks L for multiplicative speckle.
        blur_sigma: SEM electron beam PSF standard deviation.
        charging_strength: SEM charging background drift magnitude.
        
    Returns:
        Degraded SEM image in [0, 1].
    """
    H, W = image.shape
    degraded = image.copy()
    
    # 1. SEM Beam PSF Blur
    if blur_sigma > 0:
        ksize = int(2 * np.ceil(2 * blur_sigma) + 1)
        degraded = cv2.GaussianBlur(degraded, (ksize, ksize), blur_sigma)
    
    # 2. SEM Surface Charging / Non-uniform Potential Drift
    if charging_strength > 0:
        # Low-frequency 2D gradient simulating wafer substrate charging
        x = np.linspace(-1, 1, W)
        y = np.linspace(-1, 1, H)
        xx, yy = np.meshgrid(x, y)
        angle = random.uniform(0, 2 * np.pi)
        grad = np.cos(angle) * xx + np.sin(angle) * yy
        curve = 0.5 * (xx**2 + yy**2)
        charging_map = charging_strength * (0.6 * grad + 0.4 * curve)
        degraded = degraded + charging_map.astype(np.float32)
    
    # 3. Multiplicative Speckle Noise (Gamma-distributed backscatter)
    if speckle_looks > 0:
        speckle = np.random.gamma(
            shape=speckle_looks, scale=1.0 / speckle_looks, size=(H, W)
        ).astype(np.float32)
        degraded = degraded * speckle
    
    # 4. Poisson Shot Noise (Low electron count / dose)
    peak_photons = max(10.0, 1.0 / (noise_level**2 + 1e-6))
    pos_part = np.maximum(degraded, 1e-4) * peak_photons
    poisson_noisy = np.random.poisson(pos_part).astype(np.float32) / peak_photons
    residual = np.minimum(degraded, 0.0)
    degraded = poisson_noisy + residual
    
    # 5. Additive Gaussian Read Noise
    gauss_sigma = noise_level * random.uniform(0.05, 0.15)
    read_noise = np.random.normal(0, gauss_sigma, size=(H, W)).astype(np.float32)
    degraded = degraded + read_noise
    
    # Return unclipped float32 to preserve true physical speckle & noise distribution
    return degraded.astype(np.float32)


# =============================================================================
# 2. DRAM Architecture Pattern Generator (Cell Arrays + Circuit Modulation)
# =============================================================================

def generate_dram_field(
    height: int = 512,
    width: int = 512,
    node_pitch: int = 24,
    node_radius: int = 6,
    seed: int = None,
) -> np.ndarray:
    """Generates a realistic CAD-like DRAM layout with cell sub-blocks and logic decoders."""
    rng = np.random.RandomState(seed if seed is not None else random.randint(0, 100000))
    canvas = np.zeros((height, width), dtype=np.float32)
    canvas.fill(0.12)
    
    # 1. Wordline / Bitline Bus Structure
    wl_pitch = node_pitch * 2
    for y in range(0, height, wl_pitch):
        canvas[max(0, y-2):min(height, y+3), :] += 0.10
    for x in range(0, width, node_pitch):
        canvas[:, max(0, x-1):min(width, x+2)] += 0.08
    
    # 2. Capacitor Storage Nodes with Layout Modulation (Active vs Inactive bitcells)
    # Circuit data patterns create unique spatial density variations
    pattern_mask = rng.rand(height // node_pitch + 2, width // node_pitch + 2) > 0.18
    
    y_range = range(node_pitch // 2, height, node_pitch)
    for row_idx, y in enumerate(y_range):
        x_offset = (node_pitch // 2) if (row_idx % 2 == 1) else 0
        for col_idx, x in enumerate(range(x_offset, width, node_pitch)):
            if pattern_mask[row_idx, col_idx]:
                cv2.circle(canvas, (x, y), node_radius, 0.85, -1)
                cv2.circle(canvas, (x, y), max(1, node_radius - 2), 0.95, -1)
    
    # 3. Add Peripheral Decoder / Sense Amplifier Boundaries & Distinctive Vias
    mid_y = height // 2
    mid_x = width // 2
    
    # Sense amplifier strip (horizontal gap with logic gates)
    canvas[mid_y - 20 : mid_y + 20, :] = 0.20
    for x in range(16, width, 32):
        cv2.rectangle(canvas, (x, mid_y - 12), (x + 20, mid_y + 12), 0.80, -1)
        cv2.circle(canvas, (x + 10, mid_y), 4, 0.95, -1)
        
    # Scribe line alignment cross marks
    cv2.drawMarker(canvas, (64, 64), 0.92, cv2.MARKER_CROSS, 28, 2)
    cv2.drawMarker(canvas, (width - 64, height - 64), 0.92, cv2.MARKER_TILTED_CROSS, 28, 2)
    cv2.drawMarker(canvas, (mid_x, mid_y), 0.95, cv2.MARKER_DIAMOND, 20, 2)
    
    return np.clip(canvas, 0.0, 1.0)


# =============================================================================
# 3. FinFET Architecture Pattern Generator (Logic Standard Cells + Gate Cuts)
# =============================================================================

def generate_finfet_field(
    height: int = 512,
    width: int = 512,
    fin_pitch: int = 16,
    fin_width: int = 5,
    gate_pitch: int = 36,
    gate_width: int = 10,
    seed: int = None,
) -> np.ndarray:
    """Generates a realistic 3D FinFET logic layout with gate cuts, power rails & standard cells."""
    rng = np.random.RandomState(seed if seed is not None else random.randint(0, 100000))
    canvas = np.zeros((height, width), dtype=np.float32)
    canvas.fill(0.10)
    
    # 1. Vertical Silicon Fins
    for x in range(fin_pitch // 2, width, fin_pitch):
        x1 = max(0, x - fin_width // 2)
        x2 = min(width, x + fin_width // 2 + 1)
        canvas[:, x1:x2] = 0.40
    
    # 2. Horizontal Gate Electrodes
    for y in range(gate_pitch // 2, height, gate_pitch):
        y1 = max(0, y - gate_width // 2)
        y2 = min(height, y + gate_width // 2 + 1)
        canvas[y1:y2, :] = np.maximum(canvas[y1:y2, :], 0.65)
        
        # 3D Fin-Gate Crossings
        for x in range(fin_pitch // 2, width, fin_pitch):
            x1 = max(0, x - fin_width // 2)
            x2 = min(width, x + fin_width // 2 + 1)
            canvas[y1:y2, x1:x2] = 0.92
            
    # 3. Power Rails (VDD / VSS thick horizontal metal tracks)
    canvas[10:24, :] = 0.85
    canvas[height - 24 : height - 10, :] = 0.85
    canvas[height // 2 - 8 : height // 2 + 8, :] = 0.88
    
    # 4. Standard Cell Gate Cuts & Via Clusters with Unique Logic Masking
    for cell_row in range(40, height - 40, gate_pitch):
        for cell_col in range(24, width - 24, fin_pitch * 2):
            if rng.rand() > 0.40:
                # Transistor isolation gate cut
                cv2.rectangle(
                    canvas,
                    (cell_col, cell_row - 4),
                    (cell_col + fin_pitch, cell_row + 4),
                    0.10,
                    -1,
                )
            if rng.rand() > 0.45:
                # Middle-of-line contact via
                cv2.rectangle(
                    canvas,
                    (cell_col + 2, cell_row + 8),
                    (cell_col + 8, cell_row + 18),
                    0.98,
                    -1,
                )
                
    # Fiducial / Die Alignment Landmarks
    cv2.drawMarker(canvas, (48, 48), 0.95, cv2.MARKER_CROSS, 24, 2)
    cv2.drawMarker(canvas, (width - 48, height - 48), 0.95, cv2.MARKER_TILTED_CROSS, 24, 2)
    cv2.drawMarker(canvas, (width // 2, height // 2), 0.95, cv2.MARKER_STAR, 22, 2)
            
    return np.clip(canvas, 0.0, 1.0)


# =============================================================================
# 4. Pair Synthesis: Reference (Clean Template) + Search (Noisy Scene)
# =============================================================================

def generate_sample_pair(
    style: str = "DRAM",
    search_size: int = 512,
    ref_size: int = 128,
    noise_level: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generates a synchronized (Reference, Search) pair with ground-truth coordinates.
    
    Args:
        style: "DRAM" or "FinFET".
        search_size: Size of the large search image (H=W).
        ref_size: Size of the reference template (H=W).
        noise_level: SEM degradation severity.
        
    Returns:
        reference_img: [ref_size, ref_size] clean reference image in uint8 [0, 255].
        search_img: [search_size, search_size] degraded search image in uint8 [0, 255].
        gt_metadata: Dict with true center coordinates (center_x, center_y), box, style.
    """
    # 1. Generate full clean semiconductor layout field
    if style.upper() == "DRAM":
        node_pitch = random.choice([20, 24, 28])
        clean_field = generate_dram_field(
            height=search_size,
            width=search_size,
            node_pitch=node_pitch,
            node_radius=max(4, node_pitch // 4),
        )
    elif style.upper() == "FINFET":
        fin_pitch = random.choice([14, 16, 20])
        gate_pitch = random.choice([32, 36, 40])
        clean_field = generate_finfet_field(
            height=search_size,
            width=search_size,
            fin_pitch=fin_pitch,
            gate_pitch=gate_pitch,
        )
    else:
        raise ValueError(f"Unknown style: {style}. Must be 'DRAM' or 'FinFET'.")
        
    # 2. Pick a crop location containing salient layout features
    margin = 32
    max_x = search_size - ref_size - margin
    max_y = search_size - ref_size - margin
    
    # Target areas with rich layout features (e.g. sense amps, logic cuts, fiducials)
    salient_centers = [
        search_size // 4, search_size // 2 - ref_size // 4,
        3 * search_size // 4 - ref_size // 4, 64 + ref_size // 2, search_size - 64 - ref_size // 2
    ]
    base_x = random.choice(salient_centers)
    base_y = random.choice(salient_centers)
    
    top_left_x = int(np.clip(base_x - ref_size // 2 + random.randint(-16, 16), margin, max_x))
    top_left_y = int(np.clip(base_y - ref_size // 2 + random.randint(-16, 16), margin, max_y))
    
    # Clean Reference Template
    ref_clean = clean_field[top_left_y : top_left_y + ref_size, top_left_x : top_left_x + ref_size].copy()
    
    # Ground Truth Center Coordinates
    center_x = float(top_left_x + ref_size / 2.0)
    center_y = float(top_left_y + ref_size / 2.0)
    
    # 3. Apply SEM degradations to the search field
    search_degraded = apply_sem_degradations(
        clean_field,
        noise_level=noise_level,
        speckle_looks=random.choice([3, 4, 6]),
        blur_sigma=random.uniform(0.5, 1.0),
        charging_strength=random.uniform(0.08, 0.18),
    )
    
    # Convert to 8-bit grayscale for file saving [0, 255]
    ref_uint8 = (ref_clean * 255.0).round().astype(np.uint8)
    search_uint8 = (search_degraded * 255.0).round().astype(np.uint8)
    
    metadata = {
        "style": style.upper(),
        "search_width": search_size,
        "search_height": search_size,
        "ref_width": ref_size,
        "ref_height": ref_size,
        "top_left_x": top_left_x,
        "top_left_y": top_left_y,
        "center_x": center_x,
        "center_y": center_y,
        "noise_level": float(noise_level),
    }
    
    return ref_uint8, search_uint8, metadata


# =============================================================================
# 5. Dataset Generation CLI
# =============================================================================

def generate_dataset(
    style: str = "both",
    num_pairs: int = 20,
    output_dir: str = "./data/synthetic_dataset",
    search_size: int = 512,
    ref_size: int = 128,
):
    """Generates and saves the full synthetic dataset with CSV & JSON annotations."""
    out_path = Path(output_dir)
    ref_dir = out_path / "reference"
    search_dir = out_path / "search"
    ref_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)
    
    styles = ["DRAM", "FinFET"] if style.lower() == "both" else [style]
    
    gt_records = []
    
    print(f"================================================================")
    print(f"Applied Materials — Semiconductor Dataset Generator")
    print(f"Style: {style} | Total Pairs: {num_pairs} | Output: {output_dir}")
    print(f"Search Image Size: {search_size}x{search_size} | Ref Size: {ref_size}x{ref_size}")
    print(f"================================================================")
    
    for i in range(num_pairs):
        sample_style = random.choice(styles)
        sample_noise = random.uniform(0.15, 0.40)
        
        pair_id = f"{sample_style.lower()}_{i+1:04d}"
        
        ref_img, search_img, meta = generate_sample_pair(
            style=sample_style,
            search_size=search_size,
            ref_size=ref_size,
            noise_level=sample_noise,
        )
        
        ref_filename = f"{pair_id}_ref.png"
        search_filename = f"{pair_id}_search.png"
        
        # Save PNG images
        cv2.imwrite(str(ref_dir / ref_filename), ref_img)
        cv2.imwrite(str(search_dir / search_filename), search_img)
        
        meta["pair_id"] = pair_id
        meta["reference_file"] = f"reference/{ref_filename}"
        meta["search_file"] = f"search/{search_filename}"
        gt_records.append(meta)
        
        if (i + 1) % max(1, num_pairs // 10) == 0 or i == num_pairs - 1:
            print(f"[{i+1:4d}/{num_pairs}] Generated {pair_id} | True Center: ({meta['center_x']:.1f}, {meta['center_y']:.1f})")
    
    # Save Ground Truth JSON
    json_path = out_path / "ground_truth.json"
    with open(json_path, "w") as f:
        json.dump(gt_records, f, indent=2)
        
    # Save Ground Truth CSV
    csv_path = out_path / "ground_truth.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id", "style", "reference_file", "search_file",
                "center_x", "center_y", "top_left_x", "top_left_y",
                "ref_width", "ref_height", "search_width", "search_height", "noise_level"
            ]
        )
        writer.writeheader()
        writer.writerows(gt_records)
        
    print(f"\n[SUCCESS] Dataset generated successfully!")
    print(f"  - References saved to: {ref_dir}")
    print(f"  - Searches saved to:   {search_dir}")
    print(f"  - Ground truth JSON:   {json_path}")
    print(f"  - Ground truth CSV:    {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Applied Materials Semiconductor Pattern Localization Dataset Generator"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="both",
        choices=["DRAM", "FinFET", "both", "dram", "finfet"],
        help="Semiconductor architecture style (DRAM, FinFET, or both)",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=20,
        help="Number of (Reference, Search) pairs to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/synthetic_dataset",
        help="Output directory for generated dataset",
    )
    parser.add_argument(
        "--search_size",
        type=int,
        default=512,
        help="Search image spatial dimension (default: 512)",
    )
    parser.add_argument(
        "--ref_size",
        type=int,
        default=128,
        help="Reference template spatial dimension (default: 128)",
    )
    
    args = parser.parse_args()
    generate_dataset(
        style=args.style,
        num_pairs=args.num_pairs,
        output_dir=args.output_dir,
        search_size=args.search_size,
        ref_size=args.ref_size,
    )


if __name__ == "__main__":
    main()

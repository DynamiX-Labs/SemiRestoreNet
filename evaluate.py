"""
evaluate.py — 🔴 SUBMISSION-COMPLIANT Batch Inference Script.

CONTRACT:
    python evaluate.py --input_dir /path/to/degraded --output_dir /path/to/restored
    
    - Loads pretrained weights from --checkpoint_path (default: ./checkpoints/best_model.pth)
    - Processes ALL supported images (.png, .tif, .jpg, .bmp) in input_dir
    - Writes restored images to output_dir with SAME filenames
    - NO ground truth needed
    - NO manual edits required
    - Runs AS-IS on any machine with PyTorch installed

CRITICAL: This script is what the hackathon evaluators run on their H100.
          It must work without ANY modification. Test on a clean venv.
"""

import argparse
import sys
import os
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'best_model.pth')


# =============================================================================
# Utilities (inlined to avoid import issues on evaluator machines)
# =============================================================================

def load_image_grayscale(path: str) -> tuple:
    """Load image as grayscale float32 tensor.
    
    Returns:
        Tuple of (tensor [1, 1, H, W], original_info dict)
    """
    img = Image.open(path)
    original_mode = img.mode
    original_size = img.size  # (W, H)
    
    img_gray = img.convert('L')
    arr = np.array(img_gray, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    return tensor, {'mode': original_mode, 'size': original_size}


def save_image_grayscale(tensor: torch.Tensor, path: str):
    """Save tensor as grayscale image."""
    img = tensor.detach().cpu().squeeze()
    img = torch.clamp(img, 0.0, 1.0)
    arr = (img.numpy() * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(arr, mode='L')
    pil_img.save(path)


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 16) -> tuple:
    """Pad to nearest multiple for backbone compatibility."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    return tensor, (pad_h, pad_w)


def unpad(tensor: torch.Tensor, pad_sizes: tuple) -> torch.Tensor:
    """Remove padding."""
    pad_h, pad_w = pad_sizes
    if pad_h > 0:
        tensor = tensor[:, :, :-pad_h, :]
    if pad_w > 0:
        tensor = tensor[:, :, :, :-pad_w]
    return tensor


def list_images(directory: str) -> list:
    """List supported image files."""
    dirpath = Path(directory)
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(dirpath.glob(f'*{ext}'))
        files.extend(dirpath.glob(f'*{ext.upper()}'))
    return sorted(list(set(files)))


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path: str, device: torch.device):
    """Load the pretrained model from checkpoint.
    
    Tries to import model.py from the same directory.
    Falls back to loading the full model from checkpoint if model.py
    is not available.
    """
    # Add script directory to path for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    try:
        from model import create_teacher_model
        model = create_teacher_model()
    except ImportError:
        print("[WARNING] Could not import model.py, attempting to load full model from checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model' in checkpoint:
            return checkpoint['model'].to(device).eval()
        raise ImportError("Cannot load model: model.py not found and checkpoint doesn't contain full model")
    
    # Load weights
    if not os.path.isfile(checkpoint_path):
        print(f"[WARNING] Checkpoint not found at {checkpoint_path}")
        print("[WARNING] Running with randomly initialized weights (for testing only)")
        return model.to(device).eval()
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Filter state dict for matching parameter shapes (ensures zero-crash compatibility)
    model_dict = model.state_dict()
    matched_state_dict = {}
    skipped_keys = []
    
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                matched_state_dict[k] = v
            else:
                skipped_keys.append(f"{k} (ckpt {tuple(v.shape)} vs model {tuple(model_dict[k].shape)})")
                
    if skipped_keys:
        print(f"[INFO] Initialized {len(skipped_keys)} re-architected layers from scratch:")
        for sk in skipped_keys[:4]:
            print(f"  - {sk}")
            
    model.load_state_dict(matched_state_dict, strict=False)
    model = model.to(device).eval()
    
    # torch.compile() for 20-40% inference speedup on modern GPUs (H100 / A100 / RTX)
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print(f"[INFO] torch.compile() applied for faster GPU inference")
        except Exception as e:
            print(f"[INFO] torch.compile() skipped: {e}")
    
    print(f"[INFO] Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"[INFO] Checkpoint epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"[INFO] Checkpoint metrics: {checkpoint['metrics']}")
    
    return model


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def restore_image(
    model,
    image_tensor: torch.Tensor,
    device: torch.device,
    pad_multiple: int = 16,
) -> torch.Tensor:
    """Run restoration on a single image tensor.
    
    Handles padding, forward pass, and unpadding.
    
    Args:
        model: Loaded model in eval mode.
        image_tensor: Input [1, 1, H, W] in [0, 1].
        device: Computation device.
        pad_multiple: Pad spatial dims to this multiple.
        
    Returns:
        Restored image tensor [1, 1, H, W] in [0, 1].
    """
    image_tensor = image_tensor.to(device)
    
    # Pad to multiple
    padded, pad_sizes = pad_to_multiple(image_tensor, pad_multiple)
    
    # Forward pass
    output = model(padded)
    restored = output['restored'] if isinstance(output, dict) else output
    
    # Unpad
    restored = unpad(restored, pad_sizes)
    
    # Clamp to valid range
    restored = torch.clamp(restored, 0.0, 1.0)
    
    return restored


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='KLA Semiconductor Image Restoration — Automated Evaluation Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
    python evaluate.py --input_dir ./test_inputs --output_dir ./test_outputs
    python evaluate.py ./test_inputs ./test_outputs
    python evaluate.py --test_dir /path/to/test --output_dir /path/to/restored --checkpoint_path ./checkpoints/best_model.pth
        """
    )
    
    # Flags with aliases
    parser.add_argument(
        '--input_dir', '--test_dir', '-i', type=str, default=None,
        dest='input_dir',
        help='Directory containing degraded test input images'
    )
    parser.add_argument(
        '--output_dir', '-o', type=str, default=None,
        dest='output_dir',
        help='Directory to write restored output images'
    )
    # Positional arguments fallback (if evaluator runs: python evaluate.py <input_dir> <output_dir>)
    parser.add_argument(
        'positional_args', nargs='*',
        help='Optional positional arguments: <input_dir> <output_dir>'
    )
    parser.add_argument(
        '--checkpoint_path', '--weights', type=str, default=DEFAULT_CHECKPOINT,
        help=f'Path to model checkpoint (default: {DEFAULT_CHECKPOINT})'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device: "cuda", "cpu", or specific "cuda:0" (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Resolve input_dir and output_dir from flags or positionals
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if (input_dir is None or output_dir is None) and len(args.positional_args) >= 2:
        input_dir = args.positional_args[0]
        output_dir = args.positional_args[1]
    elif input_dir is None and len(args.positional_args) >= 1:
        input_dir = args.positional_args[0]
        
    if input_dir is None or output_dir is None:
        parser.error("Must provide input directory and output directory via flags (--input_dir, --output_dir) or positional arguments.")
        
    # ---- Validate inputs ----
    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ---- Device setup ----
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"[INFO] Device: {device}")
    
    # ---- List input images ----
    image_files = list_images(input_dir)
    
    if len(image_files) == 0:
        print(f"ERROR: No supported images found in {input_dir}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)
    
    print(f"[INFO] Found {len(image_files)} images in {input_dir}")
    
    # ---- Load model ----
    model = load_model(args.checkpoint_path, device)
    
    # ---- Process images ----
    total_time = 0.0
    
    for img_path in tqdm(image_files, desc="Restoring images"):
        # Load
        img_tensor, _ = load_image_grayscale(str(img_path))
        
        # Restore
        start_time = time.time()
        restored = restore_image(model, img_tensor, device)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Save with same filename
        output_path = os.path.join(output_dir, img_path.name)
        save_image_grayscale(restored, output_path)
    
    # ---- Summary ----
    avg_time = total_time / len(image_files)
    print(f"\n[DONE] Restored {len(image_files)} images")
    print(f"[DONE] Output directory: {output_dir}")
    print(f"[DONE] Average inference time: {avg_time:.3f}s per image")
    print(f"[DONE] Total time: {total_time:.1f}s")


if __name__ == '__main__':
    main()

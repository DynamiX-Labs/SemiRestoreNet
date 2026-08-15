"""Quick visual test: Input (degraded) → Model Output (restored) → Ground Truth comparison."""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import create_teacher_model
from dataset import apply_degradation_pipeline

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Check checkpoint for upscale factor
    ckpt_path = 'checkpoints/best_model.pth'
    upscale_factor = 2
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        if isinstance(sd, dict) and 'restoration_head.head.0.weight' in sd:
            if sd['restoration_head.head.0.weight'].shape[0] == 64:
                upscale_factor = 1
    
    # Load model
    model = create_teacher_model(upscale_factor=upscale_factor).to(device)
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict(sd, strict=False)
        print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')}, upscale_factor={upscale_factor})")
    model.eval()
    
    # Pick 4 sample GT images
    gt_dir = Path('data/sample_dataset/reference')
    if not gt_dir.exists():
        gt_dir = Path('data/sample_dataset/search')
    
    gt_files = sorted(gt_dir.glob('*.png'))[:4]
    if not gt_files:
        gt_files = sorted(gt_dir.glob('*.npy'))[:4]
    
    print(f"Testing on {len(gt_files)} images from {gt_dir}")
    
    os.makedirs('visual_test', exist_ok=True)
    import cv2
    
    for i, gt_path in enumerate(gt_files):
        # Load GT
        if gt_path.suffix == '.npy':
            gt = np.load(str(gt_path)).astype(np.float32)
            if gt.ndim == 3:
                gt = gt[:,:,0]
            if gt.max() > 1.0:
                gt = gt / 255.0
        else:
            gt = np.array(Image.open(gt_path).convert('L'), dtype=np.float32) / 255.0
            
        if gt.shape != (256, 256):
            gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        # Apply degradation (moderate)
        np.random.seed(42 + i)
        import random
        random.seed(42 + i)
        degraded, meta = apply_degradation_pipeline(gt, 'pure_speckle')
        
        # If 2x super-resolution, downsample degraded to 128x128 LR
        if upscale_factor == 2:
            degraded_lr = cv2.resize(degraded, (128, 128), interpolation=cv2.INTER_AREA)
        else:
            degraded_lr = degraded
        
        print(f"  Image {i}: {gt_path.name} | GT range: [{gt.min():.3f}, {gt.max():.3f}] | Degraded range: [{degraded_lr.min():.3f}, {degraded_lr.max():.3f}]")
        
        # Run model
        input_tensor = torch.from_numpy(degraded_lr.copy()).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = model(input_tensor)
            restored = output['restored'].cpu().squeeze().numpy()
        restored = np.clip(restored, 0, 1)
        
        # Compute PSNR
        mse = np.mean((restored - gt) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        
        # Save individual images
        gt_img = Image.fromarray((np.clip(gt, 0, 1) * 255).astype(np.uint8), mode='L')
        deg_vis = cv2.resize(degraded_lr, (256, 256), interpolation=cv2.INTER_CUBIC)
        deg_img = Image.fromarray((np.clip(deg_vis, 0, 1) * 255).astype(np.uint8), mode='L')
        res_img = Image.fromarray((restored * 255).astype(np.uint8), mode='L')
        
        # Create side-by-side comparison (Input | Output | GT)
        h, w = gt.shape
        comparison = Image.new('L', (w * 3 + 20, h + 30), color=0)
        comparison.paste(deg_img, (0, 30))
        comparison.paste(res_img, (w + 10, 30))
        comparison.paste(gt_img, (w * 2 + 20, 30))
        
        comparison.save(f'visual_test/comparison_{i:02d}.png')
        gt_img.save(f'visual_test/gt_{i:02d}.png')
        deg_img.save(f'visual_test/degraded_{i:02d}.png')
        res_img.save(f'visual_test/restored_{i:02d}.png')
        
        print(f"    PSNR: {psnr:.2f} dB | Saved to visual_test/comparison_{i:02d}.png")
    
    print(f"\nDone! Check the 'visual_test/' folder for results.")
    print(f"Each comparison image: [Degraded Input] | [Model Output] | [Ground Truth]")

if __name__ == '__main__':
    main()

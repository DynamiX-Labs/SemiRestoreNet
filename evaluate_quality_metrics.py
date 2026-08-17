"""
evaluate_quality_metrics.py - Official Hackathon Quality Metrics Benchmark v2.
Enhancements: Overlapping tile stitching, per-degradation breakdown, 8-Fold TTA, Model-Soup.
"""
import os, glob, argparse, random, torch, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from model import create_teacher_model
from utils import get_device
from metrics import compute_psnr, compute_ssim, compute_lpips, compute_cd_error
from dataset import apply_degradation_pipeline

def tile_stitch_forward(model, inp_t, tile_size=128, overlap=32, upscale=2):
    B, C, H, W = inp_t.shape
    stride = tile_size - overlap
    out_H, out_W = H * upscale, W * upscale
    out_sum = torch.zeros(B, C, out_H, out_W, device=inp_t.device)
    out_weight = torch.zeros(B, C, out_H, out_W, device=inp_t.device)
    w1d = torch.hann_window(tile_size * upscale, periodic=False).to(inp_t.device)
    win = w1d.unsqueeze(0) * w1d.unsqueeze(1)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H); x_end = min(x + tile_size, W)
            y_start = max(0, y_end - tile_size); x_start = max(0, x_end - tile_size)
            tile = inp_t[:, :, y_start:y_end, x_start:x_end]
            with torch.no_grad():
                pred_tile = model(tile)['restored']
            oy, ox = y_start * upscale, x_start * upscale
            oy_end, ox_end = y_end * upscale, x_end * upscale
            th, tw = pred_tile.shape[2], pred_tile.shape[3]
            w = win[:th, :tw]
            out_sum[:, :, oy:oy_end, ox:ox_end] += pred_tile * w
            out_weight[:, :, oy:oy_end, ox:ox_end] += w
    return out_sum / (out_weight + 1e-8)

def tta_forward(model, inp_t, use_tile_stitch=True, tile_size=128, upscale=2):
    preds = []
    for k in [0, 1, 2, 3]:
        for flip in [False, True]:
            x = torch.rot90(inp_t, k, dims=[-2, -1])
            if flip: x = torch.flip(x, dims=[-1])
            if use_tile_stitch and (inp_t.shape[-1] > tile_size or inp_t.shape[-2] > tile_size):
                out = tile_stitch_forward(model, x, tile_size=tile_size, upscale=upscale)
            else:
                with torch.no_grad(): out = model(x)['restored']
            if flip: out = torch.flip(out, dims=[-1])
            out = torch.rot90(out, -k, dims=[-2, -1])
            preds.append(out)
    return torch.stack(preds, dim=0).mean(dim=0)

def run_quality_metrics_benchmark(num_samples=50, checkpoint_path='checkpoints/best_finetuned_model.pth', save_plot_path='evaluation_results/hackathon_quality_metrics.png', use_tile_stitch=True):
    device = get_device()
    print(f"[INFO] Device: {device} | Checkpoint: {checkpoint_path} | Tile stitch: {use_tile_stitch}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    upscale_factor = 2
    model = create_teacher_model(upscale_factor=upscale_factor).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    ema_path = str(checkpoint_path).replace('best_finetuned_model.pth', 'best_ema_model.pth')
    if Path(ema_path).exists():
        ema_sd = torch.load(ema_path, map_location=device, weights_only=False)
        ema_sd = ema_sd.get('model_state_dict', ema_sd.get('state_dict', ema_sd))
        msd = model.state_dict()
        soup = {k: 0.5*msd[k]+0.5*ema_sd[k] if k in ema_sd and msd[k].shape==ema_sd[k].shape and msd[k].dtype.is_floating_point else msd[k] for k in msd}
        model.load_state_dict(soup); print("[INFO] Model-Soup: best+EMA averaged")
    deg_types = [('pure_downsample','Pure 2x SR'),('pure_gaussian','Pure Gaussian'),('gaussian_downsample','Gaussian+SR'),('pure_speckle','Pure Speckle'),('speckle_downsample','Speckle+SR')]
    gt_files = sorted(glob.glob('train/train/GT/*.npy')) or sorted(glob.glob('data/sample_dataset/reference/*.png'))
    files = gt_files[1100:1100+num_samples]
    print(f"[INFO] {len(files)} images x {len(deg_types)} types")
    all_psnr, all_ssim, all_lpips, all_cd, per_type = [], [], [], [], {}
    for deg_key, deg_label in deg_types:
        tp, ts = [], []
        print(f"\n  [{deg_label}]")
        for i, fpath in enumerate(files):
            gt = np.load(fpath).astype(np.float32) if fpath.endswith('.npy') else np.array(__import__('PIL').Image.open(fpath).convert('L'), dtype=np.float32)/255.0
            if gt.max() > 1.0: gt = gt/255.0
            np.random.seed(2000+i); random.seed(2000+i)
            deg, _ = apply_degradation_pipeline(gt, deg_key)
            if upscale_factor==2 and deg.shape==gt.shape:
                deg = cv2.resize(deg, (gt.shape[1]//upscale_factor, gt.shape[0]//upscale_factor), interpolation=cv2.INTER_AREA)
            inp_t = torch.from_numpy(deg).unsqueeze(0).unsqueeze(0).to(device)
            rest = torch.clamp(tta_forward(model, inp_t, use_tile_stitch=use_tile_stitch, tile_size=128, upscale=upscale_factor),0,1).cpu().squeeze().numpy()
            p = compute_psnr(rest, gt, data_range=1.0); s = compute_ssim(rest, gt, data_range=1.0)
            tp.append(p); ts.append(s); all_psnr.append(p); all_ssim.append(s)
            if (i+1)%10==0: print(f"    {i+1}/{len(files)} PSNR: {np.mean(tp):.2f} dB")
        per_type[deg_label] = {'psnr': float(np.mean(tp)), 'ssim': float(np.mean(ts))}
        print(f"    -> {deg_label}: {per_type[deg_label]['psnr']:.2f} dB | SSIM {per_type[deg_label]['ssim']:.4f}")
    print("\n  [LPIPS+CD on speckle]")
    for i, fpath in enumerate(files[:20]):
        gt = np.load(fpath).astype(np.float32) if fpath.endswith('.npy') else np.array(__import__('PIL').Image.open(fpath).convert('L'), dtype=np.float32)/255.0
        if gt.max() > 1.0: gt = gt/255.0
        np.random.seed(3000+i); random.seed(3000+i)
        deg, _ = apply_degradation_pipeline(gt, 'pure_speckle')
        if upscale_factor==2 and deg.shape==gt.shape: deg = cv2.resize(deg,(gt.shape[1]//upscale_factor,gt.shape[0]//upscale_factor),interpolation=cv2.INTER_AREA)
        inp_t = torch.from_numpy(deg).unsqueeze(0).unsqueeze(0).to(device)
        rest = torch.clamp(tta_forward(model, inp_t, use_tile_stitch=False, upscale=upscale_factor),0,1).cpu().squeeze().numpy()
        all_lpips.append(compute_lpips(rest, gt, device='cpu'))
        cd = compute_cd_error(rest, gt)
        if np.isfinite(cd): all_cd.append(cd)
    mp = float(np.mean(all_psnr)); ms = float(np.mean(all_ssim)); ml = float(np.mean(all_lpips)) if all_lpips else 0.0; mc = float(np.mean(all_cd)) if all_cd else 0.0
    print("\n"+"="*70)
    print("         OFFICIAL HACKATHON QUALITY METRICS REPORT v2             ")
    print("="*70)
    print(f"  1. pSNR  : {mp:.4f} dB")
    print(f"  2. SSIM  : {ms:.4f}")
    print(f"  3. LPIPS : {ml:.4f}")
    print(f"  *  CD    : {mc:.4f} nm")
    print("="*70)
    print("\n  Per-Degradation Breakdown:")
    for label, res in per_type.items():
        print(f"    {label:30s}: {res['psnr']:.2f} dB | SSIM {res['ssim']:.4f}")
    print("="*70)
    save_dir = os.path.dirname(save_plot_path)
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0D1117')
    fig.suptitle('SemiRestoreNet Hackathon Metrics (8-Fold TTA + Tile Stitching)', fontsize=13, color='white', fontweight='bold')
    clrs = ['#58A6FF','#3FB950','#79C0FF','#D29922','#FFA657']
    axes[0].set_facecolor('#161B22')
    names = list(per_type.keys()); vals = [per_type[n]['psnr'] for n in names]
    bars = axes[0].barh(names, vals, color=clrs, height=0.6)
    axes[0].axvline(34.0, color='#FF7B72', linestyle='--', alpha=0.8, label='Target 34 dB')
    axes[0].axvline(mp, color='white', linestyle=':', alpha=0.7, label=f'Avg {mp:.2f} dB')
    axes[0].set_xlabel('PSNR (dB)', color='white'); axes[0].set_title(f'Per-Type PSNR | Avg: {mp:.2f} dB', color='white', fontsize=11)
    axes[0].tick_params(colors='white'); axes[0].set_xlim(24, 38); axes[0].legend(facecolor='#21262D', labelcolor='white', fontsize=9)
    for bar, val in zip(bars, vals): axes[0].text(val+0.1, bar.get_y()+bar.get_height()/2, f'{val:.2f}', va='center', color='white', fontsize=9)
    axes[1].set_facecolor('#161B22'); axes[1].bar(['SSIM'],[ms],color='#3FB950',width=0.4); axes[1].axhline(0.80,color='#FF7B72',linestyle='--',label='Target>0.80')
    axes[1].set_ylim(0,1.0); axes[1].set_title(f'SSIM: {ms:.4f}',color='white',fontsize=12); axes[1].tick_params(colors='white'); axes[1].legend(facecolor='#21262D',labelcolor='white')
    axes[2].set_facecolor('#161B22'); axes[2].bar(['LPIPS'],[ml],color='#D29922',width=0.4); axes[2].axhline(0.35,color='#FF7B72',linestyle='--',label='Target<0.35')
    axes[2].set_ylim(0,1.0); axes[2].set_title(f'LPIPS: {ml:.4f}',color='white',fontsize=12); axes[2].tick_params(colors='white'); axes[2].legend(facecolor='#21262D',labelcolor='white')
    for ax in axes:
        for spine in ax.spines.values(): spine.set_color('#30363D')
    plt.tight_layout(); plt.savefig(save_plot_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"\n[SUCCESS] Scorecard saved: {save_plot_path}")
    return {'psnr': mp, 'ssim': ms, 'lpips': ml, 'cd': mc, 'per_type': per_type}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_finetuned_model.pth')
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--save_plot', type=str, default='evaluation_results/hackathon_quality_metrics.png')
    parser.add_argument('--no_tile_stitch', action='store_true')
    args = parser.parse_args()
    run_quality_metrics_benchmark(num_samples=args.num_samples, checkpoint_path=args.checkpoint, save_plot_path=args.save_plot, use_tile_stitch=not args.no_tile_stitch)

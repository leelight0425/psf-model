#!/usr/bin/env python
"""
Process REAL (non-simulated) checkerboard photos through the PSF-Estimation
pipeline — from raw photos to training-ready .npz files.

Quick start — single image
--------------------------
python process_real.py --image my_photo.tif --config configs/real.yaml --full

Batch — all images in a folder
------------------------------
python process_real.py --dir D:/photos/checkerboard_shots/ --config configs/real.yaml --full

Steps (single or combined)
--------------------------
  --step crop    only extract edge patches           (Step 1)
  --step sfr     only compute SFR curves             (Step 2)
  --step npz     only package training .npz files    (Step 3)
  --full         run all three steps in sequence

Without --full or --step, only Step 1 (crop) runs.
"""

import os
import sys
import argparse
import glob as globmod
import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import yaml
import checkerboard_rgb  # noqa: E402
import sfrmat5_py         # noqa: E402


# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_paths(cfg):
    base = os.path.join(PROJECT_ROOT, cfg['dataset_dir'], cfg['lens_name'])
    return (
        os.path.join(base, 'crop', cfg['noise']),
        os.path.join(base, 'mat',  cfg['noise']),
        os.path.join(base, 'npy'),
    )


def _collect_images(path):
    """Return a list of absolute image paths from a file or directory."""
    if os.path.isdir(path):
        exts = ('*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.bmp',
                '*.dng', '*.raw', '*.nef', '*.cr2', '*.arw')
        files = []
        for ext in exts:
            files.extend(globmod.glob(os.path.join(path, ext)))
        return sorted(files)
    else:
        return [os.path.abspath(path)]


# ── Bayer / RAW helpers ───────────────────────────────────────────

_BAYER_PATTERNS = {
    'RGGB': cv2.COLOR_BayerRG2RGB,
    'GRBG': cv2.COLOR_BayerGR2RGB,
    'GBRG': cv2.COLOR_BayerGB2RGB,
    'BGGR': cv2.COLOR_BayerBG2RGB,
}


def _load_image(image_path, bayer_pattern=None):
    """Load an image, optionally demosaicing Bayer RAW to RGB.

    Args:
        image_path:   path to image file
        bayer_pattern:'RGGB'|'GRBG'|'GBRG'|'BGGR'|None.
                      None → treat as normal RGB/gray image.

    Returns:
        np.ndarray (H,W,3) uint8 RGB image.
    """
    # ── DNG / rawpy path ──
    if image_path.lower().endswith('.dng'):
        try:
            import rawpy
            with rawpy.imread(image_path) as raw:
                rgb = raw.postprocess(gamma=(1.0, 1.0), output_bps=16,
                                      use_camera_wb=True)
            img = (rgb.astype(np.float32) / 65535.0 * 255).clip(0, 255).astype(np.uint8)
            if bayer_pattern:
                # Already demosaiced by rawpy — use as-is
                return img
            return img
        except ImportError:
            raise ImportError(
                "DNG files require rawpy.  Install with: pip install rawpy")

    # ── Standard image (OpenCV) ──
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if bayer_pattern:
        if len(img.shape) != 2:
            raise ValueError(
                f"Bayer RAW expects single-channel image, got shape {img.shape}. "
                f"Remove --raw if this is already an RGB image.")
        code = _BAYER_PATTERNS.get(bayer_pattern.upper())
        if code is None:
            raise ValueError(
                f"Unknown Bayer pattern '{bayer_pattern}'. "
                f"Choose from: {list(_BAYER_PATTERNS.keys())}")
        img = cv2.cvtColor(img, code)
        print(f"    Bayer RAW → RGB  (pattern={bayer_pattern.upper()}, "
              f"demosaiced size={img.shape[1]}×{img.shape[0]})")

    return img


# ──────────────────────────────────────────────────────────────────
# Step 1 – crop edge patches from a single real photo
# ──────────────────────────────────────────────────────────────────

def step_crop_one(image_path, crop_dir, cfg, bayer_pattern=None):
    img = _load_image(image_path, bayer_pattern=bayer_pattern)

    saved = checkerboard_rgb.checker.crop_real(
        image=img,
        savepath=crop_dir,
        pixel_size=cfg['pixel_size'],
        focal_length=cfg['focal_length'],
        edge_half_width=cfg.get('edge_half_width', 24),
        square_size=cfg.get('square_size', 200),
        hfov_max=cfg.get('hfov_max'),
    )
    return saved


def step_crop(image_paths, crop_dir, cfg, bayer_pattern=None):
    """Process one or more images — clear crop_dir on first image, append afterwards."""
    print("=" * 60)
    print("STEP 1 — Extract edge patches from real checkerboard photos")
    print("=" * 60)
    print(f"  Images : {len(image_paths)} file(s)")
    print(f"  Output : {crop_dir}")
    if bayer_pattern:
        print(f"  Bayer  : {bayer_pattern.upper()}  (demosaiced → RGB)")
    print(f"  Sensor : pixel_size={cfg['pixel_size']:.4f} mm, "
          f"focal_length={cfg['focal_length']:.2f} mm")

    os.makedirs(crop_dir, exist_ok=True)

    total = 0
    for idx, impath in enumerate(image_paths):
        print(f"\n  [{idx + 1}/{len(image_paths)}] {os.path.basename(impath)}")
        try:
            img = _load_image(impath, bayer_pattern=bayer_pattern)
        except Exception as e:
            print(f"    WARNING: {e} — skipped")
            continue
        print(f"    size={img.shape[1]}×{img.shape[0]}, "
              f"channels={1 if len(img.shape)==2 else img.shape[2]}")

        saved = step_crop_one(impath, crop_dir, cfg, bayer_pattern=bayer_pattern)
        total += len(saved)

    print(f"\n  Total edge patches saved: {total}")


# ──────────────────────────────────────────────────────────────────
# Step 2 – SFR / MTF computation
# ──────────────────────────────────────────────────────────────────

def step_sfr(crop_dir, mat_dir):
    print()
    print("=" * 60)
    print("STEP 2 — Compute SFR curves (ISO 12233 slanted-edge)")
    print("=" * 60)
    print(f"  Input  : {crop_dir}")
    print(f"  Output : {mat_dir}")
    sfrmat5_py.process_folder(crop_dir, mat_dir, npol=5, wflag=0)


# ──────────────────────────────────────────────────────────────────
# Step 3 – training data packaging (delegates to generate_fov_weight)
# ──────────────────────────────────────────────────────────────────

def step_npy(lens_name, mat_dir, npy_dir):
    print()
    print("=" * 60)
    print("STEP 3 — Package SFR into .npz training data")
    print("=" * 60)
    print(f"  Input  : {mat_dir}")
    print(f"  Output : {npy_dir}")

    os.makedirs(npy_dir, exist_ok=True)

    from scipy.io import loadmat
    import torch
    import utils.tools as tools

    mat_files = sorted([f for f in os.listdir(mat_dir) if f.endswith('.mat')])
    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found in {mat_dir}. Run step 2 first.")

    # Load all .mat files
    records = []
    for fn in mat_files:
        d = loadmat(os.path.join(mat_dir, fn))
        records.append({
            'sfr':    d['sfr'],
            'fov':    float(d['fov']),
            'rot':    float(d['rot']),
            'offset': d['offset'],
            'fn':     fn,
        })

    # Group by integer FOV bin [i, i+1), sort each bin by rot
    hfov_max = int(max(r['fov'] for r in records)) + 1

    weights_all = []   # list of per-bin weight tensors
    gt_bins    = []    # list of per-bin record lists
    fn_bins    = []    # list of per-bin filename lists

    for i in range(hfov_max):
        subset = [r for r in records if i <= r['fov'] < i + 1]
        if not subset:
            continue
        subset = sorted(subset, key=lambda r: r['rot'])
        gt_bins.append(subset)
        fn_bins.append([r['fn'] for r in subset])

        angles = torch.tensor([r['rot'] for r in subset], dtype=torch.float64)
        w = tools.weights(angles, bins=8)
        weights_all.append(w)

    weights_flat = torch.cat(weights_all)
    weights_flat = weights_flat / torch.sum(weights_flat)

    # Flatten
    gt_flat = [item for sublist in gt_bins for item in sublist]
    fn_flat = [item for sublist in fn_bins for item in sublist]

    for idx, (gt, fn) in enumerate(zip(gt_flat, fn_flat)):
        sfr = gt['sfr']
        width = (sfr.shape[0] + 1) // 2
        np.savez(
            os.path.join(npy_dir, fn.replace('.mat', '.npz')),
            sfr=sfr[:width, 1:4],            # R, G, B
            weight=weights_flat[idx].numpy() if idx < len(weights_flat) else np.array(1.0),
            rot=np.array(gt['rot']),
            fov=np.array(gt['fov']),
            offset=gt['offset'].reshape(-1)[:2],
        )

    print(f"  Total .npz files written: {len(gt_flat)}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Process real checkerboard photos for PSF-Estimation')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--image', help='Path to a single checkerboard photo')
    src.add_argument('--dir',   help='Directory of checkerboard photos (batch)')
    ap.add_argument('--config', default='configs/real.yaml',
                    help='Config YAML for real camera parameters')
    ap.add_argument('--raw', choices=['RGGB', 'GRBG', 'GBRG', 'BGGR'],
                    help='Bayer pattern for RAW images. Demosaics before cropping.')
    ap.add_argument('--full', action='store_true',
                    help='Run all 3 steps: crop → sfr → npz')
    ap.add_argument('--step', choices=['crop', 'sfr', 'npz'],
                    help='Run a single step')
    args = ap.parse_args()

    cfg = load_config(args.config)
    crop_dir, mat_dir, npy_dir = build_paths(cfg)
    bayer = args.raw  # None → RGB; 'RGGB' → demosaic

    # Resolve image list
    if args.image:
        image_paths = [os.path.abspath(args.image)]
    else:
        image_paths = _collect_images(args.dir)
        if not image_paths:
            print(f"ERROR: No images found in {args.dir}")
            return 1
        print(f"Batch mode: {len(image_paths)} image(s) found in {args.dir}\n")

    # ── Dispatch ──
    if args.step == 'crop':
        step_crop(image_paths, crop_dir, cfg, bayer_pattern=bayer)
    elif args.step == 'sfr':
        step_sfr(crop_dir, mat_dir)
    elif args.step == 'npz':
        step_npy(cfg['lens_name'], mat_dir, npy_dir)
    elif args.full:
        step_crop(image_paths, crop_dir, cfg, bayer_pattern=bayer)
        step_sfr(crop_dir, mat_dir)
        step_npy(cfg['lens_name'], mat_dir, npy_dir)
    else:
        # default: crop only
        step_crop(image_paths, crop_dir, cfg, bayer_pattern=bayer)

    print("\nDone.")

    if not args.full and not args.step:
        print("Tip: use --full to run crop → sfr → npz in one pass.")


if __name__ == '__main__':
    main()

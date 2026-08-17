# AGENTS.md

Optical aberration estimation (OAE) for PSF-Estimation (CVPR 2025). An MLP maps normalized field height `H` → Seidel aberration coefficients → wavefront → PSF → MTF, supervised by ISO 12233 slanted-edge SFR/MTF data. Current lens: `63762BB`. See `CLAUDE.md` for the full Chinese-language deep dive.

## Commands

```bash
python main.py                        # train (reads configs/63762BB.yaml, fixed seed=0)
python generate_edges_rgb.py          # step 1: synthetic blurred edges (uses configs/ss.yaml)
python sfrmat5_py.py [crop_dir mat_dir]  # step 2: ISO 12233 SFR, replaces MATLAB user_sfrmat5_rgb.m
python generate_fov_weight.py         # step 3: .mat → .npz training data (uses configs/ss.yaml)
python test_real/real_sfr_pipeline.py [imgs...] --save test_real/out   # real-photo pipeline
```

On Windows, Chinese console output garbles without `PYTHONIOENCODING=utf-8`. No test suite exists; results are verified manually via `test_real/验证报告.md`.

## Data pipeline

```
input/63762BB.xlsx (Zernike coeffs) ──①──> dataset/63762BB/crop/shot0.00/*.tif
   ──②──> dataset/63762BB/mat/shot0.00/*.mat ──③──> dataset/63762BB/npy/*.npz ──④──> result/63762BB/{timestamp}/
```

- Step 1 (`generate_edges_rgb.py`) **clears the crop and mat folders** (`clear=True`) and imports `model.checkerboard_rgb` — NOT the root-level duplicate. Both files exist; the root copy fixed an `exec`→`setattr` bug in `to()`.
- Step 2 (`sfrmat5_py.py`) only processes `.tif` filenames matching `fov(-?\d+\.\d+)_angle(-?\d+\.\d+)` and only saves `.mat` when max SFR ≥ 0.99 with no NaN.
- Step 3 (`generate_fov_weight.py`) reads a **hardcoded** mat dir `.\dataset\63762BB\mat\shot0.00` (line ~62) and writes `.npz` keys `sfr/weight/rot/fov/offset`.

## Config gotchas

- `configs/ss.yaml` (data prep) and `configs/63762BB.yaml` (training) have **different key sets**. `utils/train.py:train` actually reads `lr`, `epochs`, `interval`; `ss.yaml`'s `lr_I/lr_II/epochs_I/epochs_II` are old-format leftovers.
- Step-1/3 scripts require the `noise` key (present in `ss.yaml`: `shot0.00`).
- `configs/63762BB.yaml` can override sensor params via `sensor_res: [高, 宽]` and `s_psf: N` (passed by `main.py` into `IS(s_psf=..., sensor_res=...)`; omit to use xlsx/default).
- Sampling density: `num_psf` (default 21) controls `Hs = linspace(0,1,num_psf)` in `utils/train.py`; also drives psf_grid count. Training knobs: `lr`, `epochs`, `interval`, `seed`, `device` (`cuda`/`cpu`/`cuda:0`). `net` (`ss`=9 Seidel terms / `l`=10) is honored in both `main.py` and the `train()` loop.
- Dead config keys removed: `w_sfr`, `BS`, `loss`, `fov1`/`fov2` are not used by the code (loss is always L1).
- Real-data flow (replaces synthetic): `configs/real.yaml` sets `real: true` + `mat_dir` (output of `test_real/real_sfr_pipeline.py --save`). Its `.mat` files carry `cx/cy/img_h/img_w`; `generate_fov_weight.py` computes each edge's `fov` from pixel position via `fov = atan(r · pixelsize / efl)` — set `efl` (µm) and `pixelsize` (µm) in the config for the actual camera (defaults = xlsx values). The `lens` config key decouples the input xlsx (`input/{lens}.xlsx`) from the output `filename` prefix.
- `test_real/real_sfr_pipeline.py` auto-detects the checkerboard corner count (OpenCV requires a size; it tries ~20 common `(cols,rows)` patterns and keeps the max-corner match). Pass `--pattern-cols/--pattern-rows` to force a specific board. Inputs are normalized to RGB by `load_image_rgb`: camera RAW (`.dng/.arw/.nef/...`) via `rawpy` with **linear output** (`gamma=(1,1)`, 16-bit, no auto-bright — never gamma/tonemap, it changes MTF shape), single-channel `.tif` demosaicked with `--bayer-pattern` (default grayscale; repo assumes RGGB), 16-bit scaled by a linear 1–99 percentile clip.
- `main.py` takes the config as a positional arg: `python main.py configs/real.yaml` (default `configs/63762BB.yaml`). `generate_fov_weight.py [config]` likewise.
- yaml readers open with `encoding='utf-8'`; Chinese comments in configs break otherwise.

## Architecture

- `model/optics_rgb.py` — `IS` lens model: reads Zernike coeffs from `input/*.xlsx`; `s_basis` (Seidel basis: `ss`=9 terms, `l`=10), `Zer2PSF2` (Zernike→PSF ground truth), `Seidel2PSF`, `fov2H`/`H2fov`.
- `model/PSF_mlp.py` — `PSF_mlp` (H → 9 Seidel coeffs → wavefront → PSF via FFT) and `shift_net` (R/B vs G lateral chromatic aberration; outputs scaled ×5).
- `utils/train.py` — two-stage training: (1) per-FOV/per-channel `PSF_mlp` with loss = Σweight·|MTF−SFR|, (2) frozen PSF + `shift_net` fitting `.npz` offsets. Results saved under `result/63762BB/{MMDD-HHMMSS}/`. Also exports one `.npz` per grid cell (`save_psf_grid`) to `.../{ts}/psf_grid/` named `GR_{ts}_00_{W}x{H}_cx={cx}_cy={cy}.npz` with keys `psf/cx/cy` (grid density from `grid_rows`/`grid_cols` config keys, default 9×12).
- `sfrmat5_py.py` — line-by-line Python port of MATLAB sfrmat5; `sfrmat5_rgb(image)` → `(sfr, esf, offset, rot)`, internals `_project2/_findedge2/_tukey2/_centroid`.

## Conventions — do not casually change

- Network outputs **Seidel coefficients, not PSF pixels** (physical prior). RGB channels modeled independently, then `shift_net` corrects lateral color.
- **F=5 coordinate scaling** in `utils/tools.slice` (3D MTF 2D slice) reduces interpolation error — keep.
- ISO 12233 slanted-edge needs edge tilt **5°–45°**; near-orthogonal (<2°) edges → zero-count bins → SFR NaN. Physical limitation, not a bug.
- Bayer pattern assumed **RGGB** in `checker.mosaic`/`demosaic`; other arrays need reworking the flip logic (see `notes.md`).

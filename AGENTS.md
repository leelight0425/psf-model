# AGENTS.md

## Project

- This is a Python/PyTorch PSF-estimation project for lens `63762BB`: `H -> Seidel coefficients -> wavefront -> PSF -> MTF`, supervised by ISO 12233 SFR data.
- `main.py` is the training entrypoint; `model/optics_rgb.py` contains the optical model, `model/PSF_mlp.py` the physical networks, and `utils/train.py` the two-stage training/export loop.
- `CLAUDE.md` has additional repository-specific context; trust executable code over stale README instructions.

## Commands

Run from the repository root:

```bash
python main.py                                      # configs/63762BB.yaml
python generate_edges_rgb.py                       # synthetic edges; configs/ss.yaml
python sfrmat5_py.py crop_dir mat_dir              # ISO 12233 SFR; omit args for defaults
python generate_fov_weight.py configs/ss.yaml      # omit config for this default
python test_real/real_sfr_pipeline.py image.tif --save test_real/out
python test_mosaic_demosaic.py                    # only focused executable check
```

- There is no general automated test suite. Use `test_real/验证报告.md` for the documented manual real-photo verification.
- Set `PYTHONIOENCODING=utf-8` on Windows when Chinese console output must remain readable.

## Data Flow

- Synthetic preparation is intended to be `input/63762BB.xlsx -> edge crops -> SFR .mat -> training .npz -> main.py`; the scripts currently disagree on paths: `generate_edges_rgb.py` writes under `dataset/63762BB/{crop,mat,npy}/shot0.00`, while `generate_fov_weight.py configs/ss.yaml` reads/writes `dataset/63762BB/{mat,npy}` and `configs/63762BB.yaml` trains from `dataset/63762BB_1/npy`. Check or fix these paths before running the pipeline.
- `generate_edges_rgb.py` deletes the configured crop and MAT directory contents before regenerating them; verify paths and backups first.
- Training reads `lr`, `epochs`, `interval`, `npy`, and `net`; the legacy `lr_I`/`lr_II` and `epochs_I`/`epochs_II` fields in `ss.yaml` are ignored by `utils/train.py`.
- `sfrmat5_py.py` only parses filenames containing `fov<number>_angle<number>` or `fov<number>_v`, and saves results only for finite SFR with peak SFR at least `0.99`.

## Real Data

Use this order: measure `.mat` files, convert them to `.npz`, then train.

```bash
python test_real/real_sfr_pipeline.py photo1.tif photo2.tif --save test_real/out
python generate_fov_weight.py configs/real.yaml
python main.py configs/real.yaml
```

- The real pipeline reads `pattern_cols`/`pattern_rows` from `configs/real.yaml`; CLI flags override them. With no image arguments it scans `test_real/downloads/`.
- Valid real `.mat` files need `cx`, `cy`, `img_h`, and `img_w`; real-mode FOV is computed from those positions using `efl` and `pixelsize`.
- Edges below the default 5-degree tilt are rejected because ISO slanted-edge projection can otherwise produce NaN SFR.
- `main.py` real mode requires `wavelengths`, `na`, and `hfov`; `configs/real.yaml` supplies them, along with the real `.mat` and `.npz` paths.
- Do not run synthetic `generate_edges_rgb.py` during the real-photo flow because it clears synthetic crop/MAT outputs.
- `process_real.py` is a separate all-in-one real-photo pipeline, but expects `dataset_dir`, `lens_name`, and `noise`; it cannot use the current `configs/real.yaml` without adapting the config. Use `test_real/real_sfr_pipeline.py` with `configs/real.yaml` for the documented real flow.

## Invariants

- The networks predict physical Seidel coefficients, not PSF pixels; RGB PSFs are separate and `shift_net` models R/B lateral shift relative to G.
- Preserve the `F=5` coordinate scaling in `utils/tools.py:slice`; it reduces MTF interpolation quantization.
- Bayer handling assumes RGGB in the model checkerboard path; changing the sensor pattern requires revisiting mosaic/demosaic flips.
- Results are written under `result/{filename}/`; exports include PSF maps, `compare.png`, and `psf_grid` `.npz` files.

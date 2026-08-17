# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

PSF-Estimation：基于物理先验的成像系统模糊估计框架（CVPR 2025，A Physics-Informed Blur Learning Framework for Imaging Systems）。
目标是估计镜头的全视场 PSF：用神经网络把归一化视场高度 H 映射为 Seidel 像差系数 → 波前 → PSF → MTF，
以 ISO 12233 斜边法实测的 SFR/MTF 曲线作为监督。当前镜头为 `63762BB`。

## 常用命令

```bash
# 训练（入口，读取 configs/63762BB.yaml）
python main.py            # 或 bash run.sh（会打印耗时）

# ① 生成合成模糊边缘 patch（从 input/*.xlsx 的 Zernike 系数生成棋盘格边缘）
python generate_edges_rgb.py          # 用 configs/ss.yaml

# ② 计算 SFR（Python 版 sfrmat5，替代 MATLAB 的 user_sfrmat5_rgb.m）
python sfrmat5_py.py <crop_dir> <mat_save_dir>
python sfrmat5_py.py                  # 默认路径

# ③ 由 .mat 构建训练数据 .npz（含 sfr/weight/rot/fov/offset）
python generate_fov_weight.py         # 用 configs/ss.yaml

# 真实棋盘格照片 → 角点检测 → 边缘裁剪 → sfrmat5 完整测量流程
python test_real/real_sfr_pipeline.py --save test_real/out
```

注意：Windows 下中文输出可能乱码，运行时加 `PYTHONIOENCODING=utf-8`。
项目无测试套件，验证主要靠 `test_real/验证报告.md` 记录的结果。

## 数据管线（4 步）

```
输入 input/*.xlsx（Zernike 系数）
   │
   ▼ ① generate_edges_rgb.py
模糊边缘 patch (.tif) → dataset/{lens}/crop/
   │
   ▼ ② sfrmat5_py.py（ISO 12233 斜边法）
SFR/MTF 曲线 (.mat) → dataset/{lens}/mat/
   │
   ▼ ③ generate_fov_weight.py
训练数据 (.npz: sfr, weight, rot, fov, offset) → dataset/{lens}/npy/
   │
   ▼ ④ main.py → utils/train.py
PSF_mlp 训练；结果存入 result/{lens}/{时间戳}/（psf.npy、psfmap 图、compare.png）
```

每一步的路径由 config 里 `filename` + `dataset` 拼出；`sfrmat5_py.process_folder` 只处理文件名匹配 `fov±数字_angle±数字` 的 .tif。

## 架构与关键模块

- **main.py** — 训练入口：读 config，构建 `IS`（镜头模型）、`PSF_mlp`、`shift_net`，调用 `utils.train.train`。
- **model/optics_rgb.py** — `IS` 类：镜头物理模型。从 xlsx 读 Zernike 系数；提供 `s_basis`（Seidel 基底，`type='ss'` 9 项 / `'l'` 10 项）、`Zer2PSF2`（Zernike→PSF，作为 GT）、`fov2H`/`H2fov`（视场角↔归一化视场高度）、`Seidel2PSF`。
- **model/PSF_mlp.py** — `PSF_mlp`：MLP 输出 9 个 Seidel 系数 → `seidel2wavefront` → `wavefront2psf`（FFT→PSF）。`shift_net`：修正 R/B 相对 G 的横向色差位移（输出×5 缩放）。
- **model/checkerboard_rgb.py** — `checker` 类：`latent` 生成旋转 5° 的棋盘格、`crop` 按角点裁边缘 patch 并做 PSF 卷积；`mosaic`/`demosaic` 做 RGGB Bayer 编解码。
- **utils/train.py** — 训练主循环。两阶段：先逐 FOV/通道用 AdamW 训 PSF_mlp（loss = Σweight·|MTF−SFR|），再训 shift_net 对齐色差。`psf_map` 把 PSF 列表排成 9×12 视场网格图。
- **utils/tools.py** — `weights`（按角度分 bin 的权重）、`slice`（3D MTF 二维切片，坐标×5）、`PSF2MTF`、`downsample`、`fov2H`/`H2fov`。
- **sfrmat5_py.py** — MATLAB sfrmat5 的逐行 Python 移植（ISO 12233 斜边法）。`sfrmat5_rgb(image, npol=5, wflag=0)` 返回 `(sfr, esf_all, offset, rot)`；核心内部函数 `_project2`（投影分箱）、`_findedge2`、`_tukey2`、`_centroid`。
- **test_real/real_sfr_pipeline.py** — 真实棋盘格验证流程：`findChessboardCorners`+`cornerSubPix` 检测角点，KDTree 邻近图构建网格边缘，按倾斜角筛选后裁剪 patch 喂给 sfrmat5，只保存"完整有效"的 .mat（无 NaN、MTF 峰值≥0.99、MTF 降到 0.5 以下）。

## 关键技术约定（勿随意改动）

- **物理先验**：网络输出 Seidel 像差系数而非 PSF 像素，保证物理合理性。
- **三通道独立**：RGB 分别建波前/PSF，最后用 shift_net 校正横向色差。
- **F=5 坐标缩放**：`utils/tools.slice` 提取 MTF 2D 切片时坐标放大 5 倍减小插值误差。
- **ISO 12233 斜边法要求边缘倾斜 5°–45°**：近似正交（<2°）会导致 `_project2` 分箱零计数 → SFR=NaN，这是物理限制而非 bug（参考 `sfrmat5_dist/sfrmat5/project2.m` 的零计数警告）。
- **Bayer 阵列假定 RGGB**：`checker.mosaic`/`demosaic` 的翻转操作基于 RGGB 布局，换 CMOS 阵列需重做翻转（见 `notes.md`）。
- `configs/ss.yaml` 与 `63762BB.yaml` 字段不完全一致：`train.py` 实际读取 `lr`/`epochs`/`interval`（63762BB 提供）；ss.yaml 用的 `lr_I/lr_II` 等是旧格式。
- 根目录和 `model/` 下各有一份 `checkerboard_rgb.py`，根目录版已修复 `to()` 中 exec→setattr 的问题；`generate_edges_rgb.py` 实际 import 的是 `model.checkerboard_rgb`。

## 训练流程要点

- `main.py` 固定 seed=0，读 `configs/63762BB.yaml`（lr=5e-3, epochs=501, BS=24, net=ss）。
- 训练分两阶段（见 `utils/train.py:train`）：
  1. 逐 FOV（每 `interval` 度）+ 逐通道训练 `PSF_mlp`，监督来自 .npz 的 sfr 与 fov/rot 对应的 MTF 切片；
  2. 冻结 PSF，训练 `shift_net` 拟合 .npz 中的 offset（R-G、B-G 色差）。
- 结果含 GT（`IS.Zer2PSF2` 生成）与预测的对比图 `compare.png`。

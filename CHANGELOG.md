# 改动记录 / Changelog

记录本仓库的代码改动。格式:日期 + 改动内容。

## 2026-08-16

- **RAW/Bayer 输入支持**:`real_sfr_pipeline.py` 新增 `load_image_rgb()`,统一把输入转成 RGB uint8 再交给 sfrmat5——相机 RAW(.dng/.arw/.nef/.cr2/...) 用 `rawpy` **线性输出**(`gamma=(1,1)`、`output_bps=16`、`no_auto_bright=True`,**不做 gamma/色调曲线**,避免改变 MTF 形状;相机白平衡为每通道线性增益,不影响 MTF),再按 1%~99% 分位数线性缩放到 8bit;单通道 .tif 用新增 `--bayer-pattern`(RGGB/GRBG/GBRG/BGGR)去马赛克,否则按灰度;16bit 一律线性缩放。
- **棋盘格尺寸自动检测**:`real_sfr_pipeline.py` 的 `detect_grid` 在未指定 `--pattern-cols/rows` 时,按 `AUTO_PATTERNS`(20 种常见内角点尺寸)逐个尝试,**取检测到角点数最多**的(避免对大棋盘误匹配成更小的部分网格);返回实际匹配尺寸并打印。OpenCV 本身必须给定尺寸,无法真正"自动"。
- **配置审计与补全**:
  - `net` 配置生效:修复 `utils/train.py` 中硬编码 `type='ss'` 覆盖配置的问题,改用 `args.get('net', 'ss')`。
  - 新增 `seed`(默认 0)与 `device`(`cuda`/`cpu`/`cuda:0`)配置,`main.py` 读取;`PSF_mlp`/`shift_net`/训练循环的硬编码 `.cuda()` 全部改为跟随 device,CPU 已验证可跑。
  - 清理 `configs/63762BB.yaml` / `configs/real.yaml` 中的无效键(`w_sfr`、`BS`、`loss` 原代码未使用)。
  - `test_real/real_sfr_pipeline.py` 棋盘格内角点数 `PATTERN=(9,6)` 改为 CLI `--pattern-cols/--pattern-rows`。
- **实拍 fov 计算修正**:`generate_fov_weight.fov_from_position` 改为直接用真实相机参数 `fov = atan(r * pixelsize / efl)`,不再依赖 xlsx 的 hfov;`configs/real.yaml` 新增 `efl`(µm)与 `pixelsize`(µm)。
- **model/optics_rgb.py**:`IS.__init__` 新增可选参数 `efl` / `pixelsize`,配置可覆盖焦距与像元尺寸(main.py 与 generate_fov_weight 均传入)。
- **采样数配置**:`utils/train.py` 视场高度采样数由硬编码 21 改为读取 `num_psf`(默认 21),`configs/63762BB.yaml` 新增 `num_psf`。
- **实拍流程(替代仿真)**:新建 `configs/real.yaml`,覆盖数据准备与训练全套;`generate_fov_weight.py` 支持 `real: true` 模式(配置 `mat_dir` + 按边缘像素位置自动计算 fov,新增 `fov_from_position`);`test_real/real_sfr_pipeline.py` 保存 .mat 时新增 `cx/cy/img_h/img_w` 字段。
- **镜头文件解耦**:配置新增 `lens` 键,输入 xlsx 用 `input/{lens}.xlsx`(输出目录名仍用 `filename`);`utils/train.py`/`generate_fov_weight.py`/`generate_edges_rgb.py` 的 `config()` 均支持。
- **main.py**:支持命令行指定配置文件 `python main.py configs/real.yaml`(默认 63762BB.yaml)。
- **编码修复**:三个 `config()` 打开 yaml 显式指定 `encoding='utf-8'`(否则中文注释在 GBK 环境报 UnicodeDecodeError)。
- **model/optics_rgb.py**:`IS.__init__` 新增可选参数 `s_psf` / `sensor_res`,可在配置文件中覆盖 PSF 核大小与传感器分辨率(不填则仍读 xlsx 或按 hfov/efl 计算)。
- **main.py**:构造 `IS` 时从 config 传入 `s_psf` / `sensor_res`。
- **configs/63762BB.yaml**:新增 `sensor_res: [5575, 7434]`(高×宽)与 `s_psf: 25`。
- **AGENTS.md**:新建,沉淀仓库关键命令、数据管线、配置坑点与关键约定(英文,与 CLAUDE.md 互补)。
- **utils/train.py**:新增 `save_psf_grid()`,训练完成后按网格逐格导出独立 PSF 到 `result/{lens}/{时间戳}/psf_grid/`,文件名格式 `GR_{ts}_00_{W}x{H}_cx={cx}_cy={cy}.npz`,内含 `psf/cx/cy`;在 `train()` 收尾处调用。
- **configs/63762BB.yaml**:新增 `grid_rows`/`grid_cols` 配置(默认 9×12),控制 psf_grid 导出密度。
- **环境**:安装缺失依赖 `pandas / pyyaml / matplotlib / tqdm / openpyxl`。

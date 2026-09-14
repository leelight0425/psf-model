"""
真实棋盘格 → 边缘裁剪 → sfrmat5 完整测量流程验证脚本。

流程（对应 generate_edges_rgb.py 合成流程的真实图片版）：
  1. 读取真实棋盘格照片（RGB）
  2. findChessboardCorners + cornerSubPix 检测角点（亚像素精度）
  3. 由角点邻近图构建网格边缘，在每个边缘中点裁剪斜边 patch
  4. 对每个 patch 调用 sfrmat5_py.sfrmat5_rgb 计算 SFR/MTF
  5. 校验 SFR 有效性（max>=0.99 且无 NaN），汇总统计

用法：
    python real_sfr_pipeline.py                          # 处理 test_real/downloads 下全部图片
    python real_sfr_pipeline.py <img1> [img2 ...]        # 指定图片
    python real_sfr_pipeline.py --save <out_dir>         # 同时保存 .mat 结果

说明：
    ISO 12233 斜边法要求边缘与垂直方向偏离 5°~45°。
    若棋盘格拍摄近乎正交（倾斜 <2°），投影分箱会产生零计数 → SFR=NaN，
    这是方法的物理限制而非代码 bug（参考 sfrmat5.m 中关于零计数的警告）。
"""

import os
import sys
import re
import warnings
import numpy as np
import cv2
import yaml
from scipy.spatial import cKDTree
from scipy.io import savemat

# 项目根目录加入 sys.path，复用 sfrmat5_py 端口
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import sfrmat5_py as sf

warnings.filterwarnings('ignore')

PATTERN = (9, 6)          # 默认棋盘格内角点数（列, 行），OpenCV 约定
# 未指定 --pattern-cols/rows 时自动尝试的候选尺寸（OpenCV 必须给定尺寸，无法自动检测）
AUTO_PATTERNS = [(9, 6), (10, 7), (10, 8), (11, 8), (12, 8), (12, 9), (13, 9), (13, 10),
                 (14, 10), (9, 7), (8, 6), (8, 5), (7, 5), (9, 5), (11, 7), (11, 9),
                 (12, 10), (15, 10), (10, 9), (9, 8)]
TILT_MIN_DEG = 5.0        # ISO 12233: 斜边最小倾斜角
PATCH_FRAC = 0.35         # patch 半宽 = frac * 方格尺寸（保证只含一条边）

# 相机 RAW 扩展名（需 rawpy 处理），其余由 OpenCV 读取
RAW_EXTENSIONS = {'.dng', '.arw', '.nef', '.cr2', '.cr3', '.raf', '.rw2', '.orf', '.pef', '.srw'}
# 裸 Bayer RAW 扩展名：无元数据，直接由 OpenCV 按 --bayer-pattern 读取并去马赛克
BAYER_RAW_EXTENSIONS = {'.raw'}
# Bayer pattern → OpenCV 去马赛克转换（RGGB 为仓库默认，见 notes.md）
BAYER_TO_CV2 = {
    'RGGB': cv2.COLOR_BayerRG2RGB,
    'GRBG': cv2.COLOR_BayerGR2RGB,
    'GBRG': cv2.COLOR_BayerGB2RGB,
    'BGGR': cv2.COLOR_BayerBG2RGB,
}


def load_image_rgb(path, bayer_pattern=None):
    """把任意输入图片统一转为 RGB uint8 (H, W, 3) 供 sfrmat5 使用。

    - 相机 RAW（.dng/.arw/.nef/...）:用 rawpy 后处理（去马赛克 + 相机白平衡 + gamma）
    - 裸 Bayer RAW（.raw）:无元数据，需用 --bayer-pattern 由 OpenCV 读取单通道并去马赛克
    - npz（.npz）:读取其中的图像数组，单通道按 --bayer-pattern 去马赛克
    - 单通道（灰度或 Bayer 马赛克 .tif）:指定 --bayer-pattern 时按 Bayer 去马赛克，否则按灰度
    - 16bit:按 1%~99% 分位数裁剪缩放到 8bit（保证棋盘格对比度）
    - 其余:BGR→RGB
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError:
            raise SystemExit(f'{path} 是 RAW 文件，需要先安装 rawpy：pip install rawpy')
        raw = rawpy.imread(path)
        # 线性输出：不做 gamma/色调曲线（会改变 MTF 形状）、16bit、关闭自动提亮。
        # 相机白平衡是每通道线性增益，不影响 MTF。随后由下方 16bit 分位数裁剪线性缩放到 8bit。
        img = raw.postprocess(use_camera_wb=True, no_auto_bright=True,
                              output_bps=16, gamma=(1, 1))
        raw.close()
    elif ext in BAYER_RAW_EXTENSIONS:
        # 裸 Bayer RAW：无 rawpy 元数据，按 --bayer-pattern 读取单通道数据并去马赛克。
        if not bayer_pattern:
            raise SystemExit(f'{path} 是裸 Bayer RAW 文件，必须用 --bayer-pattern 指定布局（RGGB/GRBG/GBRG/BGGR）')
        code = BAYER_TO_CV2.get(bayer_pattern.upper())
        if code is None:
            raise SystemExit(f'未知 bayer pattern：{bayer_pattern}（可选 RGGB/GRBG/GBRG/BGGR）')
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        # 8-bit 或 16-bit 判断：size = H*W*bytes_per_pixel
        if data.size % 2 == 0 and data.size // 2 % 2 == 0:
            u16 = data.view('<u2')
            # 若按 16bit 能组成矩形且数值范围合理，则按 16bit 解析
            if (u16.size % 2 == 0) and (u16.max() > 255 or u16.max() <= 0):
                data = u16
        n = data.size
        # 从文件名解析宽x高（如 4000x3000 → W=4000, H=3000）；失败则尝试两种朝向。
        dims = None
        m = re.search(r'(\d+)\s*[xX]\s*(\d+)', os.path.basename(path))
        if m:
            w_num, h_num = int(m.group(1)), int(m.group(2))
            if w_num * h_num == n:
                dims = [(h_num, w_num)]
        if dims is None:
            dims = [(3000, 4000), (4000, 3000)]
        img = None
        for (h, w) in dims:
            if h * w == n:
                img = data.reshape(h, w)
                break
        if img is None:
            raise SystemExit(f'{path} 无法根据文件大小 {data.nbytes} 推断单通道 Bayer 形状')
        img = cv2.cvtColor(img, code)
    elif ext == '.npz':
        # npz 图像通常保存为 im 数组；与图片/裸 Bayer RAW 一样继续走后续流程。
        with np.load(path, allow_pickle=False) as archive:
            if 'im' in archive.files:
                img = archive['im']
            elif len(archive.files) == 1:
                img = archive[archive.files[0]]
            else:
                raise SystemExit(f'{path} 未找到唯一图像数组，包含的键为：{archive.files}')

        if img.ndim not in (2, 3):
            raise SystemExit(f'{path} 中的图像数组必须是二维或三维，实际形状为：{img.shape}')

        # OpenCV Bayer 转换只支持 8bit/16bit；实际 npz 为 0~1 的 float32，先转成 16bit。
        if img.dtype != np.uint8 and img.dtype != np.uint16:
            if np.issubdtype(img.dtype, np.floating):
                finite = img[np.isfinite(img)]
                if finite.size == 0:
                    return None
                scale = 65535.0 if finite.max() <= 1.0 and finite.min() >= 0.0 else 1.0
                img = np.clip(img * scale, 0, 65535).astype(np.uint16)
            else:
                img = np.clip(img, 0, 65535).astype(np.uint16)

        if img.ndim == 2:
            if bayer_pattern:
                code = BAYER_TO_CV2.get(bayer_pattern.upper())
                if code is None:
                    raise SystemExit(f'未知 bayer pattern：{bayer_pattern}（可选 RGGB/GRBG/GBRG/BGGR）')
                img = cv2.cvtColor(img, code)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            raise SystemExit(f'{path} 的图像数组通道数不支持：{img.shape}')
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            if bayer_pattern:
                code = BAYER_TO_CV2.get(bayer_pattern.upper())
                if code is None:
                    raise SystemExit(f'未知 bayer pattern：{bayer_pattern}（可选 RGGB/GRBG/GBRG/BGGR）')
                img = cv2.cvtColor(img, code)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    if img.dtype == np.uint16:
        p_low, p_high = np.percentile(img, (1, 99))
        img = np.clip((img.astype(np.float32) - p_low) / max(p_high - p_low, 1) * 255, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def detect_grid(bgr_img, pattern=None):
    """检测棋盘格角点，返回 (亚像素角点数组 (N,2), 实际采用的 pattern) 或 (None, None)。

    pattern 为 None 时按 AUTO_PATTERNS 逐个尝试，取检测到角点数最多的（避免对
    大棋盘格误匹配成更小的部分网格）。
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    candidates = [pattern] if pattern is not None else AUTO_PATTERNS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    best = None  # (角点数, pts, pattern)
    for p in candidates:
        ret, corners = cv2.findChessboardCorners(gray, p, None)
        if not ret:
            continue
        c2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        n = len(c2)
        if best is None or n > best[0]:
            best = (n, c2.reshape(-1, 2), p)
    # SB is slower but handles large, low-contrast, or mildly distorted boards
    # that the legacy detector can miss.
    if best is None and hasattr(cv2, 'findChessboardCornersSB'):
        for p in candidates:
            ret, corners = cv2.findChessboardCornersSB(gray, p, None)
            if not ret:
                continue
            n = len(corners)
            if best is None or n > best[0]:
                best = (n, corners.reshape(-1, 2), p)
    if best is None:
        return None, None
    return best[1], best[2]


def grid_edges(pts):
    """
    由角点邻近图构建网格边缘（不依赖行列顺序，对旋转棋盘格稳健）。
    返回 list[(i, j, mid_x, mid_y, tilt_deg)]，只保留倾斜角达标且距离合理的边缘。
    """
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=9)

    # 每个角点的 4 个最近邻（去掉自身）
    neighbors = {}
    for i in range(len(pts)):
        cand = []
        for k in range(1, 9):
            j = int(idx[i, k])
            if j == i:
                continue
            cand.append((float(np.linalg.norm(pts[j] - pts[i])), j, pts[j] - pts[i]))
        cand.sort(key=lambda t: t[0])
        neighbors[i] = cand[:4]

    dists = [dd for i in neighbors for (dd, j, v) in neighbors[i]]
    sq = np.median(dists)  # 方格尺寸估计

    edges = []
    seen = set()
    for i in neighbors:
        for (dd, j, v) in neighbors[i]:
            if dd > sq * 1.8:      # 排除斜对角/越界邻居
                continue
            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            seen.add(key)
            mid = (pts[i] + pts[j]) / 2
            # 边缘方向（连线方向）与最近轴（水平/垂直）的偏离角
            ang = np.degrees(np.arctan2(v[1], v[0])) % 180
            tilt = min(ang, 180 - ang)          # 相对水平
            tilt = min(tilt, 90 - tilt) if tilt <= 90 else tilt  # 取相对最近主轴
            edges.append((i, j, mid[0], mid[1], float(tilt)))
    return edges, sq


def crop_patch(rgb_img, mid_x, mid_y, half):
    """以 (mid_x, mid_y) 为中心裁剪 (2*half)x(2*half) 方形 patch，返回 RGB 数组或 None。"""
    h, w = rgb_img.shape[:2]
    x0, y0 = int(mid_x) - half, int(mid_y) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half > w or y0 + 2 * half > h:
        return None
    return rgb_img[y0:y0 + 2 * half, x0:x0 + 2 * half]


def process_image(path, save_dir=None, npol=5, min_mtf=0.99, mtf50_level=0.5,
                  tilt_min=TILT_MIN_DEG, patch_frac=PATCH_FRAC, pattern=PATTERN,
                  bayer_pattern=None):
    """处理单张真实棋盘格图片，返回统计 dict。

    Args:
        path: 图片路径（JPEG/PNG/TIFF/相机 RAW，RAW 自动转 RGB）
        save_dir: 若给定，保存有效 .mat
        npol: sfrmat5 边缘拟合多项式阶数
        min_mtf: 有效性条件①——MTF 峰值需 ≥ 该值
        mtf50_level: 有效性条件③——MTF 需降到 ≤ 该值（MTF50 可定义）
        tilt_min: 斜边最小倾斜角（度），低于则跳过
        patch_frac: patch 半宽 = frac * 方格尺寸
        pattern: 棋盘格内角点数 (列, 行)
        bayer_pattern: 单通道 .tif 的 Bayer 排列 (RGGB/GRBG/GBRG/BGGR)，None=按灰度
    """
    rgb = load_image_rgb(path, bayer_pattern)
    if rgb is None:
        print(f"  ERROR: 无法读取 {path}")
        return None

    name = os.path.splitext(os.path.basename(path))[0]
    pts, used_pattern = detect_grid(rgb, pattern)
    if pts is None:
        print(f"  ERROR: 未检测到棋盘格角点 (尝试 pattern={AUTO_PATTERNS if pattern is None else pattern})")
        return None
    if pattern is None:
        print(f"  [{name}] 自动匹配棋盘格内角点数: {used_pattern}")

    edges, sq = grid_edges(pts)
    half = max(int(sq * patch_frac), 10)

    n_tot = n_ok = n_nan = n_low = n_tilt = 0
    mtf50_list = []
    saved = 0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"  [{name}] 检测到 {len(pts)} 角点, 方格~{sq:.1f}px, 候选边缘 {len(edges)}")

    for (i, j, mx, my, tilt) in edges:
        if tilt < tilt_min:
            n_tilt += 1
            continue  # 倾斜角不达标，跳过（ISO 12233 要求）

        patch = crop_patch(rgb, mx, my, half)
        if patch is None:
            continue
        n_tot += 1
        h, w = rgb.shape[:2]
        field_azimuth = np.degrees(np.arctan2(h / 2.0 - my, mx - w / 2.0))
        try:
            sfr, esf, offset, rot = sf.sfrmat5_rgb(
                patch, npol=npol, wflag=0, field_azimuth=field_azimuth)
        except Exception:
            n_nan += 1
            continue

        if np.any(np.isnan(sfr)):
            n_nan += 1
            continue
        if np.max(sfr[:, 1]) < min_mtf:
            n_low += 1
            continue
        # 仅当 MTF 曲线确实降到 mtf50_level 以下（MTF50 可定义）才视为有效
        if not np.any(sfr[:, 1] <= mtf50_level):
            n_low += 1
            continue

        n_ok += 1
        freq = sfr[:, 0]
        mtf50 = freq[np.argmin(np.abs(sfr[:, 1] - mtf50_level))]
        mtf50_list.append(mtf50)

        if save_dir:
            out = os.path.join(save_dir, f"{name}_e{i:02d}{j:02d}_tilt{tilt:.1f}.mat")
            savemat(out, {
                'fov': np.array([[np.nan]]),
                'rot': np.array([[rot]]),
                'sfr': sfr,
                'offset': offset.reshape(1, 2),
                'tilt': np.array([[tilt]]),
                'cx': np.array([[mx]]),
                'cy': np.array([[my]]),
                'img_h': np.array([[h]]),
                'img_w': np.array([[w]]),
            })
            saved += 1

    mtf50_arr = np.array(mtf50_list)
    mtf50_finite = mtf50_arr[np.isfinite(mtf50_arr)]
    stats = {
        'name': name, 'sq': sq, 'half': half, 'n_edges': len(edges),
        'n_tot': n_tot, 'n_ok': n_ok, 'n_nan': n_nan, 'n_low': n_low,
        'n_tilt_skip': n_tilt,
        'mtf50_mean': float(np.nanmean(mtf50_arr)) if mtf50_arr.size else np.nan,
        'mtf50_med': float(np.nanmedian(mtf50_arr)) if mtf50_arr.size else np.nan,
    }
    rate = n_ok / n_tot if n_tot else 0
    print(f"    通过 {n_ok}/{n_tot} (率 {rate:.1%}); NaN={n_nan}, 不满足阈值={n_low}, 倾斜跳过={n_tilt}")
    if mtf50_arr.size:
        print(f"    MTF50 (R 通道): mean={stats['mtf50_mean']:.3f}, median={stats['mtf50_med']:.3f}, "
              f"range=[{mtf50_finite.min():.3f}, {mtf50_finite.max():.3f}] (n={len(mtf50_finite)})")
    if save_dir:
        print(f"    已保存 {saved} 个 .mat 到 {save_dir}")
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="真实棋盘格 → 边缘裁剪 → sfrmat5 测量流程。"
                    "未指定图片时默认处理 test_real/downloads 下全部图片。")
    parser.add_argument('images', nargs='*', help='图片路径（可多个）')
    parser.add_argument('--config', default='configs/real.yaml',
                        help='YAML 配置路径（默认 configs/real.yaml）')
    parser.add_argument('--save', metavar='DIR', default='test_real/out',
                        help='保存有效 .mat 的目录（默认 test_real/out）')
    parser.add_argument('--npol', type=int, default=5, help='sfrmat5 边缘拟合多项式阶数 (默认 5)')
    parser.add_argument('--min-mtf', type=float, default=0.99,
                        help='有效性①：MTF 峰值 ≥ 该值 (默认 0.99)')
    parser.add_argument('--mtf50-level', type=float, default=0.5,
                        help='有效性③：MTF 需降到 ≤ 该值 (默认 0.5)')
    parser.add_argument('--tilt-min', type=float, default=TILT_MIN_DEG,
                        help='斜边最小倾斜角/度 (默认 5.0)')
    parser.add_argument('--patch-frac', type=float, default=PATCH_FRAC,
                        help='patch 半宽 = frac × 方格尺寸 (默认 0.35)')
    parser.add_argument('--pattern-cols', type=int, default=None,
                        help='棋盘格内角点列数 (默认自动尝试候选)')
    parser.add_argument('--pattern-rows', type=int, default=None,
                        help='棋盘格内角点行数 (默认自动尝试候选)')
    parser.add_argument('--bayer-pattern', default='GBRG',
                        help='单通道 .tif 的 Bayer 排列 RGGB/GRBG/GBRG/BGGR (默认按灰度)')
    ns = parser.parse_args()
    config = {}
    if ns.config:
        try:
            with open(ns.config, encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            if ns.config != 'configs/real.yaml':
                raise
    pattern_cols = ns.pattern_cols or config.get('pattern_cols')
    pattern_rows = ns.pattern_rows or config.get('pattern_rows')
    pattern = (int(pattern_cols), int(pattern_rows)) if pattern_cols and pattern_rows else None

    args = ns.images
    if not args:
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads')
        args = [os.path.join(default_dir, f)
                for f in sorted(os.listdir(default_dir))
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.raw', '.npz'))]

    print("=" * 70)
    print("真实棋盘格 → 边缘裁剪 → sfrmat5 测量流程验证")
    print(f"  判定条件: min_mtf≥{ns.min_mtf}, MTF 降到≤{ns.mtf50_level}, "
          f"tilt_min={ns.tilt_min}°, npol={ns.npol}, patch_frac={ns.patch_frac}, pattern={pattern}")
    print("=" * 70)
    all_stats = []
    for p in args:
        print(f"\n处理: {p}")
        st = process_image(p, ns.save, npol=ns.npol, min_mtf=ns.min_mtf,
                           mtf50_level=ns.mtf50_level, tilt_min=ns.tilt_min,
                           patch_frac=ns.patch_frac, pattern=pattern,
                           bayer_pattern=ns.bayer_pattern)
        if st:
            all_stats.append(st)

    if all_stats:
        print("\n" + "=" * 70)
        print("汇总")
        print("=" * 70)
        hdr = f"{'图片':<12}{'方格px':>7}{'候选边':>7}{'通过':>7}{'率':>7}  MTF50(med)"
        print(hdr)
        for st in all_stats:
            rate = st['n_ok'] / st['n_tot'] if st['n_tot'] else 0
            mtf = f"{st['mtf50_med']:.3f}" if not np.isnan(st['mtf50_med']) else 'NaN'
            print(f"{st['name']:<12}{st['sq']:>7.1f}{st['n_edges']:>7}{st['n_ok']:>6}/{st['n_tot']:<2}{rate:>7.1%}  {mtf}")


if __name__ == '__main__':
    main()

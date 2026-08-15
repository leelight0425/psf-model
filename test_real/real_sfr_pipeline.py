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
import warnings
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.io import savemat

# 项目根目录加入 sys.path，复用 sfrmat5_py 端口
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import sfrmat5_py as sf

warnings.filterwarnings('ignore')

PATTERN = (9, 6)          # 棋盘格内角点数（列, 行），OpenCV 约定
TILT_MIN_DEG = 5.0        # ISO 12233: 斜边最小倾斜角
PATCH_FRAC = 0.35         # patch 半宽 = frac * 方格尺寸（保证只含一条边）


def detect_grid(bgr_img):
    """检测棋盘格角点，返回亚像素角点数组 (N,2) 或 None。"""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, PATTERN, None)
    if not ret:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    c2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return c2.reshape(-1, 2)


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


def crop_patch(bgr_img, mid_x, mid_y, half):
    """以 (mid_x, mid_y) 为中心裁剪 (2*half)x(2*half) 方形 patch，返回 RGB 数组或 None。"""
    h, w = bgr_img.shape[:2]
    x0, y0 = int(mid_x) - half, int(mid_y) - half
    if x0 < 0 or y0 < 0 or x0 + 2 * half > w or y0 + 2 * half > h:
        return None
    patch = bgr_img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)


def process_image(path, save_dir=None, npol=5, min_mtf=0.99, mtf50_level=0.5,
                  tilt_min=TILT_MIN_DEG, patch_frac=PATCH_FRAC):
    """处理单张真实棋盘格图片，返回统计 dict。

    Args:
        path: 图片路径
        save_dir: 若给定，保存有效 .mat
        npol: sfrmat5 边缘拟合多项式阶数
        min_mtf: 有效性条件①——MTF 峰值需 ≥ 该值
        mtf50_level: 有效性条件③——MTF 需降到 ≤ 该值（MTF50 可定义）
        tilt_min: 斜边最小倾斜角（度），低于则跳过
        patch_frac: patch 半宽 = frac * 方格尺寸
    """
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"  ERROR: 无法读取 {path}")
        return None

    name = os.path.splitext(os.path.basename(path))[0]
    pts = detect_grid(bgr)
    if pts is None:
        print(f"  ERROR: 未检测到棋盘格角点 pattern={PATTERN}")
        return None

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

        rgb = crop_patch(bgr, mx, my, half)
        if rgb is None:
            continue
        n_tot += 1
        try:
            sfr, esf, offset, rot = sf.sfrmat5_rgb(rgb, npol=npol, wflag=0)
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
    parser.add_argument('--save', metavar='DIR', help='保存有效 .mat 到目录')
    parser.add_argument('--npol', type=int, default=5, help='sfrmat5 边缘拟合多项式阶数 (默认 5)')
    parser.add_argument('--min-mtf', type=float, default=0.99,
                        help='有效性①：MTF 峰值 ≥ 该值 (默认 0.99)')
    parser.add_argument('--mtf50-level', type=float, default=0.5,
                        help='有效性③：MTF 需降到 ≤ 该值 (默认 0.5)')
    parser.add_argument('--tilt-min', type=float, default=TILT_MIN_DEG,
                        help='斜边最小倾斜角/度 (默认 5.0)')
    parser.add_argument('--patch-frac', type=float, default=PATCH_FRAC,
                        help='patch 半宽 = frac × 方格尺寸 (默认 0.35)')
    ns = parser.parse_args()

    args = ns.images
    if not args:
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloads')
        args = [os.path.join(default_dir, f)
                for f in sorted(os.listdir(default_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]

    print("=" * 70)
    print("真实棋盘格 → 边缘裁剪 → sfrmat5 测量流程验证")
    print(f"  判定条件: min_mtf≥{ns.min_mtf}, MTF 降到≤{ns.mtf50_level}, "
          f"tilt_min={ns.tilt_min}°, npol={ns.npol}, patch_frac={ns.patch_frac}")
    print("=" * 70)
    all_stats = []
    for p in args:
        print(f"\n处理: {p}")
        st = process_image(p, ns.save, npol=ns.npol, min_mtf=ns.min_mtf,
                           mtf50_level=ns.mtf50_level, tilt_min=ns.tilt_min,
                           patch_frac=ns.patch_frac)
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

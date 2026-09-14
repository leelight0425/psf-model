"""
从训练结果批量生成不同网格位置的 PSF 模糊核, 核中心以绝对像素坐标标注。

位置链路 (与实拍训练数据同源):
    cx/cy (绝对像素) -> r -> fov = atan(r*pixelsize/efl)   # 同 generate_fov_weight.fov_from_position
    H = sin(fov)/sin(hfov)                                 # 同 utils.tools.fov2H
    rot = atan2(dy_up, dx) - pi/2                          # 同 utils.train.save_psf_grid 旋转约定

PSF 取法:
    - 训练覆盖范围内 (H <= H_trained.max): 从 psf_after_shift.npy(默认)或 psf.npy(--ori)
      的 psfs 序列取最近 H 的条目, 再按 rot 旋转
    - 超出训练范围: 若结果目录有 coe.npy(train.py 保存的 Seidel 系数), 用 log-log 幂律
      (指数夹在 [0,3]) 外推系数并用 wavefront->PSF 物理模型重建; 否则退回最近邻填充
    - 训练数据洞内 (如 fov 7~15° 空档): 若结果目录有 net.pth(train.py 的训练 MLP 权重),
      在精确 H 处用训练好的网络前向, manifest source 标 'mlp-hole'; 否则退回最近邻
    - 外推位置在文件名/PNG/manifest 中以 *EXT 标记

用法:
    python generate_psf_kernels.py configs/real.yaml
    python generate_psf_kernels.py configs/real.yaml --result result/my_real/0902-153000 --out <dir>
    python generate_psf_kernels.py configs/real.yaml --ori          # 用色偏校正前的 PSF

输出 (每个网格位置):
    psf_cx{cx}_cy{cy}.npz   psf(H,W,3 float32, 单通道和=1) + cx/cy/H/fov/rot/centroid/extrapolated
    psf_cx{cx}_cy{cy}.png   标注图: 十字=几何中心, 圆圈=质心, 文字=坐标/H/fov
    manifest.csv            汇总表
"""
import argparse
import csv
import glob
import math
import os
import sys

import cv2
import numpy as np
import torch
import yaml

from model.optics_rgb import IS
from model.PSF_mlp import seidel2wavefront, wavefront2psf, PSF_mlp
from utils.train import rotatepsf
import utils.tools as tools


def load_config(path):
    with open(path, encoding='utf-8') as f:
        args = yaml.safe_load(f)
    # 与 utils.train.config 相同的派生: result_path = {cwd}/{result}/{filename}
    filename = args['filename']
    args['result_path'] = os.path.join(os.getcwd(), args['result'], filename)
    return args


def find_latest_result(result_path, psf_name):
    """在 result/{filename}/ 下找包含 psf_name 的最新时间戳目录。"""
    candidates = sorted(
        d for d in glob.glob(os.path.join(result_path, '*'))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, psf_name)))
    if not candidates:
        raise SystemExit(f'{result_path} 下没有包含 {psf_name} 的结果目录, '
                         f'请先运行 main.py 或用 --result 指定')
    return candidates[-1]


def detect_trained_H_max(psfs, Hs):
    """无 coe.npy 时, 用 psfs 序列的重复段估计实际训练覆盖的 H 上限。"""
    boundary = 0
    for i in range(1, len(psfs)):
        if not torch.allclose(psfs[i].float(), psfs[i - 1].float(), atol=1e-6):
            boundary = i
    return float(Hs[boundary])


def fit_extrapolated_coe(H_t, C_t, H_target, H_max_trained):
    """对单个颜色的 Seidel 系数按幂律外推到 H_target。

    log-log 线性回归拟合 |c| ~ a*H^p, 指数夹在 [0,3](初级像差场标度律范围),
    幅度夹在 2x 训练最大值内防发散, 符号取最靠近外推区的训练样本。
    """
    mask = (H_t > 1e-6) & (H_t <= H_max_trained + 1e-6)
    Hf, C = H_t[mask], C_t[mask]
    out = torch.zeros(C.shape[1])
    for k in range(C.shape[1]):
        c = C[:, k]
        ref = c[torch.argmax(Hf)]
        m = c.abs() > 1e-12
        if m.sum() < 2:
            out[k] = ref
            continue
        lgH, lgC = torch.log(Hf[m]), torch.log(c[m].abs())
        dH, dC = lgH - lgH.mean(), lgC - lgC.mean()
        p = float((dH * dC).sum() / (dH * dH).sum().clamp_min(1e-12))
        p = min(max(p, 0.0), 3.0)
        loga = float((lgC - p * lgH).mean())
        log_cap = math.log(2.0 * float(c.abs().max()) + 1e-30)
        val = math.exp(min(loga + p * math.log(H_target), log_cap))
        out[k] = (1.0 if float(ref) >= 0 else -1.0) * val
    return out


def rebuild_psf(coe_rgb, wf_mod, psf_mod, IS_, device):
    """由外推系数 (3,9) 经 wavefront->PSF 物理模型重建 (s,s,3) PSF。"""
    parts = []
    for color in range(3):
        seidel = torch.tensor(coe_rgb[color], dtype=torch.float32,
                              device=device).unsqueeze(0)
        WF = wf_mod(seidel, IS_, color, 1)
        AP = psf_mod(WF)
        parts.append(tools.downsample(AP.squeeze(), IS_.s_psf).squeeze(0))
    return torch.stack(parts, dim=-1)


def try_load_net(result_dir):
    """读取 train.py 导出的 net.pth(训练好的 MLP 权重); 旧结果没有则返回 None。"""
    p = os.path.join(result_dir, 'net.pth')
    if not os.path.exists(p):
        return None
    ckpt = torch.load(p, map_location='cpu')
    return ckpt.get('net', ckpt)


def hole_forward_psf(net, IS_, H):
    """洞内精确 H: 训练好的 MLP 逐色前向下采样, 返回 (s,s,3) PSF。"""
    parts = []
    with torch.no_grad():
        for color in range(3):
            _, _, psf = net(IS_, torch.tensor([H], dtype=torch.float32), color)
            parts.append(tools.downsample(psf.squeeze(0), IS_.s_psf).squeeze(0))
    return torch.stack(parts, dim=-1)


def make_hole_test(H_trained, factor=3.0):
    """训练 H 序列(去重排序)里相邻间隔显著大于中位间隔的段视为数据洞。
    返回判定函数: 目标 H 落在洞段内部 -> True (仅内部; 首尾以外不算洞)。"""
    Hs = np.unique(np.sort(np.asarray(H_trained, dtype=np.float64)))
    if len(Hs) < 3:
        return lambda H: False
    gaps = np.diff(Hs)
    hole_lo = Hs[:-1][gaps > factor * float(np.median(gaps))]
    hole_hi = Hs[1:][gaps > factor * float(np.median(gaps))]
    if len(hole_lo) == 0:
        return lambda H: False
    return lambda H: bool(np.any((H > hole_lo) & (H < hole_hi)))


def render_png(psf, cx, cy, i, j, H, fov, extrapolated, out_path, gamma, up):
    """PSF 可视化: 十字=几何中心, 圆圈=质心, 文字=绝对坐标/视场。"""
    s = psf.shape[0]
    chan = psf / np.maximum(psf.max(axis=(0, 1)), 1e-12)
    img = (np.clip(chan, 0, 1) ** gamma * 255).astype(np.uint8)
    img = cv2.resize(img, (s * up, s * up), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    c = s * up // 2

    yy, xx = np.mgrid[0:s, 0:s]
    w = psf.sum(axis=-1)
    w_sum = w.sum()
    cy_c = float((yy * w).sum() / w_sum)
    cx_c = float((xx * w).sum() / w_sum)
    cen = (int(round(cx_c * up + up / 2)), int(round(cy_c * up + up / 2)))

    cv2.line(img, (c - 40, c), (c + 40, c), (255, 255, 0), 1)   # 几何中心十字
    cv2.line(img, (c, c - 40), (c, c + 40), (255, 255, 0), 1)
    cv2.circle(img, cen, 8, (0, 0, 255), 2)                      # 质心
    if extrapolated:
        cv2.rectangle(img, (0, 0), (s * up - 1, s * up - 1), (0, 215, 255), 3)
    cv2.putText(img, f'cx={cx:.0f} cy={cy:.0f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    tag = ' *EXT' if extrapolated else ''
    cv2.putText(img, f'({i},{j}) H={H:.3f} fov={fov:.1f}{tag}', (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(out_path, img)
    return cx_c - (s - 1) / 2, cy_c - (s - 1) / 2   # 质心相对几何中心, dy 向下为正


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('config', nargs='?', default='configs/real.yaml')
    parser.add_argument('--result', default=None, help='训练结果目录(默认取最新)')
    parser.add_argument('--out', default=None, help='输出目录(默认 <result>/psf_kernels)')
    parser.add_argument('--ori', action='store_true', help='用色偏校正前 psf.npy 而非 psf_after_shift.npy')
    parser.add_argument('--grid-rows', type=int, default=None)
    parser.add_argument('--grid-cols', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=0.5, help='可视化 gamma(越小越亮)')
    ns = parser.parse_args()

    args = load_config(ns.config)
    device = torch.device(args.get('device', 'cpu'))
    psf_name = 'psf.npy' if ns.ori else 'psf_after_shift.npy'
    result_dir = ns.result or find_latest_result(args['result_path'], psf_name)
    psf_file = os.path.join(result_dir, psf_name)
    if not os.path.exists(psf_file):
        raise SystemExit(f'未找到 {psf_file}')
    print(f'结果目录: {result_dir} ({psf_name})')

    out_dir = ns.out or os.path.join(result_dir, 'psf_kernels')
    os.makedirs(out_dir, exist_ok=True)

    # IS 与 Seidel 基(与 main.py real 分支一致)
    if args.get('wavelengths') is None or args.get('na') is None or args.get('hfov') is None:
        raise SystemExit('config 缺少 wavelengths/na/hfov, 仅支持实拍(real)模式')
    IS_ = IS(filepath=None, s_psf=args.get('s_psf'), sensor_res=args.get('sensor_res'),
             efl=args['efl'], pixelsize=args['pixelsize'],
             wavelengths=args['wavelengths'], na=args['na'], hfov=args['hfov'])
    H_img, W_img = int(IS_.res[0]), int(IS_.res[1])
    efl, pixelsize, hfov = float(args['efl']), float(args['pixelsize']), int(args['hfov'])

    # 训练 PSF 序列与系数
    psfs = np.load(psf_file, allow_pickle=True).item()['psfs']
    num_psf = len(psfs)
    Hs = torch.linspace(0, 1, num_psf)
    coe_path = os.path.join(result_dir, 'coe.npy')
    coe_data = None
    if os.path.exists(coe_path):
        coe_data = np.load(coe_path, allow_pickle=True).item()
        Hs = torch.tensor(np.asarray(coe_data['H'], dtype=np.float32))
        H_train_max = float(np.max(coe_data['H_trained']))
        basis_type = coe_data.get('basis_type', args.get('net', 'ss'))
        print(f'coe.npy: 基={basis_type}, 训练覆盖 H<={H_train_max:.3f}')
    else:
        H_train_max = detect_trained_H_max(psfs, Hs)
        basis_type = args.get('net', 'ss')
        print(f'警告: 无 coe.npy, 外推退化为最近邻填充 (估计训练覆盖 H<={H_train_max:.3f})')
    IS_.seidel_basis = IS_.s_basis(IS_.wf_res, type=basis_type)
    wf_mod, psf_mod = seidel2wavefront(device), wavefront2psf(device)

    # 数据洞内前向: 需要 train.py 导出的 net.pth (训练好的 H->Seidel MLP)
    net_state = try_load_net(result_dir)
    trained_mlp = None
    if net_state is not None:
        trained_mlp = PSF_mlp(device=device)
        trained_mlp.load_state_dict(net_state)
        trained_mlp.eval()
        print('net.pth: 已加载训练 MLP, 数据洞内在精确 H 处前向 (mlp-hole)')
    else:
        print('net.pth: 未找到 (需重训导出), 数据洞内退回最近邻')
    hole_test = (make_hole_test(coe_data['H_trained'])
                 if coe_data is not None and 'H_trained' in coe_data
                 else (lambda H: False))

    rows = ns.grid_rows or int(args.get('grid_rows', 9))
    cols = ns.grid_cols or int(args.get('grid_cols', 12))
    up = 16  # PNG 放大倍数
    manifest_rows = []

    print(f'网格 {rows}x{cols}, 传感器 {W_img}x{H_img}, 输出: {out_dir}')
    for i in range(rows):
        for j in range(cols):
            # 绝对像素坐标 -> fov -> H -> rot (与训练数据同源公式)
            cx = (j + 0.5) * W_img / cols
            cy = (i + 0.5) * H_img / rows
            dx, dy_up = cx - W_img / 2, H_img / 2 - cy
            r = math.hypot(dx, dy_up)
            fov = math.degrees(math.atan(r * pixelsize / efl))
            H = math.sin(math.radians(fov)) / math.sin(math.radians(hfov))
            rot_rad = math.atan2(dy_up, dx) - math.pi / 2

            extrapolated = H > H_train_max + 1e-6
            if extrapolated and coe_data is not None:
                C_all = [coe_data[c] for c in ('R', 'G', 'B')]
                coe_ext = np.stack([
                    fit_extrapolated_coe(Hs, torch.tensor(np.asarray(c, dtype=np.float32)),
                                         H, H_train_max).numpy() for c in C_all])
                psf3 = rebuild_psf(coe_ext, wf_mod, psf_mod, IS_, device)
                source = 'coe-extrapolated'
            elif not extrapolated and hole_test(H) and trained_mlp is not None:
                psf3 = hole_forward_psf(trained_mlp, IS_, H)
                source = 'mlp-hole'
            else:
                idx = int(torch.argmin(torch.abs(Hs - H)))
                p = psfs[idx]
                if torch.is_tensor(p):          # 兼容旧版导出里带梯度的 tensor
                    p = p.detach().cpu().numpy()
                psf3 = torch.tensor(np.asarray(p, dtype=np.float32))
                source = 'nearest' if not extrapolated else 'nearest(extrapolated)'

            psf_rot = torch.stack([rotatepsf(psf3[:, :, c].float(), rot_rad)
                                   for c in range(3)], dim=-1)
            psf_rot = (psf_rot / psf_rot.sum(dim=(0, 1), keepdim=True).clamp_min(1e-12))
            psf_np = psf_rot.detach().cpu().numpy().astype(np.float32)

            tag = '_EXT' if extrapolated else ''
            stem = f'psf_cx{int(round(cx))}_cy{int(round(cy))}{tag}'
            np.savez(os.path.join(out_dir, stem + '.npz'),
                     psf=psf_np, cx=np.float32(cx), cy=np.float32(cy),
                     H=np.float32(H), fov=np.float32(fov), rot=np.float32(math.degrees(rot_rad)),
                     extrapolated=np.int8(extrapolated))
            cd_dx, cd_dy = render_png(psf_np, cx, cy, i, j, H, fov, extrapolated,
                                      os.path.join(out_dir, stem + '.png'),
                                      ns.gamma, up)
            manifest_rows.append([i, j, f'{cx:.1f}', f'{cy:.1f}', f'{H:.4f}',
                                  f'{fov:.2f}', f'{math.degrees(rot_rad):.2f}',
                                  f'{cd_dx:.3f}', f'{cd_dy:.3f}',
                                  int(extrapolated), source])

    with open(os.path.join(out_dir, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['i', 'j', 'cx_px', 'cy_px', 'H', 'fov_deg', 'rot_deg',
                    'centroid_dx_px', 'centroid_dy_px', 'extrapolated', 'source'])
        w.writerows(manifest_rows)

    n_ext = sum(r[9] for r in manifest_rows)
    print(f'完成: {len(manifest_rows)} 个核 ({n_ext} 个外推), manifest.csv 已写入')


if __name__ == '__main__':
    main()

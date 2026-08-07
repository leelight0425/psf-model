"""
Python replacement for sfrmat5 (ISO 12233 slanted-edge SFR measurement)
and user_sfrmat5_rgb.m workflow.

Usage:
    python sfrmat5_py.py                                    # default paths
    python sfrmat5_py.py <crop_dir> <mat_save_dir>          # custom paths

This script replaces the MATLAB-based step 2 (user_sfrmat5_rgb.m) in the
PSF-Estimation pipeline. It reads edge patch TIFF images, computes SFR/MTF
curves using the ISO 12233 slanted-edge method, and saves .mat files
compatible with generate_fov_weight.py.

All algorithms are ported line-by-line from the reference sfrmat5 MATLAB code
(Peter Burns, ISO 12233 4th edition) to ensure output compatibility.
"""

import os
import re
import sys
import numpy as np
from scipy.io import savemat
import cv2


# ==============================================================================
# Core sfrmat5 algorithm — exact MATLAB port
# ==============================================================================

def _tukey2(n, alpha, mid=None):
    """
    Asymmetrical Tukey Tapered Cosine Window.
    Exact port of MATLAB tukey2.m by Peter Burns.

    Args:
        n: window length
        alpha: window shape parameter (default 1 = Hann)
        mid: center of window [1:n], default n/2

    Returns:
        w: n-point column vector
    """
    if n < 3:
        return np.ones(n)

    if mid is None:
        mid = n / 2.0

    m1 = n / 2.0
    m2 = mid
    m3 = n - mid
    mm = max(m2, m3)
    n2 = int(round(2 * mm))

    # Build symmetric Tukey window of length n2
    M = (n2 - 1) / 2.0
    w_sym = np.zeros(n2)
    for k in range(int(M) + 1):
        if k <= alpha * M:
            w_sym[k] = 0.5 * (1.0 + np.cos(np.pi * (k / (alpha * M) - 1.0)))
        else:
            w_sym[k] = 1.0
        w_sym[n2 - 1 - k] = w_sym[k]

    # Crop asymmetrically
    if mid >= m1:
        w = w_sym[:n]
    else:
        w = w_sym[-n:] if n2 >= n else w_sym

    return w


def _deriv1_2d(a, fil):
    """
    1-D derivative filter applied along columns (axis=1) for each row.
    Exact port of MATLAB deriv1.m: uses conv(..., 'same') with edge fixup.

    Args:
        a: (nlin, npix) array
        fil: 1D filter coefficients (e.g. [0.5, -0.5])

    Returns:
        b: (nlin, npix) filtered array
    """
    from scipy.signal import convolve
    nlin, npix = a.shape
    b = np.zeros_like(a)

    for ii in range(nlin):
        temp = convolve(a[ii, :], fil, mode='same')
        b[ii, :] = temp
        # Edge fixup (MATLAB deriv1 does this)
        b[ii, 0] = b[ii, 1]
        b[ii, npix - 1] = b[ii, npix - 2]

    return b


def _deriv1_1d(esf, fil):
    """
    1-D derivative filter on a vector.
    Uses conv(..., 'same') with edge fixup.
    """
    from scipy.signal import convolve
    c = convolve(esf, fil, mode='same')
    # Edge fixup (MATLAB style)
    c[0] = c[1]
    c[-1] = c[-2]
    return c


def _centroid(x):
    """
    Compute centroid of a vector (weighted average of indices).
    Exact port of MATLAB centroid.m by Peter Burns.

    NOTE: Uses raw values, NOT absolute values.
    Returns 1-indexed position (matching MATLAB).

    Args:
        x: 1D vector

    Returns:
        loc: centroid position (1-indexed, same as MATLAB)
    """
    total = np.sum(x)
    if abs(total) < 1e-15:
        return len(x) / 2.0 + 0.5
    indices = np.arange(1, len(x) + 1, dtype=np.float64)
    loc = np.sum(indices * x) / total
    return loc  # 1-indexed, caller subtracts 0.5


def _findedge2(cent, nlin, npol):
    """
    Fits polynomial to edge location data.
    Exact port of MATLAB findedge2.m by Peter Burns.

    Uses centered and scaled x values for better fitting (MATLAB polyfit with 3 outputs),
    then unscales the coefficients using polyfit_convert.

    Args:
        cent: array of centroid values (edge positions per line)
        nlin: length of cent
        npol: polynomial order

    Returns:
        p: polynomial coefficients [p_npol, ..., p_1, p_0] (highest degree first)
    """
    index = np.arange(nlin, dtype=np.float64)
    y = cent.astype(np.float64)

    # Center and scale x for better numerical conditioning (MATLAB polyfit with mu)
    mu_x = np.mean(index)
    sigma_x = np.std(index)
    if sigma_x < 1e-10:
        sigma_x = 1.0
    x_scaled = (index - mu_x) / sigma_x

    # Fit on scaled x
    p_scaled = np.polyfit(x_scaled, y, npol)

    # Unscale: polyfit_convert equivalent
    # retval(n+1-j) += p2(n+1-i) * nchoosek(i, j) * (-m)^(i-j) / s^i
    m = mu_x
    s = sigma_x
    p = np.zeros(npol + 1)
    from math import comb
    for i in range(npol + 1):
        for j in range(i + 1):
            p[npol - j] += p_scaled[npol - i] * comb(i, j) * ((-m) ** (i - j)) / (s ** i)

    return p


def _fir2fix(n, m):
    """
    Correction for MTF of derivative (difference) filter.
    Exact port of MATLAB fir2fix.m by Peter Burns.

    Args:
        n: frequency data length [0 to half-sampling]
        m: length of difference filter (2 for 2-point, 3 for 3-point)

    Returns:
        correct: n×1 MTF correction array (limited to max 10)
    """
    correct = np.ones(n)
    m = m - 1  # MATLAB does this
    scale = 1.0

    for i in range(2, n + 1):  # MATLAB: for i = 2:n
        val = abs((np.pi * i * m / (2.0 * (n + 1))) /
                   np.sin(np.pi * i * m / (2.0 * (n + 1))))
        correct[i - 1] = 1.0 + scale * (val - 1.0)  # MATLAB 1-indexed
        if correct[i - 1] > 10.0:
            correct[i - 1] = 10.0

    return correct


def _project2(bb, fitme, fac=4):
    """
    Project and bin data along the edge direction.
    Exact port of MATLAB project2.m by Peter Burns.

    Args:
        bb: (nlin, npix) input data array
        fitme: polynomial coefficients for edge: x = f(y)
        fac: oversampling (binning) factor, default 4

    Returns:
        point: (nn,) oversampled ESF vector
    """
    nlin, npix = bb.shape
    slope = fitme[-2]  # linear coefficient

    nn = int(np.floor(npix * fac))

    slope_inv = 1.0 / slope if abs(slope) > 1e-15 else float('inf')
    offset = round(fac * (0 - (nlin - 1) / slope_inv))

    dell = abs(offset)
    if offset > 0:
        offset = 0

    bwidth = int(nn + dell + 150)
    barray = np.zeros((2, bwidth))

    # Precompute relative edge positions per row
    p2 = np.zeros(nlin)
    for m in range(nlin):
        y = m  # 0-indexed like in project2 (y = m-1 in MATLAB)
        p2[m] = np.polyval(fitme, y) - fitme[-1]  # fitme - fitme(end)

    # Projection and binning (MATLAB loops: n then m)
    for n in range(npix):
        x = n  # 0-indexed (x = n-1 in MATLAB)
        for m in range(nlin):
            ling = int(np.ceil((x - p2[m]) * fac)) + 1 - offset
            # MATLAB 1-indexed; convert to 0-indexed
            ling0 = ling - 1
            if ling0 < 0:
                ling0 = 0
            elif ling0 >= bwidth:
                ling0 = bwidth - 1
            barray[0, ling0] += 1.0   # count
            barray[1, ling0] += bb[m, n]  # sum

    point = np.zeros(nn)
    start = int(1 + round(0.5 * dell))  # MATLAB 1-indexed
    start0 = start - 1  # convert to 0-indexed

    # Check for zero counts and fix
    for i in range(start0, start0 + nn):
        ii = i  # working with 0-indexed
        if barray[0, ii] == 0:
            if ii == 0:
                barray[0, ii] = barray[0, ii + 1]
                barray[1, ii] = barray[1, ii + 1]
            elif ii == start0 + nn - 1:
                barray[0, ii] = barray[0, ii - 1]
                barray[1, ii] = barray[1, ii - 1]
            else:
                barray[0, ii] = (barray[0, ii - 1] + barray[0, ii + 1]) / 2.0
                barray[1, ii] = (barray[1, ii - 1] + barray[1, ii + 1]) / 2.0

    for i in range(nn):
        point[i] = barray[1, start0 + i] / barray[0, start0 + i]

    return point


def _rotatev2(a):
    """
    Rotate edge array so edge is vertical.
    Exact port of MATLAB rotatev2.m.

    Returns:
        a_out: rotated array
        nlin, npix: dimensions after rotation
        rflag: 1 if rotation was performed, 0 otherwise
    """
    nlin, npix = a.shape[0], a.shape[1]

    if a.ndim >= 3:
        mm = 1  # second channel (0-indexed: index 1 = green)
    else:
        mm = 0

    nn = 3
    testv = abs(np.mean(a[-nn:, :, mm]) - np.mean(a[:nn, :, mm]))
    testh = abs(np.mean(a[:, -nn:, mm]) - np.mean(a[:, :nn, mm]))

    rflag = 0
    if testv > testh:
        rflag = 1
        # 90-degree CCW rotation (MATLAB rotate90)
        if a.ndim == 3:
            nc = a.shape[2]
            a_rot = np.zeros((npix, nlin, nc), dtype=a.dtype)
            for c in range(nc):
                temp = a[:, :, c].T
                a_rot[:, :, c] = temp[::-1, :]
            a_out = a_rot
        else:
            temp = a.T
            a_out = temp[::-1, :]
        nlin, npix = npix, nlin
    else:
        a_out = a

    return a_out, nlin, npix, rflag


def sfrmat5_rgb(image, npol=5, wflag=0, weight=None, alpha=1.0):
    """
    Python port of sfrmat5 for RGB images (non-GUI mode, io=1, del=1).

    Exact port of sfrmat5.m by Peter Burns (ISO 12233 4th edition).

    Args:
        image: (H, W, 3) uint8 image (any channel order — processed independently)
        npol: polynomial order for edge fitting (1-5, default 5)
        wflag: window type (0=Tukey, 1=Hamming)
        weight: RGB→Luminance weights (default: [0.213, 0.715, 0.072])
        alpha: Tukey window alpha parameter (default 1.0)

    Returns:
        sfr: (nn2out, 5) array [[freq, sfr_R, sfr_G, sfr_B, sfr_lum], ...]
        esf_all: (nn, 4) supersampled ESF [R, G, B, Lum]
        offset: (2,) array [R-G, B-G] colour misregistration from ESF
        rot: float, edge rotation angle in degrees (rot = angle + 90, where
            angle = degrees(arctan(vslope)) is the edge angle from vertical)
    """
    if weight is None:
        weight = np.array([0.213, 0.715, 0.072])

    a = image.astype(np.float64)
    nlin, npix, ncol_orig = a.shape

    # ---- Form luminance record (MATLAB sfrmat5 lines 276-287) ----
    if ncol_orig == 3:
        lum = weight[0] * a[:, :, 0] + weight[1] * a[:, :, 1] + weight[2] * a[:, :, 2]
        a = np.dstack([a[:, :, 0], a[:, :, 1], a[:, :, 2], lum])
        ncol = 4
    else:
        ncol = ncol_orig

    # ---- Rotate horizontal edge to vertical (MATLAB sfrmat5 lines 290-291) ----
    a, nlin, npix, rflag = _rotatev2(a)

    # ---- Determine edge polarity (MATLAB sfrmat5 lines 294-310) ----
    tleft = np.sum(a[:, :5, 0])
    tright = np.sum(a[:, npix - 5:npix, 0])
    lowhi = 1 if tleft > tright else 0

    if lowhi:
        fil1 = np.array([-0.5, 0.5])
        fil2 = np.array([-0.5, 0.0, 0.5])
    else:
        fil1 = np.array([0.5, -0.5])
        fil2 = np.array([0.5, 0.0, -0.5])

    # ---- Smoothing window for first-pass edge detection (MATLAB lines 318-325) ----
    if wflag != 0:
        win1 = np.hamming(npix)
    else:
        win1 = _tukey2(npix, alpha)
        win1 = 0.95 * win1 + 0.05

    fitme = np.zeros((ncol, npol + 1))
    fitme1 = np.zeros((ncol, 2))
    loc = np.zeros((ncol, nlin))

    # ---- Process each color channel (MATLAB lines 327-451) ----
    for color in range(ncol):
        ch_data = a[:, :, color]

        # Step 3: 1-D derivative (MATLAB line 361)
        c_deriv = _deriv1_2d(ch_data, fil1)

        # Step 4: First-pass centroid with symmetric window (MATLAB lines 366-368)
        for n in range(nlin):
            loc[color, n] = _centroid(c_deriv[n, :] * win1)

        # Subtract 0.5 for FIR phase shift
        loc[color, :] = loc[color, :] - 0.5

        # First polynomial fit (MATLAB line 369)
        fitme[color, :] = _findedge2(loc[color, :], nlin, npol)

        # Second pass: adaptive window at edge estimate (MATLAB lines 371-386)
        for n in range(nlin):
            place = np.polyval(fitme[color, :], n)
            if wflag != 0:
                win2 = np.hamming(npix)
            else:
                win2 = _tukey2(npix, alpha, mid=place)
                win2 = 0.95 * win2 + 0.05
            loc[color, n] = _centroid(c_deriv[n, :] * win2)

        loc[color, :] = loc[color, :] - 0.5

        # Step 5: Final polynomial fit (MATLAB lines 389-393)
        fitme[color, :] = _findedge2(loc[color, :], nlin, npol)
        fitme1[color, :] = _findedge2(loc[color, :], nlin, 1)

    # ---- Edge angle & sampling correction (MATLAB lines 496-526) ----
    vslope = -fitme1[-1, -2]  # linear slope from last channel (luminance)

    # Edge angle from vertical (degrees); rot differs from angle by 90°.
    # Used as the rotation label in the training dataset instead of the
    # filename-parsed value. Normalized to [0, 180).
    angle = np.degrees(np.arctan(vslope))
    rot = (angle + 90.0) % 180.0

    # Adjust for valid lines per ISO 12233
    if abs(fitme1[-1, -2]) > 1e-10:
        nlin1 = int(np.floor(nlin * abs(fitme1[-1, -2])) / abs(fitme1[-1, -2]))
    else:
        nlin1 = nlin
    a = a[:nlin1, :, :]
    nlin = nlin1

    # Sampling correction
    delfac = np.cos(np.arctan(vslope))

    # ---- SFR computation per channel (MATLAB lines 527-609) ----
    nbin = 4
    nn = int(np.floor(npix * nbin))  # NOTE: floor, not ceil!
    nn2 = nn // 2 + 1
    mtf = np.zeros((nn2, ncol))

    dcorr = _fir2fix(nn2, 3)  # m=3 for 3-point derivative filter [0.5, 0, -0.5]

    freqlim = 1 if nbin > 1 else 2
    nn2out = int(np.floor(nn2 * freqlim / 2.0 + 0.5))  # MATLAB round (not Python banker's round)

    del_eff = delfac  # sampling pitch * cos(edge_angle)
    del2_val = del_eff / nbin

    esf_all = np.zeros((nn, ncol))

    for color in range(ncol):
        # Step 6: Project & bin → supersampled ESF (MATLAB line 556)
        esf = _project2(a[:, :, color], fitme[color, :], nbin)
        esf_all[:, color] = esf

        # Step 7: Derivative of ESF → LSF (MATLAB line 563)
        lsf = _deriv1_1d(esf, fil2)

        # Handle zero endpoints (MATLAB lines 567-571: elseif, not if)
        if lsf[0] == 0:
            lsf[0] = lsf[1] if len(lsf) > 1 else 0
        elif lsf[-1] == 0:
            lsf[-1] = lsf[-2] if len(lsf) > 1 else 0

        # MATLAB lines 574-581:
        # mm = find(c==max(c)); mm = mean(mm);
        # Shift array so it is centered. Not necessary, since we retain only
        # modulus of the DFT in step 9 (comment next 2 lines to omit).
        # mm = nn/2;
        mm = nn / 2.0

        # Step 8: Apply window (MATLAB lines 582-589)
        if wflag != 0:
            win = np.hamming(nn)
        else:
            win = _tukey2(nn, alpha, mid=mm)

        lsf_windowed = win * lsf

        # Step 9: FFT → normalize → MTF (MATLAB lines 602-603)
        temp = np.abs(np.fft.fft(lsf_windowed, nn))
        mtf[:, color] = temp[:nn2] / (temp[0] + 1e-15)

        # Step 10: FIR derivative correction (MATLAB line 606)
        mtf[:, color] = mtf[:, color] * dcorr

    # ---- Build output SFR matrix (MATLAB lines 617-625) ----
    freq = np.zeros(nn)
    for n in range(nn):
        freq[n] = (n) / (del2_val * nn)  # MATLAB: (n-1)/(del2*nn), n from 1

    sfr = np.zeros((nn2out, ncol + 1))
    for i in range(nn2out):
        sfr[i, 0] = freq[i]
        sfr[i, 1:] = mtf[i, :ncol]

    # ---- Compute colour misregistration offset from ESF ----
    # MATLAB user_sfrmat5_rgb.m lines 41-67
    trim_start = 30
    trim_end = nn - 30
    if trim_end > trim_start:
        s_esf = esf_all[trim_start:trim_end, :3]  # R, G, B only
    else:
        s_esf = esf_all[:, :3]

    n_esf = s_esf.shape[0]
    nor_esf = np.zeros_like(s_esf)
    for idx in range(3):
        vmin = np.min(s_esf[:, idx])
        vmax = np.max(s_esf[:, idx])
        if vmax > vmin:
            nor_esf[:, idx] = (s_esf[:, idx] - vmin) / (vmax - vmin)
        else:
            nor_esf[:, idx] = 0.0

    offset = np.array([
        (np.sum(nor_esf[:, 0]) - np.sum(nor_esf[:, 1])) / 4.0,  # R - G
        (np.sum(nor_esf[:, 2]) - np.sum(nor_esf[:, 1])) / 4.0,  # B - G
    ])

    return sfr, esf_all, offset, rot


# ==============================================================================
# Batch processing (replaces user_sfrmat5_rgb.m)
# ==============================================================================

def process_folder(dir_path, save_path, npol=5, wflag=0):
    """
    Process all TIFF edge patches in a folder.
    Equivalent to user_sfrmat5_rgb.m batch loop.

    Args:
        dir_path: path to folder with .tif edge patch images
        save_path: path to save .mat output files
        npol: polynomial order (default 5)
        wflag: window flag (0=Tukey, 1=Hamming)
    """
    os.makedirs(save_path, exist_ok=True)

    file_list = sorted([f for f in os.listdir(dir_path) if f.endswith('.tif')])

    if not file_list:
        print(f"ERROR: No .tif files found in {dir_path}")
        return

    print(f"Found {len(file_list)} TIFF files to process")
    print(f"  Input:  {dir_path}")
    print(f"  Output: {save_path}")
    print(f"  npol={npol}, wflag={wflag} (window={'Tukey' if wflag==0 else 'Hamming'})")
    print()

    pattern = r'fov(-?\d+\.\d+)_angle(-?\d+\.\d+)'
    n_saved, n_skipped, n_errors = 0, 0, 0

    for filename in file_list:
        filepath = os.path.join(dir_path, filename)

        match = re.search(pattern, filename)
        if not match:
            print(f"SKIP: {filename} — cannot parse fov/rot from filename")
            n_skipped += 1
            continue

        fov = float(match.group(1))
        rot_gt = float(match.group(2))  # ground-truth rotation from filename (print only)

        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"SKIP: {filename} — cannot read file")
            n_skipped += 1
            continue

        # Ensure 3-channel; convert BGR→RGB to match MATLAB imread channel order
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print(f"Processing: {filename}  (fov={fov:.2f}°, rot_gt={rot_gt:.2f}°) ...", end=" ")

        try:
            sfr, esf, offset, rot = sfrmat5_rgb(image, npol=npol, wflag=wflag)

            # Validity check matching MATLAB: max(sfr(:,2))==1 && no NaN
            sfr_ok = (np.max(sfr[:, 1]) >= 0.99 and
                      not np.any(np.isnan(sfr)))

            if sfr_ok:
                save_filename = filename.replace('.tif', '.mat')
                savemat(os.path.join(save_path, save_filename), {
                    'fov': np.array([[fov]]),
                    'rot': np.array([[rot]]),
                    'sfr': sfr,
                    'offset': offset.reshape(1, 2),
                })
                print(f"OK (rot_meas={rot:.2f}°)")
                n_saved += 1
            else:
                print(f"SKIP (invalid SFR)")
                n_skipped += 1

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            n_errors += 1

    print(f"\nDone. Saved: {n_saved}, Skipped: {n_skipped}, Errors: {n_errors}")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    dir_path = r'.\dataset\63762BB\crop\shot0.00'
    save_path = r'.\dataset\63762BB\mat\shot0.00'

    if len(sys.argv) >= 3:
        dir_path = sys.argv[1]
        save_path = sys.argv[2]

    dir_path = os.path.abspath(dir_path)
    save_path = os.path.abspath(save_path)

    process_folder(dir_path, save_path, npol=5, wflag=0)

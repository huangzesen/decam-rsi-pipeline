"""Render the stamps from cutouts_remote.py: per candidate a contact sheet (PNG) and a blink animation (GIF).

Sheet: one pair per detection, in time order: top = the detection (centred, ticks mark the centre), bottom = the same
RA/Dec in another night's exposure (a static source would show there too). Labels: UT date, band, exposure time, S/N.
GIF: the detection stamps in time order, centred on the body (the body stays put, the sky moves).
Usage: python -I harness/cutouts_render.py runs/r1/observer/batch01/cutouts/batch01
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from PIL import Image, ImageDraw, ImageFont

base = sys.argv[1]
S = np.load(base + ".npz")["stamps"]; M = json.load(open(base + ".json"))
out = os.path.dirname(base)
Z = ZScaleInterval(contrast=0.15)
PIX = 0.263


def norm(s):
    s = s.copy(); good = np.isfinite(s)
    if not good.any():
        return np.zeros_like(s)
    s[~good] = np.nanmedian(s)
    lo, hi = Z.get_limits(s[good])
    return np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1)


def orient(st, cd):
    """Turn a CCD stamp to north up, east left (origin lower), from its CD matrix."""
    if not cd:
        return st
    M = np.linalg.inv(np.array(cd, float))           # sky (dRA*cos, dDec) -> pixel (dx, dy)
    n, e = M @ [0, 1], M @ [1, 0]
    if abs(n[0]) > abs(n[1]):                       # north runs along x: swap axes
        st = st.T; n, e = n[::-1], e[::-1]
    if n[1] < 0:
        st = st[::-1, :]; e = np.array([e[0], -e[1]])
    if e[0] > 0:
        st = st[:, ::-1]
    return st


def ticks(ax, n):
    c = n // 2; g, L = 5, 8
    for (x0, x1, y0, y1) in ((c - g - L, c - g, c, c), (c + g, c + g + L, c, c), (c, c, c - g - L, c - g), (c, c, c + g, c + g + L)):
        ax.plot([x0, x1], [y0, y1], color="#ff5a36", lw=1.2)


def ut(mjd):
    return Time(mjd, format="mjd").iso[:16]


for lab, rows in M.items():
    n = len(rows); ncol = min(7, n); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(2 * nrow, ncol, figsize=(1.95 * ncol, 2.45 * 2 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, m in enumerate(rows):
        r, c = 2 * (k // ncol), k % ncol
        a = axes[r][c]
        if m["stamp"] is not None:
            st = orient(S[m["stamp"]], m.get("cd")); a.imshow(norm(st), cmap="gray", origin="lower", interpolation="nearest"); ticks(a, st.shape[0])
        a.set_title(f"{ut(m['mjd'])}\n{m['band']} {m['exptime']:.0f}s  S/N {m['snr']:.1f}", fontsize=7.5)
        b = axes[r + 1][c]
        if m["ref"]:
            st = orient(S[m["ref"]["stamp"]], m["ref"].get("cd")); b.imshow(norm(st), cmap="gray", origin="lower", interpolation="nearest"); ticks(b, st.shape[0])
            b.set_title(f"same sky {ut(m['ref']['mjd'])[:10]}\n{m['ref']['band']}", fontsize=7, color="#555555")
    fig.suptitle(f"{lab}: {n} DECam detections (top) and the same sky on another night (bottom). "
                 f"Stamps {S.shape[1] * PIX:.0f}\" x {S.shape[1] * PIX:.0f}\", north up, east left; ticks mark the centre.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.6)
    fig.savefig(os.path.join(out, f"{lab}_stamps.png"), dpi=130); plt.close(fig)

    frames = []
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for m in rows:
        if m["stamp"] is None:
            continue
        st = orient(S[m["stamp"]], m.get("cd")); im = (norm(st)[::-1] * 255).astype(np.uint8)
        img = Image.fromarray(im).resize((st.shape[1] * 3, st.shape[0] * 3), Image.NEAREST).convert("RGB")
        d = ImageDraw.Draw(img); w = img.size[0]; c = w // 2; g, L = 15, 22
        for (x0, y0, x1, y1) in ((c - g - L, c, c - g, c), (c + g, c, c + g + L, c), (c, c - g - L, c, c - g), (c, c + g, c, c + g + L)):
            d.line((x0, y0, x1, y1), fill=(255, 90, 54), width=2)
        d.rectangle((0, 0, w, 20), fill=(0, 0, 0))
        d.text((5, 3), f"{lab}  {ut(m['mjd'])} UT  {m['band']} {m['exptime']:.0f}s  S/N {m['snr']:.1f}", fill=(255, 255, 255), font=font)
        frames.append(img)
    if frames:
        frames[0].save(os.path.join(out, f"{lab}_track.gif"), save_all=True, append_images=frames[1:], duration=900, loop=0)
    print(lab, n, "detections ->", f"{lab}_stamps.png", f"{lab}_track.gif")

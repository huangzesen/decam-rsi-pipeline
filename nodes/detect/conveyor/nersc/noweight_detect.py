#!/usr/bin/env python3
"""Detection for exposures whose weight (oow) and/or mask (ood) file cannot be had from the archive (2026-11-23, user:
"Process without weights": u19's 63 exposures, every archive copy cut short or failing its checksum). For each stem with an
ooi image: a synthetic weight per CCD = 1 / (1.4826 * MAD of the image)^2 (0 on non-finite pixels) and a synthetic mask of
zeros, only where the real file is missing; they are written to STAGE/<YYMMDD>/ next to links to the real files, then
archive10/detect_pad.py's process() runs on STAGE with its normal output to the share's det/. Stems done this way are listed
in OUT/NOWEIGHT.txt. Usage: python -I noweight_detect.py IMG_DIR OUT_DIR STAGE STEMS_FILE [--workers N]"""
import glob
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "$NERSC_SCRATCH/archive10")
import detect_pad as DP  # noqa: E402


def stage(stem, img, st):
    from astropy.io import fits
    warnings.simplefilter("ignore")
    d = os.path.join(img, stem[4:10]); o = os.path.join(st, stem[4:10]); os.makedirs(o, exist_ok=True)
    fi = sorted(glob.glob(os.path.join(d, stem + "_ooi_*.fits.fz")))
    if not fi:
        return "no image"
    base = os.path.basename(fi[0]); made = []
    for f in [fi[0], fi[0].replace("_ooi_", "_oow_"), fi[0].replace("_ooi_", "_ood_")]:
        t = os.path.join(o, os.path.basename(f))
        if os.path.exists(f) and not os.path.lexists(t):
            os.symlink(os.path.abspath(f), t)
    w_out = os.path.join(o, base.replace("_ooi_", "_oow_")); m_out = os.path.join(o, base.replace("_ooi_", "_ood_"))
    need_w, need_m = not os.path.exists(w_out), not os.path.exists(m_out)
    if need_w or need_m:
        with fits.open(fi[0], memmap=False) as hi:
            W = [fits.PrimaryHDU()]; M = [fits.PrimaryHDU()]
            for h in hi[1:]:
                try:
                    x = h.data
                except Exception:
                    continue
                if x is None:
                    continue
                x = x.astype(np.float32); ok = np.isfinite(x)
                med = float(np.median(x[ok])) if ok.any() else 0.0
                mad = float(np.median(np.abs(x[ok] - med))) if ok.any() else 0.0
                iv = 1.0 / (1.4826 * mad) ** 2 if mad > 0 else 0.0
                name = h.header.get("EXTNAME")
                W.append(fits.ImageHDU(np.where(ok, iv, 0).astype(np.float32), name=name))
                M.append(fits.ImageHDU(np.zeros(x.shape, np.int16), name=name))
        for need, hl, out, what in ((need_w, W, w_out, "weight"), (need_m, M, m_out, "mask")):
            if need:
                tmp = out[:-3] + ".tmp"
                fits.HDUList(hl).writeto(tmp, overwrite=True); os.replace(tmp, out); made.append(what)
    return "staged" + (" with synthetic " + "+".join(made) if made else "")


def run(args):
    stem, img, out, st = args
    try:
        msg = stage(stem, img, st)
    except Exception as e:
        return stem, "stage ERROR " + repr(e)[:200]
    if msg == "no image":
        return stem, msg
    try:
        _, m2 = DP.process(stem, st, out, 4.5)
    except Exception as e:
        m2 = "detect ERROR " + repr(e)[:200]
    return stem, msg + "; " + m2


if __name__ == "__main__":
    img, out, st, sf = sys.argv[1:5]
    nw = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 32
    stems = [s for s in open(sf).read().replace(",", " ").split() if s]
    with ProcessPoolExecutor(nw) as ex, open(os.path.join(out, "NOWEIGHT.txt"), "a") as L:
        for s, msg in ex.map(run, [(s, img, out, st) for s in stems]):
            print(s, msg, flush=True); L.write(f"{s}\t{msg}\n"); L.flush()

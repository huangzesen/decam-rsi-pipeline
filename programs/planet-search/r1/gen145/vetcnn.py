"""gen 143: a small self-trained CNN that scores a candidate from per-night image stamps along its orbit.

Input per candidate: up to NMAX night stamps (S x S px, S = 15, ~3.9"), each the mean of that night's per-exposure
stamps centred on the predicted position, every stamp in units of its own pixel noise (background = median of the
cutout outside 2 FWHM, noise = 1.4826 MAD there), masked pixels 0, then x sqrt(n_exp) so each night stamp has unit
pixel noise.  Night stamps are a SET (order-free): a shared conv encoder per night, then mean / max / second-max /
min pooling over nights, then an MLP -> one logit ("this is a real point source moving along this track").

Training (work/train_vetcnn.py, offline in the session): real DECam stamps at random sky positions of the
development exposures as backgrounds (they contain real stars, galaxies, artifacts and noise), positives =
Gaussian-PSF bodies injected into those stamps with per-night S/N drawn from faint multi-night and one-bright-night
regimes, negatives = backgrounds biased upwards the way a hypothesis search biases noise (re-centred on the best
local PSF-filtered pixel).  The weights ship as vetcnn.pt.
"""
import math, os
import numpy as np

S = 15
H = S // 2
NMAX = 12
HERE = os.path.dirname(os.path.abspath(__file__))


def norm_stamp(im, wt, mk, fw, cx, cy, half=H):
    """im/wt/mk: cutout arrays (2RC+1 square) with the target at pixel (cx, cy) (integers, centre of cutout ideally).
    Returns S x S float32 stamp in noise units, or None."""
    good = (mk == 0) & (wt > 0) & np.isfinite(im)
    ny, nx = im.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    ann = good & (rr2 > (2.0 * fw) ** 2)
    if ann.sum() < 30:
        ann = good & (rr2 > (1.2 * fw) ** 2)
    if ann.sum() < 20:
        return None
    v = im[ann]
    b = float(np.median(v))
    sd = float(1.4826 * np.median(np.abs(v - b)))
    if not (np.isfinite(sd) and sd > 0):
        return None
    z = np.where(good, (im - b) / sd, 0.0)
    if cy - half < 0 or cx - half < 0 or cy + half + 1 > ny or cx + half + 1 > nx:
        return None
    z = z[cy - half:cy + half + 1, cx - half:cx + half + 1]
    return np.clip(z, -10, 30).astype(np.float32)


_NET = None


def _build():
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, c=24, e=48):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(1, c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(c, 2 * c, 3, padding=1, stride=2), nn.ReLU(),   # 15 -> 8
                nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU())
            self.fc = nn.Sequential(nn.Linear(2 * c * 2 + 2, e), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(4 * e + 1, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x, m):
            # x (B, N, S, S), m (B, N) 1 = night present
            B, N = x.shape[:2]
            h = self.enc(x.reshape(B * N, 1, S, S))                    # (BN, 2c, 8, 8)
            cen = h[:, :, 3:5, 3:5].mean((2, 3))                         # centre
            glb = h.mean((2, 3))
            xs = x.reshape(B * N, S, S)
            c3 = xs[:, H - 1:H + 2, H - 1:H + 2].mean((1, 2), keepdim=False)[:, None]
            tot = xs.mean((1, 2))[:, None]
            f = self.fc(torch.cat([cen, glb, c3, tot], 1)).reshape(B, N, -1)
            mm = m[:, :, None]
            big = 1e4
            fmean = (f * mm).sum(1) / mm.sum(1).clamp(min=1)
            fmax = (f - big * (1 - mm)).max(1).values
            fmin = (f + big * (1 - mm)).min(1).values
            # second-largest per feature (leave-brightest-out)
            fs = (f - big * (1 - mm)).sort(1, descending=True).values
            f2 = torch.where(mm.sum(1) >= 2, fs[:, 1], fmean)
            nn_ = mm.sum(1) / NMAX
            return self.head(torch.cat([fmean, fmax, f2, fmin, nn_], 1)).squeeze(1)
    return Net


def load(path=None):
    global _NET
    if _NET is not None:
        return _NET
    import torch
    path = path or os.path.join(HERE, "vetcnn.pt")
    if not os.path.exists(path):
        return None
    net = _build()()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    _NET = net
    return net


def score(night_stamps):
    """night_stamps: list (per candidate) of list of S x S arrays (one per night, possibly empty).
    Returns np.array of logits (nan where < 2 nights)."""
    net = load()
    C = len(night_stamps)
    out = np.full(C, np.nan, np.float32)
    if net is None or C == 0:
        return out
    import torch
    X = np.zeros((C, NMAX, S, S), np.float32)
    M = np.zeros((C, NMAX), np.float32)
    for i, ns in enumerate(night_stamps):
        ns = [a for a in ns if a is not None][:NMAX]
        for j, a in enumerate(ns):
            X[i, j] = a; M[i, j] = 1
    ok = M.sum(1) >= 2
    if not ok.any():
        return out
    with torch.no_grad():
        lg = net(torch.from_numpy(X[ok]), torch.from_numpy(M[ok])).numpy()
    out[ok] = lg
    return out

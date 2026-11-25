import numpy as np, torch, math
F = torch.nn.functional
torch.manual_seed(1); rng = np.random.default_rng(1)
def gf(x, sig):
    rad = int(3.0 * sig + 0.5)
    k = torch.exp(-0.5 * (torch.arange(-rad, rad + 1, dtype=torch.float32) / sig) ** 2); k = k / k.sum()
    x = F.conv2d(x[None, None], k.view(1, 1, 1, -1), padding=(0, rad))
    return F.conv2d(x, k.view(1, 1, -1, 1), padding=(rad, 0))[0, 0]
def cr(u):   # Catmull-Rom weights for offsets -1,0,1,2
    return [(-u**3 + 2*u**2 - u) / 2, (3*u**3 - 5*u**2 + 2) / 2, (-3*u**3 + 4*u**2 + u) / 2, (u**3 - u**2) / 2]
def bil(m, fx, fy):
    ix = np.floor(fx).astype(int); iy = np.floor(fy).astype(int); ax = fx - ix; ay = fy - iy
    return (m[iy, ix] * (1 - ax) + m[iy, ix + 1] * ax) * (1 - ay) + (m[iy + 1, ix] * (1 - ax) + m[iy + 1, ix + 1] * ax) * ay
def bic(m, fx, fy):
    ix = np.floor(fx).astype(int); iy = np.floor(fy).astype(int); wx = cr(fx - ix); wy = cr(fy - iy)
    out = 0
    for a in range(4):
        for b in range(4):
            out = out + wy[a] * wx[b] * m[iy + a - 1, ix + b - 1]
    return out
N = 1600
for fwhm in (0.85, 1.0, 1.2, 1.5):
    s = fwhm / 0.263 / 2.3548
    norm = 1.0 / (4 * math.pi * s * s)
    # sources on a grid at random sub-pixel phases, amplitude -> S/N 1 per unit flux (noise-free), plus separate noise map
    img = torch.zeros((N, N)); pos = []
    yy, xx = np.mgrid[-12:13, -12:13]
    for cy in range(40, N - 40, 40):
        for cx in range(40, N - 40, 40):
            py = cy + rng.uniform(0, 2); px = cx + rng.uniform(0, 2)
            iy, ix = int(round(py)), int(round(px))
            g = np.exp(-0.5 * ((yy + iy - py) ** 2 + (xx + ix - px) ** 2) / s ** 2) / (2 * math.pi * s * s)
            img[iy - 12:iy + 13, ix - 12:ix + 13] += torch.from_numpy(g.astype(np.float32)); pos.append((py, px))
    pos = np.array(pos)
    snr_s = (gf(img, s) / math.sqrt(norm)).numpy()        # per unit flux: peak = sqrt(norm) -> normalise to 1
    snr_s /= math.sqrt(norm)
    snr_n = (gf(torch.randn((N, N)), s) / math.sqrt(norm)).numpy()
    # lattice = every 2 native pixels, made by bilinear sampling at lattice nodes (nodes at native half-pixel offsets, generic phase)
    ly, lx = np.mgrid[0:(N - 8) // 2, 0:(N - 8) // 2]
    off = 0.37
    def tolat(m): return bil(m, lx * 2.0 + 2 + off, ly * 2.0 + 2 + off)
    Ls, Ln = tolat(snr_s), tolat(snr_n)
    fy = (pos[:, 0] - 2 - off) / 2; fx = (pos[:, 1] - 2 - off) / 2
    node = Ls[np.rint(fy).astype(int), np.rint(fx).astype(int)]
    pool = np.max([Ls[np.rint(fy).astype(int) + a, np.rint(fx).astype(int) + b] for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0)
    rx = rng.uniform(5, Ln.shape[1] - 6, 200000); ry = rng.uniform(5, Ln.shape[0] - 6, 200000)
    nb, nc = bil(Ln, rx, ry), bic(Ln, rx, ry)
    print(f"seeing {fwhm}: signal at true pos  bilinear {bil(Ls, fx, fy).mean():.3f}  bicubic {bic(Ls, fx, fy).mean():.3f}  nearest node {node.mean():.3f} best node(3x3) {pool.mean():.3f} | "
          f"noise std node {Ln.std():.3f} bilinear {nb.std():.3f} bicubic {nc.std():.3f}")

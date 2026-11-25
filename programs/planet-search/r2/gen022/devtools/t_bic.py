import re, numpy as np, torch, math, textwrap
src = open("$RSI_SANDBOX/r2/gen022/workspace/submission/finder.py").read()
i0 = src.index("    def bic_at(k, x, y):"); i1 = src.index("    # how surprising is a pooled S/N")
j0 = src.index("    def bil_at(k, x, y):"); j1 = src.index("    BIC = not os.environ")
PS = 0.526; f32 = torch.float32
rng = np.random.default_rng(0)
H, W = 300, 400
yy, xx = np.mgrid[0:H, 0:W]
sig = 0.6 / PS
m = np.zeros((H, W)); truth = []
for cy in range(20, H - 20, 20):
    for cx in range(20, W - 20, 20):
        py, px = cy + rng.uniform(0, 1), cx + rng.uniform(0, 1); amp = rng.uniform(3, 9)
        m += amp * np.exp(-0.5 * ((yy - py) ** 2 + (xx - px) ** 2) / sig ** 2); truth.append((py, px, amp))
truth = np.array(truth)
m8 = np.clip(np.rint(m * 10), -127, 127).astype(np.int8)
m8[100:110, 200:230] = -128
raw = [torch.from_numpy(m8)]; rawf = [raw[0].reshape(-1)]; mx0 = [-37]; my0 = [11]; mW = [W]; mH = [H]
g = dict(torch=torch, PS=PS, f32=f32, rawf=rawf, mx0=mx0, my0=my0, mW=mW, mH=mH)
exec(textwrap.dedent(src[i0:i1]), g); exec(textwrap.dedent(src[j0:j1]), g)
x = torch.tensor((truth[:, 1] + mx0[0]) * PS, dtype=f32); y = torch.tensor((truth[:, 0] + my0[0]) * PS, dtype=f32)
vc, okc = g["bic_at"](0, x, y); vb, okb = g["bil_at"](0, x, y)
a = torch.tensor(truth[:, 2], dtype=f32)
print("valid bic/bil:", int(okc.sum()), int(okb.sum()), "of", len(a))
print("ratio to true amplitude: bicubic %.3f  bilinear %.3f" % (float((vc / a)[okc].mean()), float((vb / a)[okb].mean())))
print("max |bic - true|/true:", float(((vc - a).abs() / a)[okc].max()))
# at nodes bicubic == node value
xn = torch.tensor((np.arange(50, 150) + mx0[0]) * PS, dtype=f32); yn = torch.tensor((np.arange(60, 160) + my0[0]) * PS, dtype=f32)
vc, okc = g["bic_at"](0, xn, yn); vb, okb = g["bil_at"](0, xn, yn)
print("at nodes max |bic - bil|:", float((vc - vb)[okc & okb].abs().max()), int(okc.sum()))
# borders / masked
xe = torch.tensor([(-37) * PS, (-37 + W - 1) * PS, (215 - 37) * PS, (-37 + 0.5) * PS, (-37 + W - 2.5) * PS], dtype=f32); ye = torch.tensor([11 * PS, (11 + H - 1) * PS, (105 + 11) * PS, (11 + 0.5) * PS, (11 + H - 2.5) * PS], dtype=f32)
print("edge ok:", g["bic_at"](0, xe, ye)[1].tolist(), g["bil_at"](0, xe, ye)[1].tolist())
x2 = torch.rand(5, 13) * 100 + 10; print("2-d shape:", g["bic_at"](0, x2, x2)[0].shape)

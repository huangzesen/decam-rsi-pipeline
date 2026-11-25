import sys, os, math, importlib.util, numpy as np, torch, time
def load(path, name, env):
    for k, v in env.items(): os.environ[k] = v
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old = load('/app/history/gen022/finder.py', 'fold', {})
var = {'it4 lt1': load('/app/submission/finder.py', 'f1', {'FINDER_KEPIT': '4', 'FINDER_KEPLT': '1'}),
       'it6 lt1': load('/app/submission/finder.py', 'f2', {'FINDER_KEPIT': '6', 'FINDER_KEPLT': '1'}),
       'it5 lt1': load('/app/submission/finder.py', 'f4', {'FINDER_KEPIT': '5', 'FINDER_KEPLT': '1'}),
       'it4 lt0': load('/app/submission/finder.py', 'f3', {'FINDER_KEPIT': '4', 'FINDER_KEPLT': '0'})}
ref = load('/app/submission/finder.py', 'f9', {'FINDER_KEPIT': '30', 'FINDER_KEPLT': '0'})     # converged reference
torch.manual_seed(1); AS = math.pi / 180 / 3600
n, K = 20000, 60
for label, rlo, rhi, vmax in (('27-120 au, any direction up to 1.38 v_circ', 27, 120, 1.38), ('27-60 au, up to 1.12', 27, 60, 1.12)):
    r = torch.exp(torch.rand(n, dtype=torch.float64) * math.log(rhi / rlo)) * rlo; g = 1 / r
    al = (torch.rand(n, dtype=torch.float64) - 0.5) * 0.3; be = (torch.rand(n, dtype=torch.float64) - 0.5) * 0.3
    vc = torch.sqrt(old.MU * g ** 3)
    d = torch.randn(n, 3, dtype=torch.float64); d = d / d.norm(dim=1, keepdim=True) * (torch.rand(n, 1, dtype=torch.float64) ** 0.5) * vmax
    x = torch.stack([al, be, g, d[:, 0] * vc, d[:, 1] * vc, d[:, 2] * vc], 1)
    dt = (torch.rand(n, K, dtype=torch.float64) - 0.5) * 2 * 5000.0
    ph = torch.rand(n, K, dtype=torch.float64) * 2 * math.pi
    ob = torch.stack([torch.cos(ph), torch.sin(ph) * 0.92, torch.sin(ph) * 0.39], -1)
    R = ref.kmodel(x, dt, ob)
    # orbit elements for classing
    r0 = torch.stack([al, be, torch.ones_like(al)], 1) / g[:, None]; v0 = x[:, 3:] / g[:, None]
    r0n = r0.norm(dim=1); inva = 2 / r0n - (v0 * v0).sum(1) / old.MU; a = 1 / inva.clamp(min=1e-9)
    h = torch.linalg.cross(r0, v0).norm(dim=1); ecc = torch.sqrt((1 - h * h / old.MU * inva).clamp(min=0)); q = a * (1 - ecc)
    tno = (q > 20) & (inva > 0)
    print(label, ' share with q > 20 au and bound:', float(tno.double().mean()))
    for nm, m in [('gen022', old)] + list(var.items()):
        dd = (m.kmodel(x, dt, ob) - R).norm(dim=-1) / AS
        print(f"   {nm:8s} vs converged: all orbits max {float(dd.max()):.2e}, share > 1e-3\": {float((dd > 1e-3).double().mean()):.2e};   q > 20 au: max {float(dd[tno].max()):.2e}, share > 1e-3\": {float((dd[tno] > 1e-3).double().mean()):.2e}, > 1e-5\": {float((dd[tno] > 1e-5).double().mean()):.2e}")

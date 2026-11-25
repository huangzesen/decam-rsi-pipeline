"""The OF201 field through the pair proxy: measured raw pairs per hypothesis (gb4 / gb5 runs) against the proxy with
(a) the field's own detection counts (data / fraction of each exposure inside 2.6 deg) and (b) the counts the archive
model predicts for the same exposures. Writes FIELD.json for estimate.py.

Usage: python -I footprint/field.py EXPOSURES.npz FIELD.json [--n1 n1_pred]
"""
import argparse, collections, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import proxy as P
import link_cpu as LC
from estimate import NODES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps"); ap.add_argument("out")
    ap.add_argument("--data", default="data_real/of201_nsc_n1.npz"); ap.add_argument("--n1", default="n1_pred")
    ap.add_argument("--runs", default="runs/vista/gb[45]_*.json")
    a = ap.parse_args()
    tz = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "g_table.npz")); tab = (tz["s"], tz["R"], tz["G"])
    d, meta = LC.load(a.data)
    ra0, dec0 = np.degrees(meta["ra0"]), np.degrees(meta["dec0"])
    W = LC.window(d, meta, 3, 1.5, 2)
    ew = W["exps"]
    ra, dec = np.degrees(d["exp_ra"][ew]), np.degrees(d["exp_dec"][ew])
    night, xe = np.floor(d["exp_t"][ew]).astype(int), d["exp_obs"][ew]
    x, y = P.tangent(ra0, dec0, ra, dec)
    w = P.lens(P.R0, 2.6, np.hypot(x, y)) / (np.pi * P.R0 ** 2)
    n_data = np.bincount(d["det_exp"], minlength=len(d["exp_t"]))[ew] / np.maximum(w, 1e-3)
    X = np.load(a.exps)
    pos = {n: i for i, n in enumerate(X["name"])}
    n_model = X[a.n1][[pos[n] for n in d["exp_name"][ew]]]
    meas = collections.defaultdict(list)
    for f in glob.glob(a.runs):
        if "rank" not in f:
            for q in json.load(open(f))["per_gamma"]:
                meas[q["r"]].append(q["pairs"])
    rs = np.array(sorted(meas)); mp = np.array([np.mean(meas[r]) for r in rs])
    m_nodes = np.exp(np.interp(np.log(NODES), np.log(rs), np.log(mp)))
    pd_ = P.window_pairs(tab, ra0, dec0, ra, dec, night, n_data, xe, 1 / NODES)
    pm = P.window_pairs(tab, ra0, dec0, ra, dec, night, n_model, xe, 1 / NODES)
    out = dict(ra0=ra0, dec0=dec0, n_exp=int(len(ew)), nodes=NODES.tolist(), measured=m_nodes.tolist(), proxy_data=pd_.tolist(),
               proxy_model=pm.tolist(), n1_data_sum=float(n_data.sum()), n1_model_sum=float(n_model.sum()))
    json.dump(out, open(a.out, "w"), indent=1)
    for r, mm, p1, p2 in zip(NODES, m_nodes, pd_, pm):
        print(f"{r:6.0f} au  measured {mm:.3e}  proxy(data) {p1:.3e} x{mm / p1:.3f}  proxy(model) {p2:.3e} x{mm / p2:.3f}")
    print(f"detections per window (whole disks): data {n_data.sum():.0f}, model {n_model.sum():.0f}")


if __name__ == "__main__":
    main()

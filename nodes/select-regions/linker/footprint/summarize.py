"""Totals of the whole-archive estimate (estimate.py output) in GB200-hours and Vista gb SU, per method and per
starting distance; where the cost sits (top patch-windows, declination bands, exposures per window).

Cost of a patch-window = sum over the field's bands of T_band / speed-up(method, band) x k(gamma) x proxy_pw(gamma)
/ measured_field(gamma), with T_band the field's GB200 time for that band (jobs <slurm-job-id> / 1109527 / 1109528),
measured_field the field's raw pairs per hypothesis and k = measured / proxy(field's own counts) (1.20-1.27).

Usage: python -I footprint/summarize.py EST.json FIELD.json [--json OUT.json]
"""
import argparse, json
import numpy as np


def speed(f60, f85, r):
    x = np.clip((np.log(r) - np.log(61.0)) / (np.log(86.5) - np.log(61.0)), 0, 1)
    return np.exp(np.log(f60) * (1 - x) + np.log(f85) * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("est"); ap.add_argument("field"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    E, F = json.load(open(a.est)), json.load(open(a.field))
    nodes = np.array(E["nodes"]); lr = np.log(nodes)
    meas, pdat = np.array(F["measured"]), np.array(F["proxy_data"])
    k = meas / pdat
    bands = np.array(E["bands"]); rb, Tb = bands[:, 0], bands[:, 1]
    W = E["windows"]
    PR = np.array([w["proxy"] for w in W])                           # (n_pw, n_nodes)
    # ratio of each patch-window's predicted pairs to the field's measured pairs, at each band's distance
    rat_nodes = PR * k / meas
    rat = np.exp(np.array([np.interp(np.log(rb), lr, np.log(np.maximum(r, 1e-30))) for r in rat_nodes]))   # (n_pw, n_bands)
    out = dict(n_patch_windows=len(W), n_patches=len(set(w["p"] for w in W)), field_gb200_h=float(Tb.sum() / 3600), methods={})
    print(f"{len(W)} patch-windows in {out['n_patches']} patches; the OF201 field's window = 1.00 by construction "
          f"(its 50-1000 au search: {Tb.sum() / 3600:.1f} GB200-h brute)")
    print(f"field-equivalents (sum of pair ratios at 60 au): {rat[:, np.argmin(abs(rb - 61))].sum():.0f}")
    for name, (f60, f85) in E["speed"].items():
        cost = (rat * (Tb / speed(f60, f85, rb))).sum(1) / 3600         # GB200-h per patch-window
        row = {}
        for r0 in [50, 60, 80, 100, 150]:
            mb = rb >= r0
            c = (rat[:, mb] * (Tb[mb] / speed(f60, f85, rb[mb]))).sum() / 3600
            row[f"from_{r0}au"] = dict(gb200_h=float(c), su_gb=float(c / 4))
        srt = np.sort(cost)[::-1]
        row["top1pct_share"] = float(srt[:max(1, len(srt) // 100)].sum() / srt.sum())
        row["top10pct_share"] = float(srt[:max(1, len(srt) // 10)].sum() / srt.sum())
        out["methods"][name] = row
        print(f"{name:24} " + "  ".join(f">={r0}au {row[f'from_{r0}au']['gb200_h']:9.3g} GB200-h ({row[f'from_{r0}au']['su_gb']:8.3g} SU)" for r0 in [50, 60, 100])
              + f"  top 1%: {row['top1pct_share']:.0%}, top 10%: {row['top10pct_share']:.0%}")
    # galactic latitude of each patch, and a cap on seed exposures per window (pairs ~ seeds^2, verify ~ seeds^2 x
    # exposures / seeds): cost x (0.8 q^2 + 0.2 q), q = cap / n_exp, for windows above the cap [E]
    rad = np.radians
    pra, pdec = np.array([w["ra"] for w in W]), np.array([w["dec"] for w in W])
    sb = (np.sin(rad(pdec)) * np.sin(rad(27.12825)) + np.cos(rad(pdec)) * np.cos(rad(27.12825)) * np.cos(rad(pra - 192.85948)))
    gb = np.degrees(np.arcsin(np.clip(sb, -1, 1)))
    nx = np.array([w["n_exp"] for w in W], float)
    def sep(r0, d0):
        return np.degrees(np.arccos(np.clip(np.sin(rad(pdec)) * np.sin(rad(d0)) + np.cos(rad(pdec)) * np.cos(rad(d0)) * np.cos(rad(pra - r0)), -1, 1)))
    clean = (np.abs(gb) > 15) & (sep(80.9, -69.8) > 8) & (sep(13.2, -72.8) > 4)        # off the plane and the Magellanic Clouds
    out["n_clean"] = int(clean.sum())
    out["variants"] = {}
    for name in ["index+veto", "index+veto+sharp", "index+veto+sharp+flags"]:
        f60, f85 = E["speed"][name]
        mb = rb >= 50
        cost = (rat[:, mb] * (Tb[mb] / speed(f60, f85, rb[mb]))).sum(1) / 3600
        v = {"all": float(cost.sum()), "|b|>15": float(cost[np.abs(gb) > 15].sum()), "|b|>15 no MCs": float(cost[clean].sum())}
        for cap in [300, 150, 100]:
            q = np.minimum(1, cap / nx)
            cc = cost * (0.8 * q * q + 0.2 * q)
            v[f"seed cap {cap}"] = float(cc.sum())
            v[f"seed cap {cap}, |b|>15 no MCs"] = float(cc[clean].sum())
        out["variants"][name] = v
        print(f"{name:24} from 50 au, GB200-h: " + ", ".join(f"{k} {x:.3g}" for k, x in v.items()))
    # the table for the report: method x sky x seed cap x starting distance
    out["table"] = {}
    print(f"windows with > 1000 exposures: {(nx > 1000).sum()} ({(clean & (nx > 1000)).sum()} clean)")
    print(f"{'GB200-h':42} " + " ".join(f"{'>=' + str(r0) + ' au':>9}" for r0 in [50, 60, 80, 100]))
    for name in ["index", "index+veto", "index+veto+sharp", "index+veto+sharp+flags"]:
        f60, f85 = E["speed"][name]
        per = rat * (Tb / speed(f60, f85, rb)) / 3600                     # (n_pw, n_bands)
        for sky, msk in [("all", np.ones(len(W), bool)), ("clean", clean), ("clean, <= 1000 exp", clean & (nx <= 1000)),
                         ("clean, > 1000 exp", clean & (nx > 1000))]:
            for cap in [0, 300, 150, 100]:
                q = np.minimum(1, cap / nx) if cap else np.ones(len(W))
                f = (0.8 * q * q + 0.2 * q)[:, None]
                vals = {r0: float((per * f)[msk][:, rb >= r0].sum()) for r0 in [50, 60, 80, 100]}
                key = f"{name} | {sky} | cap {cap or 'none'}"
                out["table"][key] = vals
                print(f"{key:56} " + " ".join(f"{vals[r0]:9.3g}" for r0 in [50, 60, 80, 100]))
    # where the brute-force cost sits
    cost = (rat * Tb).sum(1) / 3600
    print(f"off the plane (|b| > 15) and the Magellanic Clouds: {clean.sum()} patch-windows, {cost[clean].sum() / cost.sum():.1%} of the cost; "
          f"median {np.median(cost[clean]):.1f} GB200-h each (brute), field = {Tb.sum() / 3600:.1f}")
    dec = np.array([w["dec"] for w in W]); nexp = np.array([w["n_exp"] for w in W])
    print("by declination (brute GB200-h share):")
    for lo, hi in [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 90)]:
        m = (dec >= lo) & (dec < hi)
        print(f"  dec {lo:+d}..{hi:+d}: {m.sum():5d} patch-windows, {cost[m].sum() / cost.sum():5.1%}")
    print("by exposures per window:")
    for lo, hi in [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1e9)]:
        m = (nexp >= lo) & (nexp < hi)
        print(f"  {lo}-{hi:.0f}: {m.sum():5d} patch-windows, {cost[m].sum() / cost.sum():5.1%}, median {np.median(cost[m]) if m.any() else 0:.2f} GB200-h each")
    top = np.argsort(cost)[::-1][:8]
    print("costliest patch-windows (brute):", [(round(W[i]["ra"], 1), round(W[i]["dec"], 1), W[i]["n_exp"], round(float(cost[i]), 1)) for i in top])
    out["by_dec"] = {f"{lo}..{hi}": float(cost[(dec >= lo) & (dec < hi)].sum() / cost.sum()) for lo, hi in [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 90)]}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

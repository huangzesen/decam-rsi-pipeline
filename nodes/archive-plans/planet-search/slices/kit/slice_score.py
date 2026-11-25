"""Steps 3 and 4 of a slice: its reward on the known bodies, and its candidates.

Reads what slice_run.py kept (WORKDIR/out/<sid>/) and the slice's known bodies (SLICE_DIR/private/known.json).

3. reward     the RSI task grader's rules (rsi/harness/grade.py):
   - slots: over all windows of the slice, the 25 x n_windows candidates with the highest p are kept (a window offers
     at most 200); candidates inside 25 au are ignored;
   - a kept candidate's orbit is propagated (two-body about the Sun, light time) to every exposure of its window; it
     is a hit on a known body when it lies within max(2 arcsec, the body's 3-sigma) of it on >= 3 nights; a body
     counts once; scored bodies are tried before neutral ones;
   - tier of a body in a window, from the forced photometry taken during the run: 1 = >= 3 nights with one exposure
     at S/N >= 5; 2 = >= 3 nights whose night stack reaches 5; 3 = the stack of all exposures reaches 5 on >= 3
     nights; 0 = too faint in these images; None = orbit too uncertain (3-sigma > 1.5 arcsec). Weights 1 / 2 / 4;
     tiers 0 and None are neutral. A window that did not run keeps its predicted tiers, so its bodies count as missed;
   - reward = weight of the bodies found / weight of all findable bodies;  recall = bodies found / findable bodies.
   Writes WORKDIR/reward.json (numbers only) and WORKDIR/private/reward_bodies.json (every body).

4. candidates every kept candidate, with what it is:
   - "known": it matched a known body;
   - "lead": it matched none, has p >= 0.5, no known body within 10 arcsec, and passes the screen measured on the
     images during the run (rules of rsi/candidates/screen_new.py): >= 3 nights with a stack S/N >= 4.5 on its
     track, the same sky spot on other nights below S/N 3 (not a static source), and S/N >= 5 without its
     brightest exposure;
   - "unmatched": the rest.
   Writes WORKDIR/candidates.json and one file per lead in WORKDIR/leads/, in the input format of the novelty check
   (rsi/candidates/novelty_queue/inbox/). Leads hold sky positions of possibly new bodies: private.

Usage: python -I lib/slice_score.py SLICE_DIR WORKDIR [--k 25] [--tol 2.0] [--max-sigma 1.5] [--p-lead 0.5]
"""
import argparse, json, math, os, shutil, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

W_TIER = {1: 1.0, 2: 2.0, 3: 4.0}


def measured_tier(body, phot, max_sigma):
    if body["max_sigma"] > max_sigma:
        return None
    vis = [x for x in phot if "hdu" in x and x.get("flag") == "" and x.get("err")]
    nights = {}
    for x in vis:
        nights.setdefault(x["night"], []).append(x)
    n1 = sum(any((x["snr"] or 0) >= 5 for x in v) for v in nights.values())
    n2 = sum(C.stack([x["flux"] for x in v], [x["err"] for x in v]) >= 5 for v in nights.values())
    tot = C.stack([x["flux"] for x in vis], [x["err"] for x in vis]) if vis else 0.0
    return 1 if n1 >= 3 else 2 if n2 >= 3 else 3 if (tot >= 5 and len(nights) >= 3) else 0


def screen(ph):
    """The lead screen on a candidate's photometry record (cand_phot.json). Returns (verdict, reasons)."""
    if ph is None:
        return "not measured", []
    good = sum(v >= 4.5 for v in ph["nights"].values())
    st = [v for v in ph["static"].values() if v is not None]
    why = []
    if good < 3:
        why.append(f"{good} nights >= 4.5")
    if st and max(st) >= 3:
        why.append(f"static {max(st)}")
    if ph["drop"] < 5:
        why.append(f"without brightest {ph['drop']}")
    return ("pass" if not why else "reject"), why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slice"); ap.add_argument("work")
    ap.add_argument("--k", type=int, default=25); ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--max-sigma", type=float, default=1.5); ap.add_argument("--p-lead", type=float, default=0.5)
    a = ap.parse_args()
    sl = os.path.abspath(a.slice)
    card = json.load(open(os.path.join(sl, "slice.json")))
    W = json.load(open(os.path.join(sl, "windows.json")))["windows"]
    kp = os.path.join(sl, "private", "known.json")
    K = json.load(open(kp))["windows"] if os.path.exists(kp) else None
    out = os.path.join(a.work, "out")

    # every window's candidates, then the slice's slots
    win, allc = {}, []
    for w in W:
        d = os.path.join(out, w["sid"])
        info = json.load(open(os.path.join(d, "info.json"))) if os.path.exists(os.path.join(d, "info.json")) else {}
        ran = os.path.exists(os.path.join(d, "done.json")) and os.path.exists(os.path.join(d, "exposures.csv"))
        cands = C.load_cands(os.path.join(d, "candidates.json")) if ran else []
        for c in cands:
            c.update(sid=w["sid"], key=w["key"])
        win[w["key"]] = dict(w=w, d=d, info=info, ran=ran, cands=cands)
        allc += cands
    slots = a.k * len(W)
    kept = sorted(allc, key=lambda c: -c["p"])[:slots]
    cut = kept[-1]["p"] if len(allc) > slots else None

    tw = gw = 0.0
    per_tier = {t: [0, 0] for t in (1, 2, 3)}
    bodies_out, cand_out, leads = [], [], []
    nmeas = seconds = 0
    for key, x in win.items():
        w, d = x["w"], x["d"]
        seconds += x["info"].get("seconds") or 0
        kb = (K or {}).get(str(key), [])
        ph = json.load(open(os.path.join(d, "known_phot.json"))) if os.path.exists(os.path.join(d, "known_phot.json")) else None
        nmeas += ph is not None
        B = [dict(b, tier=measured_tier(b, ph.get(b["pdes"], []), a.max_sigma) if ph is not None else b["tier_pred"],
                  recs={r["stamp"]: r for r in b["recs"]}) for b in kb]
        for b in B:
            if b["tier"] in W_TIER:
                tw += W_TIER[b["tier"]]; per_tier[b["tier"]][1] += 1
        taken = {}
        if x["ran"]:
            stamps, mjds, obs, night = C.read_exposures(os.path.join(d, "exposures.csv"))
            cp = {}
            if os.path.exists(os.path.join(d, "cand_phot.json")):
                cp = {r["rank"]: r for r in json.load(open(os.path.join(d, "cand_phot.json")))["candidates"]}
            for c in [c for c in kept if c["key"] == key]:
                row = dict(slice=card["id"], sid=w["sid"], key=key, rank=c["rank"], p=c["p"], r=round(c["r"], 2), mag=c["mag"],
                           epoch_mjd=c["epoch"], state=c["state"])
                pos = C.positions(c, mjds, obs)
                if pos is None or c["r"] < C.RMIN_AU:
                    cand_out.append(dict(row, what="ignored: unbound or inside 25 au"))
                    continue
                ra, de = pos
                hit, near, dup = None, None, False
                for b in sorted(B, key=lambda b: b["tier"] not in W_TIER):
                    ok = set()
                    for k, st in enumerate(stamps):
                        r = b["recs"].get(st)
                        if r is None or r.get("ra") is None:
                            continue
                        s = float(C.sep(ra[k], de[k], r["ra"], r["dec"]))
                        near = s if near is None else min(near, s)
                        if s <= max(a.tol, min(r.get("s_ra") or 0, 60), min(r.get("s_dec") or 0, 60)):
                            ok.add(night[k])
                    if len(ok) >= 3 and b["pdes"] in taken:
                        dup = True
                    elif hit is None and len(ok) >= 3:
                        hit = b
                row["near_known_arcsec"] = None if near is None else round(near, 1)
                if hit is None and dup:
                    cand_out.append(dict(row, what="known (second candidate on a body already found)"))
                    continue
                if hit is not None:
                    taken[hit["pdes"]] = dict(rank=c["rank"], p=c["p"])
                    if hit["tier"] in W_TIER:
                        gw += W_TIER[hit["tier"]]; per_tier[hit["tier"]][0] += 1
                    cand_out.append(dict(row, what="known", known_tier=hit["tier"]))
                    continue
                verdict, why = screen(cp.get(c["rank"]))
                p = cp.get(c["rank"])
                row.update(screen=verdict, screen_why=why, nights_snr=p["nights"] if p else None,
                           static_snr=p["static"] if p else None, snr_all=p["tot"] if p else None,
                           snr_without_brightest=p["drop"] if p else None)
                lead = c["p"] >= a.p_lead and verdict == "pass" and (near is None or near > 10)
                cand_out.append(dict(row, what="lead" if lead else "unmatched"))
                if lead:
                    first = {}
                    for k in range(len(stamps)):
                        if p["snr"][k] is not None:
                            first.setdefault(night[k], k)
                    lid = f"{card['id']}-{w['sid']}-p{c['p']:.5f}-r{c['r']:.1f}"
                    leads.append(dict(id=lid, slice=card["id"], sid=w["sid"], window=key, p=c["p"], r=round(c["r"], 2), mag=c["mag"],
                                      points=[[mjds[k], float(ra[k]), float(de[k]), 0.3, "DECam"] for k in first.values()],
                                      confirm=True, nights_snr=p["nights"], static_snr=p["static"], snr_all=p["tot"],
                                      snr_without_brightest=p["drop"], program=card.get("program", {}).get("name")))
        for b in B:
            bodies_out.append(dict(window=key, sid=w["sid"], pdes=b["pdes"], name=b.get("name"), H=b.get("H"), r=b.get("r"), V=b.get("V"),
                                   tier=b["tier"], tier_pred=b["tier_pred"], ran=x["ran"], found=b["pdes"] in taken, **taken.get(b["pdes"], {})))

    nf, nt = sum(v[0] for v in per_tier.values()), sum(v[1] for v in per_tier.values())
    status = {}
    for x in win.values():
        s = x["info"].get("status", "not run")
        status[s] = status.get(s, 0) + 1
    um = [c for c in cand_out if c["what"] in ("lead", "unmatched")]
    rew = dict(slice=card["id"], program=card.get("program", {}).get("name"), windows=len(W), window_status=status,
               slots=slots, candidates_offered=len(allc), candidates_kept=len(kept), p_cut=cut,
               reward=round(gw / tw, 4) if tw else None, recall=round(nf / nt, 4) if nt else None,
               weight_found=gw, weight_total=tw, findable_bodies=nt, found=nf,
               by_tier={str(t): dict(found=v[0], total=v[1]) for t, v in per_tier.items()},
               not_scored=dict(too_faint=dict(found=sum(b["found"] for b in bodies_out if b["tier"] == 0),
                                              total=sum(b["tier"] == 0 for b in bodies_out)),
                               uncertain_orbit=dict(found=sum(b["found"] for b in bodies_out if b["tier"] is None),
                                                    total=sum(b["tier"] is None for b in bodies_out))),
               windows_with_measured_tiers=nmeas, known_list=K is not None,
               kept_known=sum(c["what"].startswith("known") for c in cand_out), kept_unmatched=len(um),
               unmatched_p_ge_lead=sum(c["p"] >= a.p_lead for c in um), leads=len(leads),
               finder_seconds=round(seconds, 1), a100_hours=round(seconds / 3600, 2))
    json.dump(rew, open(os.path.join(a.work, "reward.json"), "w"), indent=1)
    os.makedirs(os.path.join(a.work, "private"), exist_ok=True)
    json.dump(dict(rew, bodies=bodies_out), open(os.path.join(a.work, "private", "reward_bodies.json"), "w"), indent=1)
    json.dump(dict(slice=card["id"], slots=slots, p_cut=cut, candidates=sorted(cand_out, key=lambda c: -c["p"])),
              open(os.path.join(a.work, "candidates.json"), "w"), indent=1)
    ld = os.path.join(a.work, "leads")
    shutil.rmtree(ld, ignore_errors=True); os.makedirs(ld)
    for l in leads:
        json.dump(l, open(os.path.join(ld, l["id"] + ".json"), "w"), indent=1)

    print(f"slice {card['id']}: {status}")
    if K is None:
        print("3 reward:     no known-body list in this slice")
    else:
        print(f"3 reward:     {rew['reward']}  ({gw:g} of {tw:g} weight; recall {nf} of {nt} findable bodies; "
              + ", ".join(f"tier {t} {v[0]}/{v[1]}" for t, v in per_tier.items()) + ")")
    print(f"4 candidates: {len(kept)} kept of {len(allc)} offered ({slots} slots): {rew['kept_known']} known bodies, "
          f"{len(um)} unmatched ({rew['unmatched_p_ge_lead']} with p >= {a.p_lead}), {len(leads)} leads -> {ld}")
    print(f"finder time {seconds / 3600:.2f} GPU-hours; files: reward.json, candidates.json, leads/, private/reward_bodies.json in {a.work}")


if __name__ == "__main__":
    main()

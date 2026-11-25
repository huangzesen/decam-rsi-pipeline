# gen067 diagnosis workflow: bodies bright on only 2 nights (judge verdict + proposed bright-pair seed patch, UNTESTED)

**Verdict on the three reports.** All three rest on a wrong picture of the scenario. INJD is scen03, and its `exposures.csv` has 12 exposures on 8 nights (MJD 60496, 502, 517, 520, 535, 544, 547, 553). Five nights have 2 exposures about 10 minutes apart; nights 502, 535 and 553 have only 1. It is not "6 nights × 2–3 exposures", so every N≈15 figure is off. What changes:

- **Bright night.** The night co-add (`night_seeds`, stackd/finder.py:1491) is at most 5·√n_k because maps are clamped at :293. That is 5.0 on a 1-exposure night and 7.07 on a 2-exposure night.
  - The fix analyst's `BPT=6` therefore never fires for any pair involving nights 502, 535 or 553, which is 3 of 8 nights.
- **Weak night.** The night stack is 0.9·√n_k, so 0.9–1.27. With only about 9 weak exposures in total, weak nights carry almost no position information.
  - The fix analyst's "weak-night evidence picks d" is wrong for this scenario.
  - The new polish weak-night patch never runs on 1-exposure weak nights (`len(idx) < 2: continue`, :1200). On 2-exposure nights it takes an argmax over 441 noise pixels.
- **Body total.** It is about (5·n_bright + 0.9·n_weak)/√12, roughly 5.5–7.9. GATE3 passes it: `nstrong=2` because clamped 5 ≥ 4.5, `big3`, `ncov≥3` (:1015–1022). The REST median ≥ 0 check is roughly a coin flip if the weak-night positions are wrong. The gates analyst is right that the gates and the p computation are not the blocker.

**Confirmed (from the code):**
- **Linker:** cannot form these tracks (link.py 3-night cells; stack.py `ZT=6.8`).
- **Night seeds:** a 3rd night with ≥ 2.5 (`GOOD`) in the 3×3 max-pooled co-add is required (:1620–1622). For these bodies that only happens through a noise excursion. Every distance in `dg` is tried for the same bright pair, so whichever d's prediction lands on noise survives.
  - The rows are sorted by `sc`, which is noise-driven on the weak nights.
  - The `hit_ >= 2` dedup (:883–889) then keeps the first row and drops the other distances of the same pair.
  - The seed's d, and so its weak-night prediction, is effectively chosen by noise.
- **`DTMAX=25`** (:1553): it removes 13 of the 28 night pairs here.
- **CEM** (:1040–1064) maximises the clamped total with a distance sigma of 0.03·r. On the bright-night plateau it has no positional gradient, so d drifts toward weak-night noise. `cv_rerank` and `dedup` never correct states.

**(1) Root cause.** Two bright nights fix 4 of the 6 orbit parameters. The remaining distance / radial degeneracy is resolved by noise, never by a prior. That happens in the 3rd-night requirement plus `sc` ranking in `night_seeds` and in the CEM refinement. The 3rd-night prediction scales roughly as ½·(1.5"/day² at 40 au)·τ₁τ₂·(δd/d), which is 55–230" times δd/d for the nights here. With d chosen by noise it ends up more than 2" off on every weak night, so no candidate scores 3 nights.

The physical prior does fix d. The injection uses speed 0.92–1.05·v_c, ±20° direction and ±5% radial, and `to_state` (:809) uses zero radial velocity, so |v_dframe|/vcs(d) ≈ 1 sets d to about 1–2%. That puts nights adjacent to a bright night (3–6 days away) within about 1–2". Cutting `DTMAX` and requiring a 3rd night by luck lose the rest. Seeds are also time-sliced (0.15·budget, :856), and the gen066 run used `--budget-s 400`.

**(2) Patch: a bright-pair, prior-fixed-distance seed path.** These candidates skip CEM and polish, are scored by the existing `nights_good`/p formula, get p × `TNO_BP_W` (0.5), and are added through `dedup`, so existing higher-p candidates always win. `TNO_BP=0` disables it. Nothing here has been run.

**A. `night_seeds`, set-up.** Old (stackd/finder.py:1557–1559):
```python
    rows, frames = [], []
    TF0t = TF0
    nd = 0; ncand = 0
```
New:
```python
    rows, frames = [], []
    TF0t = TF0
    nd = 0; ncand = 0
    # gen 67: bright-pair seeds (TNO_BP=0 off): peaks near the clamp ceiling on 2 nights, distance fixed by the circular-speed
    # prior (|v|/vcs ~ BP_R0, angle to vc <= BP_A), no DTMAX, no 3rd-night requirement; returned with r[7] = -2.
    BP = os.environ.get("TNO_BP", "1") != "0"
    BPF = float(os.environ.get("TNO_BP_F", "0.85")); BPV = float(os.environ.get("TNO_BP_V", "0.10"))
    BPR0 = float(os.environ.get("TNO_BP_R0", "0.985")); BPC = math.cos(math.radians(float(os.environ.get("TNO_BP_A", "35"))))
    BPK = int(os.environ.get("TNO_BP_K", "300")); BPN = int(os.environ.get("TNO_BP_N", "300"))
    bp_pairs = [(a, b) for a in range(nN) for b in range(a + 1, nN)]
    bri = {}
    for k in range(nN):
        wk = np.asarray(wf[nidx[k]], float); zc = 5.0 * wk.sum() / math.sqrt(max((wk ** 2).sum(), 1e-9))
        for m_ in GI[k]:
            ii_ = np.flatnonzero(ZK[m_] >= BPF * zc)
            bri[m_] = ii_[np.argsort(-ZK[m_][ii_])[:BPK]]
    bp_best = {}
```

**B. Inside `for d in dg`.** Old (:1579–1580):
```python
            Q.append(np.stack([q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]], 1))
        if GPUP:
```
New:
```python
            Q.append(np.stack([q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]], 1))
        if BP:
            Bq = [Q[k][bri[gsel[k]]] for k in range(nN)]
            Bt = [cKDTree(q_) if len(q_) else None for q_ in Bq]
            hk, hr, hC = [], [], []
            vcv = np.array([vcx, vcy])
            for a, b in bp_pairs:
                if Bt[b] is None or len(Bq[a]) == 0:
                    continue
                dt = tN[b] - tN[a]
                rad = dt * vcs * (BPV + abs(1 - BPR0) + 2.2 * math.sqrt(max(0.0, 0.5 * (1 - BPC)))) + 2 * TOL
                lst = Bt[b].query_ball_point(Bq[a] + dt * vcv[None], rad)
                ia = np.repeat(np.arange(len(lst)), [len(l) for l in lst])
                if len(ia) == 0:
                    continue
                ib = np.concatenate([np.asarray(l, dtype=np.int64) for l in lst])
                v = (Bq[b][ib] - Bq[a][ia]) / dt
                sp = np.hypot(v[:, 0], v[:, 1])
                rr = np.abs(sp / vcs - BPR0)
                ok = (rr <= BPV + TOL / (dt * vcs)) & ((v @ vcv) >= BPC * sp * vcs)
                for j in np.flatnonzero(ok):
                    pa = PK[gsel[a]][bri[gsel[a]][ia[j]]]; pb = PK[gsel[b]][bri[gsel[b]][ib[j]]]
                    key = (a, int(pa[0]) // 4, int(pa[1]) // 4, b, int(pb[0]) // 4, int(pb[1]) // 4)
                    if key in bp_best and bp_best[key][0] <= rr[j]:
                        continue
                    hk.append(key); hr.append(float(rr[j]))
                    hC.append(np.r_[Bq[a][ia[j]] + v[j] * (tref - tN[a]), v[j]])
            if hk:
                Cb = np.array(hC); vb = np.zeros((len(Cb), nN), np.float32); cb = np.zeros((len(Cb), nN), bool)
                for k in range(nN):
                    P = Cb[:, :2] + Cb[:, 2:] * (tN[k] - tref)
                    wv = TD[2][None] + P[:, 0:1] * TD[0][None] + P[:, 1:2] * TD[1][None]
                    X = d * wv / np.linalg.norm(wv, axis=1)[:, None] - oN[k][None]
                    q = X @ TF0t.T
                    ix = np.rint((q[:, 0] / q[:, 2] - x0) / PIX).astype(np.int64)
                    iy = np.rint((q[:, 1] / q[:, 2] - y0) / PIX).astype(np.int64)
                    okb = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
                    vv = NMs[gsel[k]][torch.from_numpy(np.where(okb, iy * W + ix, 0)).to(dev)].float().cpu().numpy()
                    vb[:, k] = np.where(okb, vv, 0); cb[:, k] = okb & (vv != 0)
                scb = np.clip(vb, -2, 6).sum(1) / np.sqrt(np.maximum(cb.sum(1), 1))
                fib = len(frames); frames.append((TD, tref, oref))
                for j, key in enumerate(hk):
                    if key not in bp_best or hr[j] < bp_best[key][0]:
                        bp_best[key] = (hr[j], (float(scb[j]), d, float(Cb[j, 2]), float(Cb[j, 3]),
                                                float(Cb[j, 0]), float(Cb[j, 1]), fib, -2))
        if GPUP:
```
The extra `frames.append` on hit distances is harmless; to_state indexes by `fi`.

**C. Return.** Old (:1652–1653):
```python
    rows.sort(key=lambda r: -r[0])
    return rows, frames
```
New:
```python
    rows.sort(key=lambda r: -r[0])
    if BP and bp_best:
        rbp = sorted((x[1] for x in bp_best.values()), key=lambda r: -r[0])[:BPN]
        log_(f"bright-pair seeds: {len(bp_best)} pair keys, {len(rbp)} kept")
        rows = rows + rbp
    return rows, frames
```

**D. `main`, split the rows (no CEM).** Old (:850):
```python
    skind = np.zeros(0, int)
```
New:
```python
    skind = np.zeros(0, int)
    BPS = []      # gen 67: bright-pair night-seed states, scored unrefined after polish
```
Old (:864–865):
```python
            NSs, nsc = [], []
            for r in nrows[:4 * NSN]:
```
New:
```python
            NSs, nsc = [], []
            for r in [r for r in nrows if r[7] < 0]:
                s_, ep_ = to_state((r[0], r[1], r[2], r[3], r[4], r[5], r[6] + off_))
                x = orbit.propagate(s_[:3], s_[3:], np.array([t_ep - ep_, t_ep - ep_ + 1e-3]))
                st_ = np.r_[x[0], (x[1] - x[0]) / 1e-3]
                if np.all(np.isfinite(st_)):
                    BPS.append(st_)
            nrows = [r for r in nrows if r[7] >= 0]
            for r in nrows[:4 * NSN]:
```

**E. After polish.** Old (:1313–1315):
```python
                log("polish failed:", repr(ex)[:300])
        try:
            rk_ = np.array(rkd)
```
New:
```python
                log("polish failed:", repr(ex)[:300])
        if BPS:
            try:
                Sb = np.array(BPS); vb_, cb_ = score_states(Sb)
                tb_ = total(vb_, cb_); gb_ = nights_good(vb_, cb_)
                rk2, rs2 = np.array(rkd), np.array(rsc)
                ref_ = rs2[rk2 == 2] if int((rk2 == 2).sum()) >= 30 else rs2
                med_ = np.median(ref_); mad_ = 1.4826 * np.median(np.abs(ref_ - med_)) + 1e-3
                BPW = float(os.environ.get("TNO_BP_W", "0.5"))
                add = []
                for j in range(len(Sb)):
                    r_ = np.linalg.norm(Sb[j, :3]); v2_ = Sb[j, 3:] @ Sb[j, 3:]
                    if gb_[j] < 3 or r_ < 25 or 0.5 * v2_ - GM / r_ >= 0:
                        continue
                    z_ = (tb_[j] - med_) / mad_
                    p_ = BPW * 0.8 / (1 + math.exp(min(60.0, max(-60.0, -(z_ - 4.5) / 0.8))))
                    add.append(dict(epoch_mjd=float(t_ep), state=[float(x) for x in Sb[j]], p=float(min(max(p_, 1e-4), 0.999))))
                cands = dedup(cands + add, score_states, t_ep)
                write_out(a.out, cands)
                log(f"bright-pair: {len(Sb)} states, {len(add)} pass gates, {sum(1 for c in cands if c in add)} in output")
            except Exception as ex:
                log("bright-pair failed:", repr(ex)[:300])
        try:
            rk_ = np.array(rkd)
```

**Why it can't displace strong candidates:**
- p is at most 0.4 before the wrapper's 0.7 `STACK_W`, so at most 0.28.
- Noise pairs (total about 3) get z far below 4.5, so p is about 0.
- `dedup` sorts by p, so an existing same-body candidate always wins.
- The 200 cap is unchanged.

**Cost.** At most 300 bright peaks per map, 28 pairs per distance, KD queries only. All names used are in scope in `night_seeds`: `wf`, `nidx`, `GI`, `ZK`, `PK`, `Q`, `gsel`, `TD`, `tref`, `oref`, `vcx`, `vcy`, `vcs`, `TOL`, `NMs`, `TF0t`, `x0`, `y0`, `PIX`, `W`, `H`, `dev`, `torch`, `cKDTree`. In `main`: `to_state`, `off_`, `t_ep`, `GM`, `dedup`, `a.out`.

**Validation (next generation, not run).** INJD at `--budget-s 900` with `TNO_KEEP=1`. Check the `bright-pair:` log line, and per D body the number of nights within 2". `TNO_BP=0` restores gen067-before behaviour.

# Analyst reports

**Report: can any seed path produce a state near a body with 2 bright nights and nothing else?**

**Bottom line:** The linker cannot. All three stackd paths can, but only under conditions. The window search and the night seeds are the realistic routes. Seeding is not a hard zero, so the 0/20 in gen066 is not explained by seeding alone. It was a Mac run at `--budget-s 400` (history/gen066/trace.md:1316), not 900, which shortens every time-bounded stage.

**1. Linker (linker/)**
- **Seeds:** detections at S/N >= 3.0–3.5 (PE) that have a same-night partner exposure, or singles at S1 4.3–5.0 (link.py:359, 374–433). On a weak night (0.9 per exposure) one seed needs a ~2.1–4σ noise excursion, and a partner pair needs two. In practice that is zero.
- **Linkage:** a hash cell needs pair endpoints on >= 3 distinct nights (link.py:516–529, 544), and the final members need >= 3 nights (link.py:630). Extending to the >= 3σ catalogue only happens after a 3-night cell exists, so 2 bright nights can never start a linkage.
- **hough.py:** finds within-night tracklets only (link.py:326–353).
- **linker/stack.py:** pairs from the two bright nights do form (seeds S/N >= 3.5, stack.py:157–164). But it then requires z >= ZT = 6.8 over the other nights' exposures (clamped to [-3,4]) and npos >= 2 nights with a night stack above 1.5 (stack.py:173, 222–241). The expected z is 0.9·sqrt(N_other), about 2.8 for 10 exposures, and would need about 57 exposures to reach 6.8. So: no.

**2. stackd window search (stackd/finder.py:338–368, 519–697)**
- Windows are contiguous ranges of >= 3 nights (`b_ >= a_+2`, :340). Each night contributes up to NPN frames (3 on Mac, 8 on CUDA). The first window is the one with the most nights within HMAX. Further windows must share at most 1 night with every chosen window, and there are at most 6 (:361–368). Later windows can be skipped for time (:470).
- If both bright nights fall in one searched window, the peak score is roughly (2·n_b·5 + 0.9·n_w)/sqrt(n). For a 3-night window with 3 exposures per night that is about 11, well above PTHR 4.2–4.5 (:337), and the two nights fix the velocity. This path works.
- If only one bright night is in the window, the velocity is degenerate and the seed is wrong.
- Coverage is the limit: with 6 nights and 3-night windows, the overlap rule lets roughly 2–3 windows through, covering about 6–9 of the 15 night pairs (my estimate, not measured). Time-driven extra windows (:711–776) and the cap NSEED = 1500 on Mac, ×6 on CUDA (:782–799), apply on top.

**3. Deep windows (:400–468, 548–660)**
- Prior: |v − v_circ| <= 0.3·v_circ at 34–52 au, then 0.55 at 28–60 and 0.3 at 60–110 (:405, :416).
- Three things penalise this body class:
  - DCLIP clamps each frame at 4 (:567–570).
  - LMO subtracts the largest co-add term (:612–626, 641–642), and each night collapses to about one co-add (GSPAN 0.08 d). That removes one of the two bright nights, so about half the signal is gone.
  - The score is then a velocity-contrast z (:647–662).
- Injected directions of ±20° give |Δv| up to about 0.35·v_c (injectD.py), which is outside the first deep prior. This path is weak.

**4. Night seeds (`night_seeds`, :1403–1656)**
- **Bright nights:** they are easily found as night peaks (NZ 3.7, :1409). Maps are clamped to 5, so the night co-add is about 5·sqrt(n), roughly 7.9. The CAP of 1500–12000 peaks is not a problem.
- **3-night requirement:** a hard gate. The predicted position must read >= GOOD = 2.5 on >= 3 nights in the 3×3 max-pooled night co-add (:1394, :1621–1623; docstring :1407).
  - A weak night's co-add carries about 0.9·sqrt(2.5), roughly 1.4, of signal, plus the max-of-9-pixels bias.
  - My rough estimate is 20–40% per weak night, so about 60–85% that at least one of 4 weak nights passes. The 3rd night then reaches >= 2.5 by chance.
  - The prediction from 2 nights must land within ±1 px (1"), on a distance grid stepped at 3" of parallax nonlinearity (:1544), which lowers this further.
- **Pair-time hole:** pairs are formed only for night gaps <= DTMAX = 25 d (:1555–1556). On scen03 (8 nights over 57 days per gen065 notes), a body whose two bright nights are more than 25 d apart can never be paired.
- **Deadline:** the night-seed slice is about 0.15·budget (:856), with distances ordered by closeness to 42 au. At 400 s, bodies at 30 or 60 au may never be reached.
- Ranking is not an issue: the body's score of about (6+6+2.5+~1.3·3)/sqrt(6), roughly 7.5 (:1622), would put it near the top. Dedup (:873–901) is not an issue either.

**What to check next (log lines, no new runs needed beyond one TNO_KEEP run):**
- `windows (nights, cost)` and `window i:`, to see which night pairs were searched.
- `night seeds: x/y distances`, to catch a deadline hit.
- Per D body: are both bright nights inside one chosen window? Is the bright-night gap <= 25 d? Does any refined kind-1 or kind-2 state pass within 2" on both bright nights?

If seeds are present, the loss is downstream: CEM, the GATE3 p calibration, or the fitted 3rd-night position being more than 2" off. The cheapest seed-side fixes are a night-seed mode that admits 2 nights when both have night peaks at z >= ~6 and the gap is <= DTMAX (with the 3rd-night requirement relaxed, at the cost of more noise pairs), and raising DTMAX.

The main file is $RSI_SANDBOX/r1/gen067/workspace/submission/stackd/finder.py, with $RSI_SANDBOX/r1/gen067/workspace/submission/linker/link.py and $RSI_SANDBOX/r1/gen067/workspace/submission/linker/stack.py.

---

**Verdict:** none of the gates, the p computation, the merge or the 200 cap would drop a seeded body of this class. The likely failure is positional: the state the code ends up with is probably more than 2" off on every weak night, so the grader never sees a 3rd night. Nothing below was run; the arithmetic assumes 6 nights with 2.5 exposures each, so N≈15.

**Clamped-map arithmetic, which corrects NOTES.md.** NOTES' "10/sqrt(n_cov)" assumes only one bright exposure per bright night. At 2-3 exposures per night, the bright nights give about 5 exposures × 5 (clamped) = 25. The weak nights give about 10 × 0.9 = 9. That makes total ≈ 34/√15 ≈ 8.8 (`total`, stackd/finder.py:947-952).

**Gates are not the blocker (stackd/finder.py:994-1024).**
- On the bright nights, `vmax ≥ 2.5`, so g=2.
- `big` (8.8 ≥ 5.5) holds, but a weak night's stack is 0.9·√2.5 ≈ 1.4, around GATE2=1.5 (:1008). So it passes only sometimes.
- GATE3 passes: big3 ≥ 3.0, nstrong=2, ncov ≥ 3, and the median of the other nights is about +1.4 ≥ 0 (:1015-1022). So nn=3.
- `to_cands` (:954-975) therefore keeps the body, with p ≤ 0.8 (:973).

**p is low but not fatal.**
- In stackd, p ≤ 0.8·sigmoid(...). The logged top p of 0.76 is just under that 0.8 cap.
- In the wrapper, stack-only candidates get ×0.7 (`STACK_W`, finder.py:19, 196-197), so p ≤ 0.56. `dens_calib` can multiply that by 0.6 more (:84-97), giving about 0.34.
- That is below `P_STRONG` = 0.8 (:26), so the body lands in the interleaved tail (:103-124).
- The final cut is `rank_balance(cs)[:200]` (:247); stackd also caps at 200 (:51, :1707). Either way, a body with total 8.8 should still be well inside the top 200.

**Main failure 1: CEM refinement (stackd/finder.py:1040-1067).**
- CEM maximises the clamped total. On the bright nights the maps sit on a flat plateau at 5, a few pixels wide, so the optimiser gets no positional signal inside it.
- The bright nights alone leave the distance and velocity degenerate, so the weak-night position is set only by the weak exposures.
- The true track's weak-night share is about 9/√15 ≈ 2.3. A noise-only alternative track scores about 0.82·N(0,1), and the best of the many tracks consistent with the 2 bright nights reaches about 1.7-2.0.
- So CEM picks the true weak-night position only about half the time. Otherwise it is several arcsec off on all weak nights.
- `cv_rerank` (:1712) only reorders candidates and never changes states.

**Main failure 2: the polish weak-night patch (:1212-1240).**
- It takes the argmax of a night stack over the radius-10 px circle, about 317 px, which is N(0,1) per pixel.
- Noise alone reaches ≥ 3 with probability about 1-(1-0.00135)^317 ≈ 0.35 per weak night, assuming independent pixels.
- The body pixel (about 1.4 + N(0,1)) reaches ≥ 3 only about 6% of the time, and is the argmax even less often.
- So across 4 weak nights there are about 1.3 noise points on average, and almost never a true one.
- The ≤1" clipping fit (:1243-1259) usually rejects these. It then falls back to `dgrid_fit` (:1115-1160), which pins the distance at 13 values and can fit a noise point that lies near the distance line.
- The acceptance check (:1289), `tot2[1] ≥ 0.97·tot2[0]` and `ng2 ≥ 3`, barely depends on weak nights. GATE3 still gives 3, so a wrong 3rd-night state gets accepted.
- With no point at all, polish skips the candidate (:1241) and keeps the CEM state.

**Merge does not rescue it.** `merge` / `dedup_sky` (finder.py:193-240) only match and dedup candidates; they never move a state. A linker match can't happen either, because a body bright on only 2 nights is too few detections to seed the linker.

**Recommendations:**
1. Raise `POL_NS`, or require the night-stack peak to lie within about 3" of the prediction and to be the maximum along the predicted track line, not the maximum over the whole circle.
2. In polish's acceptance, require the weak-night stack S/N at the new state to be ≥ the old one.
3. Run `TNO_POL_DEBUG` on INJD (listed as a debug run, not computed here) to measure each D candidate's off-sky distance per night before and after polish.

---

**Design: a bright-pair seed path in `night_seeds()` for bodies bright on only 2 nights. Untested; nothing was run.**

**Why stackd never seeds these bodies today (from the code):**
1. `night_seeds` keeps a pair only if it reads ≥ 2.5 on ≥ 3 nights in the night co-adds (`stackd/finder.py:1620-1622` on CPU, `1393-1394` on GPU). A weak night co-add is about 0.9·√2.5 ≈ 1.4, so a body with 4 weak nights passes only by noise luck.
2. Night pairs more than 25 days apart are never tried (`TNO_NS_DTMAX`, `:1553-1554`). Bright nights are chosen at random (`experiments/injectD.py`), so in a 57-day scenario many bright pairs fall outside this window.
3. Rows are trimmed by forced-sum score (`:1648`), and noise 3-night rows can push out 2-night rows.
4. The seed dedup (`:883-889`, `hit_ >= 2`) drops every other distance hypothesis of the same pair, because they all agree on the 2 bright nights.

**The change (one block in `night_seeds`, plus a 3-line merge):**

Before the distance loop (`:1557`):
```python
BPT = float(os.environ.get("TNO_BP_T", "6.0"))      # 0 = off (gen066)
BPV = float(os.environ.get("TNO_BP_V", "0.15"))     # circular-speed tolerance (fraction of vcs)
BPN = int(os.environ.get("TNO_BP_N", "300"))
bp_pairs = [(a, b) for a in range(nN) for b in range(a+1, nN)]   # no DTMAX
rows_bp = []
```
Inside `for d in dg`, after `Q` is built (`:1579`):
```python
if BPT > 0:
  for a, b in bp_pairs:
    ma = ZK[gsel[a]] >= BPT; mb = ZK[gsel[b]] >= BPT
    if not ma.any() or not mb.any(): continue
    dt = tN[b]-tN[a]; Qa, Qb = Q[a][ma], Q[b][mb]
    v = (Qb[None] - Qa[:, None]) / dt                       # all pairs (small sets)
    ok = hypot(v - [vcx, vcy]) <= BPV*vcs + TOL/dt          # tight circular prior fixes d
    for ia, ib in argwhere(ok):
      p0 = Qa[ia] + v[ia,ib]*(tref - tN[a])
      vals_k = forced read of NMs[gsel[k]] at prediction, all k   # reuse :1606-1619 body
      rest = [k for k covered, k not in (a,b)]
      if len(rest) < 1 or median(vals_k[rest]) < 0: continue
      s = sum(clip(vals_k[rest], -2, 6)) / sqrt(len(rest))   # weak-night evidence picks d
      rows_bp.append((s, d, vx, vy, px, py, fi, 2, (a, ia_global, b, ib_global)))
```
After the loop: for each pair key, keep the single best-`s` distance. Take the top `BPN` by `s` and return them as their own rows, outside the `:1648` trim. In `main` at `:861`, put these rows first in `nrows` and leave them out of the `hit_ >= 2` dedup. They are then refined by CEM like any other night seed. GATE3 (`:1015`) and the polish weak-night centroids (`:1190`) do the rest.

**Why a 3rd night can land within 2":**
- Two nights give 4 constraints. The tight circular prior (|v| ≈ v_c(d), and the d-frame model has no radial velocity) supplies the other 2.
- The observed rate is mostly reflex motion, Earth's ~30 km/s over d. Injected speeds are 0.92-1.05 v_c with ±5 % radial, so the TNO speed is uncertain by ~0.3 km/s, which gives δd/d ≈ 1 %.
- The 3rd-night error is roughly the curvature term ½·a_E·τ₁τ₂/d, about 230"·(δd/d) for nights ~10 days from a bright night and ~30 days from the other.
- So a night adjacent to a bright night (τ₁ ≈ 1-3 days) is nearly always within 2". A night 10+ days away is borderline, roughly 50 %. Only one 3rd night is needed.
- Main risk: the REST guard (`:1017-1021`, median ≥ 0) rejects about half of true bodies whose weak-night predictions are off by more than about 1.5". For these seeds, consider `TNO_GATE3_REST=-0.5`.

**False-positive estimate:**
- With per-exposure maps clamped at 5, a one-frame artifact reaches at most 5/√n ≈ 3.5 in a night co-add. BPT = 6 therefore needs a consistent multi-exposure source. This assumes the bright nights have ≥ 2 exposures and all of them are bright; if only one exposure per night is bright, BPT must drop to about 3.
- Assume ≤ 100 such peaks per night. The circular-prior window (≈ 0.15·72"/day·dt) covers ~7e-4 of the field for dt = 10 days, giving ~7 chance pairs per night pair per distance. Over 15 pairs and ~50 distances that is ~5e3 candidates. The REST median and best-d-per-pair steps cut this to a few hundred, capped at `BPN` = 300.
- Noise pairs end with a clamped total of ~2.5-3 and about 0.8·sigmoid p, so they sit below strong candidates. True bodies reach ~5-8.

**Runtime:** pair search over ≤ 100² peaks per pair, 15 pairs and ~50 distances, is under 2 s in numpy, and the forced reads reuse the existing NMs maps. CEM on 300 extra seeds is a few percent of the refinement budget, which already handles thousands of seeds.

**Validate (next gen, at --budget-s 900):** run INJD with `TNO_KEEP=1` and log the `rows_bp` count, how many fall within 2" of a D body on both bright nights, and that body's 3rd-night offset after polish. `TNO_BP_T=0` restores gen066 behaviour.
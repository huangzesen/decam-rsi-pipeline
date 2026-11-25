"""PI (and title, when the archive has one) of the DECam proposals that hold the most window exposures, for labelling
the groups of groups.py. One archive query per proposal (raw header fields DTPI, DTTITLE); known ones are kept.

Usage: python -I footprint/fetch_proposal_pis.py footprint/groups_windows.json footprint/proposal_titles.json [--top 90]
"""
import argparse, collections, json, os, time, urllib.request

API = "https://astroarchive.noirlab.edu/api"
ap = argparse.ArgumentParser()
ap.add_argument("windows"); ap.add_argument("out"); ap.add_argument("--top", type=int, default=90)
a = ap.parse_args()
e = collections.Counter()
for w in json.load(open(a.windows)):
    e[w["prop"]] += w["n_exp"]
out = json.load(open(a.out)) if os.path.exists(a.out) else {}
for p, _ in e.most_common(a.top):
    if p in out:
        continue
    body = {"outfields": ["proposal", "DTPI", "DTTITLE"],
            "search": [["instrument", "decam"], ["proc_type", "raw"], ["obs_type", "object"], ["proposal", p]]}
    try:
        req = urllib.request.Request(f"{API}/adv_search/find/?limit=1", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        x = r[1] if len(r) > 1 else {}
        out[p] = dict(pi=x.get("DTPI"), title=x.get("DTTITLE"))
    except Exception as ex:
        print(p, repr(ex)[:100])
        continue
    time.sleep(0.2)
json.dump(out, open(a.out, "w"), indent=1)
print(len(out), "proposals")

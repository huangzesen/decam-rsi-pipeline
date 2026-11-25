"""At NERSC: for each CFHT image in cfht_targets.json read its headers, find the CCD holding each target, download that CCD only.
Also fetches Gaia stars for a wide box (the box only). Finished files are skipped, so resubmitting carries on.
Usage: python -I cfht_fetch.py TARGETS OUTDIR"""
import concurrent.futures as cf, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request, warnings
from astropy.io import fits
from astropy.wcs import WCS
warnings.filterwarnings("ignore")
T = json.load(open(sys.argv[1])); D = sys.argv[2]; os.makedirs(D, exist_ok=True)
B = "https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/"
UA = {"User-Agent": "archive-cover/1.0"}


def get(u, tries=4, timeout=300):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=timeout).read()
        except Exception as e:
            err = e; time.sleep(5 * (i + 1))
    raise err


def gaia(path):
    if os.path.exists(path):
        return
    q = "SELECT ra, dec, pmra, pmdec, phot_g_mean_mag AS g FROM gaiadr3.gaia_source WHERE ra BETWEEN 213 AND 235 AND dec BETWEEN -20 AND -11 AND phot_g_mean_mag BETWEEN 15 AND 19.5"
    data = urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", PHASE="RUN", MAXREC=3000000, QUERY=q)).encode()
    r = urllib.request.urlopen(urllib.request.Request("https://gea.esac.esa.int/tap-server/tap/async", data=data, headers=UA), timeout=120); job = r.geturl()
    for _ in range(240):
        ph = get(job + "/phase", timeout=60).decode().strip(); print("gaia", ph, flush=True)
        if ph in ("COMPLETED", "ERROR", "ABORTED"):
            break
        time.sleep(15)
    assert ph == "COMPLETED", ph
    b = get(job + "/results/result", timeout=600)
    if b[:2] == b"\x1f\x8b":
        import gzip; b = gzip.decompress(b)
    open(path + ".part", "wb").write(b); os.replace(path + ".part", path); print("gaia rows", b.count(b"\n") - 1, flush=True)


def one(i):
    ts = [t for t in T if t["id"] == i]; out = {}
    txt = get(f"{B}{i}p.fits.fz?fhead=true").decode("ascii", "replace")
    blocks = re.split(r"(?m)^END\s*$", txt)
    for k, b in enumerate(blocks):
        if "CRVAL1" not in b or "ccd" not in b:
            continue
        h = fits.Header.fromstring(b.strip("\n") + "\n", sep="\n")
        nx, ny = h.get("ZNAXIS1", h.get("NAXIS1")), h.get("ZNAXIS2", h.get("NAXIS2"))
        if not nx or not ny:  # the primary header has no image
            continue
        s = re.findall(r"\d+", h.get("DATASEC", f"[1:{nx},1:{ny}]")); x0, x1, y0, y1 = [int(v) - 1 for v in s[:4]]
        hh = h.copy()
        for c in ("NAXIS1", "NAXIS2"):
            hh[c] = h.get("Z" + c, h.get(c))
        hh["NAXIS"] = 2
        w = WCS(hh, naxis=2)
        for t in ts:
            x, y = w.all_world2pix(t["ra"], t["dec"], 0)
            if x0 + 40 <= x <= x1 - 40 and y0 + 40 <= y <= y1 - 40:
                out[t["name"]] = [k, float(x), float(y)]
    for e in sorted({v[0] for v in out.values()}):
        p = f"{D}/{i}_{e}.fits"
        if not (os.path.exists(p) and os.path.getsize(p) > 100000):
            b = get(f"{B}{i}p.fits.fz?cutout=%5B{e}%5D")
            assert b[:6] == b"SIMPLE", b[:60]
            open(p + ".part", "wb").write(b); os.replace(p + ".part", p)
    return i, out


gaia(D + "/gaia.csv")
ids = sorted({t["id"] for t in T}); done = {}; p = D + "/map.json"
if os.path.exists(p):
    done = json.load(open(p))
todo = [i for i in ids if i not in done]; print(len(ids), "images,", len(todo), "to do", flush=True)
with cf.ThreadPoolExecutor(4) as ex:
    fs = {ex.submit(one, i): i for i in todo}
    for n, f in enumerate(cf.as_completed(fs)):
        try:
            i, o = f.result(); done[i] = o
        except Exception as e:
            print("FAILED", fs[f], repr(e)[:150], flush=True)
        if n % 10 == 0:
            json.dump(done, open(p + ".tmp", "w")); os.replace(p + ".tmp", p); print(n, "done", flush=True)
json.dump(done, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
z = sum("Z" in v for v in done.values())
print("DONE", len(done), "images; TRKSUB01 on a CCD in", z, "; files", len([f for f in os.listdir(D) if f.endswith(".fits")]))

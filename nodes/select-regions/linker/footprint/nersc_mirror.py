"""Catalogue of the DECam images already at NERSC (the DESI Legacy Surveys staging area,
/global/cfs/cdirs/cosmo/staging/decam, readable by every NERSC user), matched against the whole DECam archive.

2026-11-15

In:  data_real/nersc_decam_mirror/lists/ooi_*.txt   "relative path  size" per InstCal image (ooi), one file per subfolder
                                                    (made on Perlmutter with find, 2026-11-14/15; copied with rsync)
     data_real/noirlab_decam_meta/decam_meta_<year>.json     every DECam science exposure (fetch_archive_meta.py)
     data_real/noirlab_decam_instcal/instcal_<year>.json     every InstCal file at NOIRLab (fetch_archive_instcal.py)
     footprint/groups_exposures.json                         the search windows of >= 3 nights (groups.py)
Out: data_real/nersc_decam_mirror/mirror_exposures.tsv       one line per mirrored exposure
     footprint/nersc_mirror.json                             the totals printed here

Metadata only; nothing is downloaded. Usage (in linker/): ~/anaconda3/bin/python -I footprint/nersc_mirror.py
"""
import collections, glob, json, math, os, re

D = "data_real/nersc_decam_mirror"
pat = re.compile(r"(c4d_\d{6}_\d{6})_ooi_([^_]+)_([^.]+)\.fits\.fz$")

# 1. the mirror: exposure stamp -> copies
mir = collections.defaultdict(list)          # stamp -> [(folder, relative path, size, filter, version tag)]
nfiles = 0; nbytes = 0; perfolder = collections.Counter(); odd = 0
for f in sorted(glob.glob(f"{D}/lists/ooi_*.txt")):
    folder = os.path.basename(f)[4:-4]
    for line in open(f):
        p, _, s = line.rstrip("\n").rpartition(" ")
        m = pat.search(p)
        if not m:
            odd += 1; continue
        mir[m.group(1)].append((folder, p, int(s), m.group(2), m.group(3)))
        nfiles += 1; nbytes += int(s); perfolder[folder] += 1

# 2. the archive: science exposures and InstCal files at NOIRLab
F = ["ra", "dec", "caldat", "exptime", "filter", "proposal", "dateobs", "expnum", "object", "release", "stamp"]
inst = {}                                    # stamp -> [filter, size of the ooi file]
e2s = {}                                     # exposure number -> stamp, from the InstCal file names
for f in sorted(glob.glob("data_real/noirlab_decam_instcal/instcal_*.json")):
    for name, md5, size, rel, expnum in json.load(open(f))["rows"]:
        m = pat.search(name)
        if m and m.group(1) not in inst:
            inst[m.group(1)] = [m.group(2), size]; e2s[expnum] = m.group(1)
meta = {}; nostamp = 0                       # the 2012-2013 rows carry no stamp: take it from the InstCal name
for f in sorted(glob.glob("data_real/noirlab_decam_meta/decam_meta_*.json")):
    for r in json.load(open(f))["rows"]:
        s = r[10] or e2s.get(r[7])
        if s:
            meta[s] = r
        else:
            nostamp += 1
have = set(inst)                             # exposures with an InstCal image at NOIRLab
sci = set(meta)


def ecl_lat(ra, dec):
    e = math.radians(23.4392911); a = math.radians(ra); d = math.radians(dec)
    return math.degrees(math.asin(math.sin(d) * math.cos(e) - math.cos(d) * math.sin(e) * math.sin(a)))


def gal_lat(ra, dec):
    a = math.radians(ra - 192.85948); d = math.radians(dec); dp = math.radians(27.12825)
    return math.degrees(math.asin(math.sin(d) * math.sin(dp) + math.cos(d) * math.cos(dp) * math.cos(a)))


def band(v, edges, names):
    for e, n in zip(edges, names):
        if v < e:
            return n
    return names[-1]


EB = ([5, 15, 30], ["|b_ecl| < 5", "5-15", "15-30", ">= 30"])
GB = ([10, 20], ["|b_gal| < 10", "10-20", ">= 20"])
XB = ([30, 60, 120], ["< 30 s", "30-60 s", "60-120 s", ">= 120 s"])


def tally(stamps):
    """Counts by year, filter, sky and exposure time of the stamps that are science exposures."""
    t = dict(year=collections.Counter(), filter=collections.Counter(), ecl=collections.Counter(),
             gal=collections.Counter(), exptime=collections.Counter(), proposal=collections.Counter(),
             public=collections.Counter())
    for s in stamps:
        r = meta.get(s)
        if r is None:
            continue
        t["year"][r[2][:4]] += 1; t["filter"][r[4] or "?"] += 1; t["proposal"][r[5] or "?"] += 1
        if r[0] is not None and r[1] is not None:
            t["ecl"][band(abs(ecl_lat(r[0], r[1])), *EB)] += 1; t["gal"][band(abs(gal_lat(r[0], r[1])), *GB)] += 1
        t["exptime"][band(r[3] or 0, *XB)] += 1
        t["public"]["public" if (r[9] or "9999")[:10] <= "2026-11-15" else "proprietary"] += 1
    return t


ms = set(mir)
m_sci = ms & sci; m_have = ms & have
T_arch = tally(have & sci); T_mir = tally(m_sci)
out = dict(
    asof="2026-11-15", source="/global/cfs/cdirs/cosmo/staging/decam (NERSC, DESI Legacy Surveys staging area)",
    files=nfiles, odd_names=odd, exposures=len(ms), bytes_ooi=nbytes, tb_ooi=round(nbytes / 1e12, 1),
    files_per_folder=dict(perfolder.most_common()),
    exposures_with_several_copies=sum(1 for v in mir.values() if len(v) > 1),
    archive=dict(science_exposures=len(sci) + nostamp, science_without_any_file_name=nostamp, with_instcal=len(have), science_with_instcal=len(have & sci)),
    mirror_in_science_list=len(m_sci), mirror_in_instcal_list=len(m_have),
    mirror_not_in_science_list=len(ms - sci), mirror_not_in_instcal_list=len(ms - have),
    share_of_science_with_instcal=round(len(m_sci & have) / len(have & sci), 4),
)
for k in ("year", "filter", "ecl", "gal", "exptime", "public"):
    out["by_" + k] = {n: dict(archive=T_arch[k][n], mirror=T_mir[k][n],
                              share=round(T_mir[k][n] / T_arch[k][n], 3) if T_arch[k][n] else None)
                      for n in sorted(set(T_arch[k]) | set(T_mir[k]))}
titles = {}
try:
    titles = json.load(open("footprint/proposal_titles.json"))
except Exception:
    pass
out["by_proposal_top20"] = [dict(proposal=p, mirror=n, archive=T_arch["proposal"][p],
                                 title=(titles.get(p) if isinstance(titles.get(p), str) else None))
                            for p, n in T_mir["proposal"].most_common(20)]

# 3. the search windows (>= 3 nights): how much of each is already at NERSC
W = json.load(open("footprint/groups_exposures.json"))["windows"]
nw = len(W); full = 0; most = 0; none = 0; ne = 0; nem = 0; hist = collections.Counter()
wexp = set()
for wid, rows in W.items():
    n = len(rows); k = sum(1 for r in rows if r[0] in ms)
    ne += n; nem += k; wexp.update(r[0] for r in rows)
    fr = k / n
    full += fr == 1; most += fr >= 0.9; none += k == 0
    hist[min(int(fr * 10), 9)] += 1
out["windows"] = dict(windows=nw, exposure_slots=ne, slots_in_mirror=nem, share_of_slots=round(nem / ne, 4),
                      distinct_exposures=len(wexp), distinct_in_mirror=len(wexp & ms),
                      share_of_distinct=round(len(wexp & ms) / len(wexp), 4),
                      windows_all_in_mirror=full, windows_90pct_in_mirror=most, windows_none_in_mirror=none,
                      windows_by_tenth_in_mirror={f"{i * 10}-{i * 10 + 10}%": hist[i] for i in range(10)})
out["mirror_not_in_any_window"] = len(ms - wexp)

json.dump(out, open("footprint/nersc_mirror.json.new", "w"), indent=1)
os.replace("footprint/nersc_mirror.json.new", "footprint/nersc_mirror.json")

with open(f"{D}/mirror_exposures.tsv.new", "w") as fo:
    fo.write("stamp\tfilter\tcopies\tsize\tpath\tin_science_list\tin_noirlab_instcal\tin_window\tra\tdec\tnight\texptime\tproposal\texpnum\n")
    for s in sorted(ms):
        c = max(mir[s], key=lambda x: (x[0] == "DECam_CP-DR11", x[0] == "DECam_CP-DR10c", x[0].startswith("DECam_CP"), x[1]))
        r = meta.get(s)
        fo.write("\t".join(str(x) for x in (
            s, c[3], len(mir[s]), c[2], c[1], int(s in sci), int(s in have), int(s in wexp),
            *( (r[0], r[1], r[2], r[3], r[5], r[7]) if r else ("", "", "", "", "", "")))) + "\n")
os.replace(f"{D}/mirror_exposures.tsv.new", f"{D}/mirror_exposures.tsv")

print(json.dumps({k: v for k, v in out.items() if not k.startswith("by_")}, indent=1))
for k in ("year", "filter", "ecl", "gal", "exptime", "public"):
    print(k); [print("  ", n, v) for n, v in out["by_" + k].items()]
print("proposals"); [print("  ", x) for x in out["by_proposal_top20"]]

"""Build the candidates sub-page of the planet RSI tracker: <private-page-url><pages-slug>/candidates/ (2026-11-10; owned by the TRKSUB01 session, hooked into build_page.py by the <coordinator-session> session).
Inputs: rsi/candidates/candidates_status.json (hand-kept bodies, aliases, notes), rsi/candidates/novelty_queue/RESULTS.md (catalogue checks) and
rsi/candidates/confirm_queue/RESULTS.md (other-image checks). Only categories are taken from the two tables: no positions, dates, image ids or
names of catalogued bodies reach the page (it is semi-public). Writes index.html only when the content changes (key in out/.candidates_key).
Standard library only (the rk job runs system python3). Usage: python3 -I build_candidates.py [--out PATH]"""
import collections, hashlib, html, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); CAND = os.path.join(os.path.dirname(HERE), "candidates")
OUT = os.path.expanduser("$PAGES_OUT/candidates/index.html")
KEY = os.path.join(HERE, "out", ".candidates_key")
ID = re.compile(r"^r(\d+)-gen(\d+)-(F\d+)-p([0-9.]+)-r([0-9.]+)$")
E = html.escape


def rows(path):
    out = []
    for l in open(path):
        if l.startswith("| 20"):
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            if len(c) >= 3:
                out.append(c)
    return out


def cat_check(v):
    if v.startswith("OURS"):
        return "our own filing"
    if v.startswith(("KNOWN", "PROBABLY")):
        return "a known body"
    if v.startswith("NOT IN MPC"):
        return "not in the catalogues"
    if v.startswith("CHECK BY HAND"):
        return "near a catalogued body"
    return None


def cat_conf(v):
    if v.startswith("SEEN IN OTHER IMAGES"):
        return "seen in other images"
    if v.startswith("NO DECISIVE DATA"):
        return "no decisive data"
    if v.startswith("NOT SEEN"):
        return "not seen"
    return None


def collect(S):
    G = collections.OrderedDict(); of = {}; known = {}  # of: queue id -> row key, so a repeat joins its body's row
    for c in rows(os.path.join(CAND, "novelty_queue", "RESULTS.md")):
        m = ID.match(c[1])
        if not m:
            continue
        run, gen, f, p, r = m.groups(); k = S["aliases"].get(f"{f}:{p}", f"{f}:{p}")  # one row per body
        sb = re.search(r"same body as (\S+?)[,)]", c[2]); ours = re.match(r"OURS (\S+)", c[2]); kn = re.match(r"(?:KNOWN|PROBABLY) (\S+ \S+)", c[2])
        if f"{f}:{p}" not in S["aliases"]:
            if sb and sb.group(1) in of:
                k = of[sb.group(1)]
            elif ours and any(b["key"] == ours.group(1) for b in S["bodies"]):
                k = ours.group(1)
            elif kn:
                k = known.setdefault(kn.group(1), k)  # the designation is used only to group, never shown
        if kn:
            known.setdefault(kn.group(1), k)
        of[c[1]] = k
        g = G.setdefault(k, dict(field=f, p=float(p), r=set(), runs=set(), gens=set(), check=[], conf=[]))
        g["p"] = max(g["p"], float(p)); g["r"].add(float(r)); g["runs"].add(int(run)); g["gens"].add(int(gen)); x = cat_check(c[2])
        if x:
            g["check"].append(x)
    for c in rows(os.path.join(CAND, "confirm_queue", "RESULTS.md")):
        m = ID.match(c[1].split(" ")[0])
        if not m:
            continue
        run, gen, f, p, r = m.groups(); k = of.get(c[1].split(" ")[0], S["aliases"].get(f"{f}:{p}", f"{f}:{p}")); x = cat_conf(c[2])
        if k in G and x:
            G[k]["conf"].append(x)
    by = {b["key"]: b for b in S["bodies"]}; leads = []
    for k, g in G.items():
        body = by.get(k)
        chk = g["check"][-1] if g["check"] else "—"
        if "not in the catalogues" in g["check"] and chk == "near a catalogued body":
            chk = "not in the catalogues"
        conf = "seen in other images" if "seen in other images" in g["conf"] else (g["conf"][-1] if g["conf"] else "not sent")
        if body:
            stage, status = body["stage"], body["status"]; conf = body.get("conf", conf)
        elif chk == "a known body":
            stage, status = "known", "A known body"
        elif conf == "seen in other images":
            stage, status = "seen", "Seen in other images"
        else:
            stage, status = "lead", ("Unconfirmed: " + conf) if conf != "not sent" else "Unconfirmed: catalogue check only"
        leads.append(dict(key=k, field=g["field"], p=g["p"], r=sorted(g["r"]), runs=sorted(g["runs"]), gens=sorted(g["gens"]), check=chk, conf=conf,
                          stage=stage, status=status, title=body["title"] if body else f"{g['field']} r{sorted(g['r'])[-1]:.1f}", note=S["lead_notes"].get(k, "")))
    return leads


CSS = """
:root { --bg:#f5f6f8; --panel:#fff; --fg:#14171c; --muted:#5a6371; --line:#dde2e9; --accent:#2a78d6;
  --new:#1a7f4b; --newbg:#e6f4ec; --filed:#2a78d6; --filedbg:#e8f1fc; --seen:#a3620a; --seenbg:#fdf2e1; --known:#6b7280; --knownbg:#eef0f3; --lead:#5a6371; --leadbg:#f1f3f6;
  --serif:"Source Serif 4", Georgia, serif; --sans:"IBM Plex Sans", -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; --mono:"IBM Plex Mono", ui-monospace, Menlo, monospace; color-scheme:light; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0d1116; --panel:#151b23; --fg:#e7ebf1; --muted:#9aa3b1; --line:#263040; --accent:#3987e5;
  --new:#5cc98d; --newbg:#12291d; --filed:#6aa8f0; --filedbg:#13243a; --seen:#f0b45a; --seenbg:#2b2112; --known:#a2a9b5; --knownbg:#1d232c; --lead:#9aa3b1; --leadbg:#1a2029; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#0d1116; --panel:#151b23; --fg:#e7ebf1; --muted:#9aa3b1; --line:#263040; --accent:#3987e5;
  --new:#5cc98d; --newbg:#12291d; --filed:#6aa8f0; --filedbg:#13243a; --seen:#f0b45a; --seenbg:#2b2112; --known:#a2a9b5; --knownbg:#1d232c; --lead:#9aa3b1; --leadbg:#1a2029; color-scheme:dark; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 var(--sans); }
a { color:var(--accent); }
.wrap { max-width:60rem; margin-inline:auto; padding-block:28px 64px; padding-left:max(16px, env(safe-area-inset-left,0px)); padding-right:max(16px, env(safe-area-inset-right,0px)); display:grid; gap:40px; }
header { display:grid; gap:10px; }
.back { font-size:13px; }
.eyebrow { font:500 12px/1.4 var(--mono); letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
h1 { font:600 clamp(28px,5.5vw,40px)/1.15 var(--serif); margin:0; text-wrap:balance; }
.take { margin:0; font-size:17px; max-width:62ch; }
.updated { font-size:13px; color:var(--muted); }
h2 { margin:0; font:600 22px/1.3 var(--serif); }
section { display:grid; gap:14px; }
.intro { margin:0; color:var(--muted); max-width:64ch; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 10.5rem), 1fr)); gap:10px; }
.tile { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; display:grid; gap:2px; }
.tile .n { font:400 30px/1.1 var(--serif); font-variant-numeric:tabular-nums; }
.tile .l { font-size:13px; color:var(--muted); line-height:1.4; }
.card { background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--c, var(--line)); border-radius:10px; padding:18px 20px; display:grid; gap:12px; }
.card.new { --c:var(--new); } .card.filed { --c:var(--filed); } .card.seen { --c:var(--seen); } .card.known { --c:var(--known); }
.top { display:flex; flex-wrap:wrap; align-items:baseline; gap:6px 14px; }
.top h3 { margin:0; font:600 21px/1.25 var(--serif); }
.pill { display:inline-block; font:500 12.5px/1.2 var(--sans); padding:4px 10px; border-radius:999px; background:var(--pb); color:var(--pc); white-space:nowrap; }
.pill.new { --pb:var(--newbg); --pc:var(--new); } .pill.filed { --pb:var(--filedbg); --pc:var(--filed); } .pill.seen { --pb:var(--seenbg); --pc:var(--seen); }
.pill.known { --pb:var(--knownbg); --pc:var(--known); } .pill.lead { --pb:var(--leadbg); --pc:var(--lead); }
.sum { margin:0; font-size:16px; max-width:62ch; }
.by { margin:0; font-size:13px; color:var(--muted); }
dl.facts { margin:0; display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 13rem), 1fr)); gap:8px 18px; }
dl.facts div { display:grid; gap:0; }
dl.facts dt { font-size:12.5px; color:var(--muted); }
dl.facts dd { margin:0; font-variant-numeric:tabular-nums; }
ul.tests { margin:0; padding:0; list-style:none; display:grid; gap:8px; }
ul.tests li { display:grid; grid-template-columns:1.5rem 1fr; gap:0 6px; }
ul.tests .mk { font-weight:600; text-align:center; }
ul.tests .ok { color:var(--new); } ul.tests .no { color:#c2410c; } ul.tests .wait { color:var(--seen); }
ul.tests b { font-weight:600; }
ul.tests span.d { color:var(--muted); }
.next { margin:0; font-size:13.5px; color:var(--muted); }
.steps { margin:0; padding:0; list-style:none; display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 10.5rem), 1fr)); gap:10px; counter-reset:s; }
.steps li { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; counter-increment:s; font-size:14px; }
.steps li::before { content:counter(s); display:block; font:600 13px var(--mono); color:var(--accent); margin-bottom:2px; }
.steps b { display:block; }
.tw { overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--panel); }
table { border-collapse:collapse; width:100%; min-width:40rem; font-size:14px; }
th, td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
th { font:500 12.5px var(--sans); color:var(--muted); background:var(--panel); position:sticky; top:0; }
tr:last-child td { border-bottom:0; }
td.note { color:var(--muted); font-size:13px; max-width:28rem; }
td .id { font-family:var(--mono); font-size:13px; }
.foot { font-size:13px; color:var(--muted); max-width:64ch; margin:0; }
"""


def card(b):
    mk = {True: ("ok", "✓"), False: ("no", "✗"), None: ("wait", "…")}
    tests = "".join(f'<li><span class="mk {mk[t[1]][0]}">{mk[t[1]][1]}</span><span><b>{E(t[0])}.</b> <span class="d">{E(t[2])}</span></span></li>' for t in b.get("tests", []))
    facts = "".join(f"<div><dt>{E(k)}</dt><dd>{E(v)}</dd></div>" for k, v in b.get("facts", []))
    nxt = f'<p class="next">Next: {E(b["next"])}</p>' if b.get("next") else ""
    return (f'<article class="card {b["stage"]}"><div class="top"><h3>{E(b["title"])}</h3><span class="pill {b["stage"]}">{E(b["status"])}</span></div>'
            f'<p class="sum">{E(b["summary"])}</p><p class="by">Field {E(b["field"])} · found by {E(b["found_by"])}</p>'
            f'<dl class="facts">{facts}</dl><ul class="tests">{tests}</ul>{nxt}</article>')


def build():
    S = json.load(open(os.path.join(CAND, "candidates_status.json"))); leads = collect(S); B = S["bodies"]
    new = [b for b in B if b["stage"] in ("filed", "new")]; seen = [b for b in B if b["stage"] == "seen"]; known = [b for b in B if b["stage"] == "known"]
    n_leads = len(leads); n_known = sum(1 for l in leads if l["stage"] == "known"); n_seen = len({l["title"] for l in leads if l["stage"] in ("seen", "new", "filed")} | {b["title"] for b in new + seen})
    tiles = [(n_leads, "run-2 leads checked against the catalogues"), (n_known, "turned out to be known bodies"), (n_seen, "seen in images their program never used"),
             (len(new), "confirmed new (followed over years)"), (sum(b["stage"] == "filed" for b in B), "filed, awaiting confirmation")]
    order = {"filed": 0, "new": 1, "seen": 2, "known": 3, "lead": 4}
    trs = "".join(f'<tr><td><span class="id">{E(l["title"])}</span></td><td>{", ".join("run " + str(x) for x in l["runs"])}<br><span class="id">rounds {", ".join(str(g) for g in l["gens"])}</span></td>'
                  f'<td>{E(l["check"])}</td><td>{E(l["conf"])}</td><td><span class="pill {l["stage"]}">{E(l["status"])}</span></td><td class="note">{E(l["note"])}</td></tr>'
                  for l in sorted(leads, key=lambda l: (order[l["stage"]], l["field"], -l["p"])))
    steps = [("Found", "a run's program reports a moving body it believes is real"), ("Catalogue check", "one orbit fit against every catalogued body and unlinked tracklet nearby"),
             ("Other images", "the predicted place measured in images the program never saw, with fake tracks as a control"),
             ("Other years", "the orbit from one season predicts another; the body must be there"), ("Filed", "positions reported; an official confirmation follows")]
    body = (f'<header><div class="back"><a href="../">← Planet RSI tracker</a></div><div class="eyebrow">Planet RSI · candidates</div>'
            f'<h1>What the runs have found</h1><p class="take">{len(new)} bodies are confirmed new; {sum(b["stage"] == "filed" for b in B)} of them are filed. '
            f'Below: each confirmed body with how we know, then every lead the runs reported and where it stands.</p></header>'
            f'<section><div class="tiles">' + "".join(f'<div class="tile"><span class="n">{n}</span><span class="l">{E(t)}</span></div>' for n, t in tiles) + '</div></section>'
            f'<section><h2>Confirmed new</h2><p class="intro">Seen again in other images and in other years, on one orbit, and not in any catalogue.</p>' + "".join(card(b) for b in new) + '</section>'
            + (f'<section><h2>Seen in other images</h2><p class="intro">Real moving sources whose orbits are not yet secure enough to call new.</p>' + "".join(card(b) for b in seen) + '</section>' if seen else "")
            + (f'<section><h2>Known bodies</h2><p class="intro">Real, but already catalogued.</p>' + "".join(card(b) for b in known) + '</section>' if known else "")
            + f'<section><h2>How a lead becomes a discovery</h2><ol class="steps">' + "".join(f"<li><b>{E(a)}</b>{E(d)}</li>" for a, d in steps) + '</ol></section>'
            + f'<section><h2>Every lead checked</h2><p class="intro">One row per body (a body reported in several rounds appears once). "r" is the program\'s first distance guess in au.</p>'
            f'<div class="tw"><table><thead><tr><th>Lead</th><th>Reported by</th><th>Catalogue check</th><th>Other images</th><th>Where it stands</th><th>Note</th></tr></thead><tbody>{trs}</tbody></table></div></section>'
            '<p class="foot">Positions, dates and image details of bodies not yet filed are left off this page, as are the names of catalogued bodies in the task\'s fields.</p>')
    return body


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else OUT
    body = build(); key = hashlib.sha256((body + CSS).encode()).hexdigest()
    real = out == OUT  # a test render (--out) neither reads nor writes the key
    if real and os.path.exists(out) and os.path.exists(KEY) and open(KEY).read() == key:
        print("candidates unchanged"); return
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<title>Planet RSI Candidates</title><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">'
            f'<style>{CSS}</style></head><body><main class="wrap">{body}<p class="updated">Updated {time.strftime("%Y-%m-%d %H:%M %Z")}</p></main></body></html>')
    os.makedirs(os.path.dirname(out), exist_ok=True); tmp = out + ".tmp"; open(tmp, "w").write(page); os.replace(tmp, out)
    if real:
        os.makedirs(os.path.dirname(KEY), exist_ok=True); open(KEY, "w").write(key)
    print("candidates written", out)


if __name__ == "__main__":
    main()

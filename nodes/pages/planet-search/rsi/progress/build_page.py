"""Build the RSI progress tracker for <private-page-url> refresh out/*.json (make_progress.py), embed them in
page.html, and write $PAGES_OUT/index.html only when the data changed
(the 'updated' stamp alone does not count). Run by $PAGE_PUBLISHER (job <publisher-job>), which publishes.

Usage: python3 -I build_page.py [--out PATH]
"""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser("$PAGES_OUT/index.html")
NAMES = ("generations", "activity", "fields", "truth", "downloads", "setup", "events", "runs", "compare", "spotlight")


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else OUT
    spec = importlib.util.spec_from_file_location("make_progress", os.path.join(HERE, "make_progress.py"))
    mp = importlib.util.module_from_spec(spec); spec.loader.exec_module(mp)
    mp.main()
    # The trace reader is part of this page. It rebuilds only finished rounds that changed; a failure there must not stop the tracker.
    try:
        spec = importlib.util.spec_from_file_location("build_traces", os.path.join(HERE, "build_traces.py"))
        bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)
        bt.main()
    except (Exception, SystemExit) as e:
        print("traces not rebuilt:", e)
    # The score breakdown sub-page (scores/), rewritten only when a round's grade or a verdict changed.
    try:
        spec = importlib.util.spec_from_file_location("build_scores", os.path.join(HERE, "build_scores.py"))
        bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
        bs.main()
    except (Exception, SystemExit) as e:
        print("scores not rebuilt:", e)
    # The candidates sub-page (candidates/), written by build_candidates.py (owned by the TRKSUB01 session, 2026-11-10);
    # skipped while that file does not exist, and a failure there must not stop the tracker.
    if os.path.exists(os.path.join(HERE, "build_candidates.py")):
        try:
            spec = importlib.util.spec_from_file_location("build_candidates", os.path.join(HERE, "build_candidates.py"))
            bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)
            bc.main()
        except (Exception, SystemExit) as e:
            print("candidates not rebuilt:", e)
    data ={n: json.load(open(os.path.join(HERE, "out", n + ".json"))) for n in NAMES}
    key = json.dumps({**data, "setup": [{k: v for k, v in data["setup"][0].items() if k != "updated"}]}, sort_keys=True)
    stamp = os.path.join(HERE, "out", ".page_key")
    tpl = open(os.path.join(HERE, "page.html")).read()
    if os.path.exists(out) and os.path.exists(stamp) and open(stamp).read() == key and os.path.getmtime(out) > os.path.getmtime(os.path.join(HERE, "page.html")):
        print("unchanged"); return
    html = tpl.replace("/*RSI_DATA*/", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    open(tmp, "w").write(html); os.replace(tmp, out)
    open(stamp, "w").write(key)
    print("wrote", out, len(html))


if __name__ == "__main__":
    main()

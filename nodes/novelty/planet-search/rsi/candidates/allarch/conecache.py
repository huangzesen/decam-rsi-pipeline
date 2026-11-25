"""Legacy Survey cone cache dr10_cones.json shared by nsc_path.py, wide_search.py and verdicts.py (2026-11-19: several processes wrote
it at once and readers hit half-written files). Load retries; writes merge with what is on disk, go to a temporary file and are
renamed into place, and happen every 20 new cones and at exit."""
import atexit, json, os, time
_new = 0


def load(path):
    for a in range(10):
        try:
            return json.load(open(path)) if os.path.exists(path) else []
        except (ValueError, OSError):
            time.sleep(2)
    raise RuntimeError("cone cache unreadable: " + path)


def save(path, cones, force=False):
    global _new
    _new += 0 if force else 1
    if not force and _new % 20:
        return
    try:
        disk = load(path)
    except RuntimeError:
        disk = []
    seen = {(round(c["ra"], 6), round(c["dec"], 6)) for c in cones}
    merged = cones + [c for c in disk if (round(c["ra"], 6), round(c["dec"], 6)) not in seen]
    tmp = f"{path}.tmp{os.getpid()}"; json.dump(merged, open(tmp, "w")); os.replace(tmp, path)


def register(path, cones):
    atexit.register(lambda: save(path, cones, force=True))


def put(path, key, value):
    """Result files shared by parallel runs (nsc_path.json, nsc_pairs.json, wide_*.json): re-read, set one key, write to a
    temporary file and rename, so one run never drops another's entries."""
    d = {}
    for a in range(10):
        try:
            d = json.load(open(path)) if os.path.exists(path) else {}
            break
        except ValueError:
            time.sleep(1)
    else:
        raise RuntimeError("result file unreadable: " + path)
    d[key] = value; tmp = f"{path}.tmp{os.getpid()}"; json.dump(d, open(tmp, "w"), indent=0); os.replace(tmp, path)
    return d

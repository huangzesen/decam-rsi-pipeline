"""Track solar orbits through a partition of the two-endpoint admissible region."""
import ctypes
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from companion_graph import build_library
from survey_arcs import certify

PAIR = np.dtype([("a", "i4"), ("b", "i4")] +
                [(s, "f4") for s in ("g", "sg", "h", "sh", "score")])
SEED = np.dtype([("rows", "i4", 5), ("g", "f4"), ("score", "f4")])
P = ctypes.c_void_p
I = ctypes.c_int
D = ctypes.c_double


def balanced(score, ex, cap):
    """Mix uniform-within-shutter ranks with the absolute point-source odds."""
    if len(score) <= cap:
        return np.argsort(-score, kind="stable")
    order = np.lexsort((-score, ex))
    _, first, count = np.unique(ex[order], return_index=True, return_counts=True)
    rank = np.arange(len(order)) - np.repeat(first, count)
    value = score[order] - .65*np.log1p(rank/32.)
    sel = np.argpartition(value, -cap)[-cap:]
    return order[sel[np.argsort(-value[sel], kind="stable")]]


class Bank:
    def __init__(self, index, visits, signal, ids, priority, repeat, companions, depths, lever, east):
        lib = build_library("return_cells")
        self.lib = lib
        lib.return_bank.argtypes = [P, I, P, I, P, P, P, P, P, P, I, D, D]
        lib.return_bank.restype = P
        lib.release_return_bank.argtypes = [P]
        lib.release_return_bank.restype = None
        lib.return_pairs.argtypes = [P, P, I, I, I, D, I, P, P]
        lib.return_pairs.restype = I
        lib.return_search.argtypes = [P, P, I, I, D, I, P, P]
        lib.return_search.restype = I
        lib.return_three.argtypes = [P, P, I, I, D, I, P, P, P]
        lib.return_three.restype = I
        target = index.target
        q = np.full(len(target), -5., "f4")
        peer = np.full(len(target), -1, "i4")
        priority = np.nan_to_num(np.asarray(priority, "f4"), nan=-12.)
        priority = np.clip(priority-np.median(priority), -9., 7.)
        valid = (companions >= 0) & (repeat[:, 0] > .35)
        peer[ids[valid]] = companions[valid]
        # Coincident shutters can confirm a transient even when its motion is
        # unresolved. Its stationary-source evidence is in the input priority.
        score = priority + .83*np.minimum(repeat[:, 0], 6.) + .28*np.clip(repeat[:, 3], -2., 4.)
        q[ids] = score
        ex = target["exposure"][ids]
        select = balanced(score, ex, min(410000, max(15000, len(ids)//4)))
        self.anchors = ids[select].astype("i4")
        # Keep both well-measured single shutters and repeated point sources.
        extra = np.flatnonzero(valid)
        keep = np.union1d(select, extra[balanced(score[extra], ex[extra], 250000)])
        endpoints = np.ascontiguousarray(ids[keep], "i4")
        signal = np.ascontiguousarray(signal, "f4")
        lever = np.ascontiguousarray(lever, "f8")
        depths = np.ascontiguousarray(np.where(np.isfinite(depths), depths, -40.), "f8")
        self.storage = (target, visits, signal, q, peer, depths, lever)
        self.handle = lib.return_bank(target.ctypes.data, len(target), visits.ctypes.data, len(visits),
                    signal.ctypes.data, q.ctypes.data, peer.ctypes.data, depths.ctypes.data,
                    lever.ctypes.data, endpoints.ctypes.data, len(endpoints), *map(float, east))
        if not self.handle:
            raise MemoryError("could not build the admissible-region catalogue")

    def close(self):
        if self.handle:
            self.lib.release_return_bank(self.handle)
            self.handle = None

    def pairs(self, anchors, family, threads, seconds, cap):
        out = np.empty(cap, PAIR)
        tested = ctypes.c_int()
        anchors = np.ascontiguousarray(anchors, "i4")
        n = self.lib.return_pairs(self.handle, anchors.ctypes.data, len(anchors), family,
                threads, seconds, cap, out.ctypes.data, ctypes.byref(tested))
        if n < 0:
            raise RuntimeError("two-visit ranging failed")
        return out[:n].copy(), tested.value

    def search(self, pairs, threads, seconds, cap):
        out = np.empty(cap, SEED)
        counts = np.zeros(6, "i8")
        pairs = np.ascontiguousarray(pairs, PAIR)
        n = self.lib.return_search(self.handle, pairs.ctypes.data, len(pairs), threads,
                    seconds, cap, out.ctypes.data, counts.ctypes.data)
        if n < 0:
            raise RuntimeError("range-cell intersections failed")
        return out[:n].copy(), counts

    def extend(self, triples, threads, seconds, cap, cores=None):
        from shutter_ranging import TRIPLE
        triples = np.ascontiguousarray(triples, TRIPLE)
        out = np.empty(cap, SEED)
        tested = ctypes.c_int()
        n = self.lib.return_three(self.handle, triples.ctypes.data, len(triples), threads,
                    seconds, cap, out.ctypes.data, ctypes.byref(tested),cores.ctypes.data if cores is not None else None)
        if n < 0:
            raise RuntimeError("predicting returns from three visits failed")
        return out[:n].copy(), tested.value


def distinct(pairs):
    if not len(pairs):
        return pairs
    order = np.argsort(-pairs["score"], kind="stable")
    a, b = pairs["a"][order], pairs["b"][order]
    key = (np.minimum(a, b).astype("i8") << 32) | np.maximum(a, b)
    first = np.unique(key, return_index=True)[1]
    return pairs[order[np.sort(first)]]


def work(index, visits, signal, ids, priority, repeat, peers, depths, lever, east,
         control, threads, seconds):
    started = time.monotonic()
    end = started + seconds
    stop = control["event"]
    if stop.is_set() or not len(ids):
        return
    bank = Bank(index, visits, signal, ids, priority, repeat, peers, depths, lever, east)
    try:
        device = control.get('shutter_device', 'cpu')
        try:
            import shutter_ranging
            now = time.monotonic()
            share = .70 if device == 'cuda' else .42
            shutter_ranging.scan(bank, index, ids, east, control, threads,
                                 now + share * (end - now), device)
        except (RuntimeError, ArithmeticError, TypeError, ValueError, OSError, MemoryError) as exc:
            print("measured shutter graph:", str(exc)[:200], flush=True)
        started = time.monotonic()
        seconds = end - started
        for family, share in enumerate((.50, .79, 1.)):
            left = started + seconds*share-time.monotonic()
            if stop.is_set() or left < 2.:
                continue
            pairs, tested = bank.pairs(bank.anchors, family, threads, max(.2, .26*left), 145000)
            pairs = distinct(pairs)
            if stop.is_set() or not len(pairs):
                continue
            batch = 0
            deadline = started + seconds*share
            while batch < len(pairs) and time.monotonic()+1. < deadline and not stop.is_set():
                room = min(25., deadline-time.monotonic())
                found, stats = bank.search(pairs[batch:batch+4500], threads, .65*room, 18000)
                batch += int(stats[0])
                if len(found) and not stop.is_set():
                    order = np.argsort(-found["score"], kind="stable")
                    rows = found["rows"][order]
                    orbits = certify(index, rows[:, :4], found["g"][order], rows[:, 4],
                                     threads, max(.25, min(.35*room, end-time.monotonic())), 3200)
                    control["certified"].put(orbits)
                if not stats[0]:
                    break
            print("two-visit return cells:", family, tested, "anchors,", batch,
                  "pairs searched of", len(pairs), flush=True)
    finally:
        bank.close()


def begin(*args):
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="admissible-orbits")
    future = pool.submit(work, *args)
    def done(job):
        if job.exception() is not None:
            print("two-visit return cells:", repr(job.exception())[:200], flush=True)
    future.add_done_callback(done)
    pool.shutdown(wait=False)
    return future

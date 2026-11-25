"""Four-position orbit search using exposure pairs selected from live silicon."""
import ctypes
import time
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from attributables import SEED
from companion_graph import VISIT, build_library
import quartet_geometry as geometry
import quartet_footprints


POINT = np.dtype([(k, 'f4') for k in ('x', 'y', 'mag', 'sigma', 'quality')]+
                 [('row', 'i4'),('rawx','f4'),('rawy','f4'),('exposure','i4')])
PAIR = np.dtype([('pos', 'f4', 4), ('mag', 'f4'), ('variance', 'f4'), ('quality', 'f4'),
                 ('a', 'i4'), ('b', 'i4'),('native','f4',4),('e0','i4'),('e1','i4')])
ptr, integer, real = ctypes.c_void_p, ctypes.c_int, ctypes.c_double


class PairIndex:
    def __init__(self, target, ids, q, visits, signal, limit=3200000, device='cpu', pole=None):
        ex = target['exposure'][ids]
        good = np.isfinite(q+target['mag'][ids]+target['x'][ids]+target['y'][ids])
        ids, q, ex = ids[good], np.asarray(q[good], 'f4'), ex[good]
        if len(ids) > limit:
            # Apply part of the cutoff within each depth and visit. This
            # retains the best faint sources in imaging with a crowded sky.
            mb = np.clip(np.floor(2.*(target['mag'][ids]-12.)).astype('i4'), 0, 31)
            group = ex*32+mb
            b = np.clip(((q-np.median(q)+9.)*3).astype('i4'), 0, 78)
            h = np.bincount(group*79+b, minlength=32*len(visits)*79).reshape(-1, 79)
            tail = np.cumsum(h[:, ::-1], axis=1)[:, ::-1]
            rarity = -np.log((tail[group, b]-.48*h[group, b]+1.)/(tail[group, 0]+2.))
            rank = q+.85*np.minimum(4., rarity)
            take = np.argpartition(-rank, limit-1)[:limit]
            ids, ex, q = ids[take], ex[take], q[take]
        order = np.argsort(ex, kind='stable')
        ids, ex, q = ids[order], ex[order], q[order]
        xy = np.column_stack((target['x'][ids], target['y'][ids]))
        self.xy, self.ex = xy, ex
        ne = len(visits)
        edges = np.searchsorted(ex, np.arange(ne+1)).astype('i4')
        counts = np.maximum(1, np.diff(edges))
        self.center = np.column_stack([np.bincount(ex, weights=xy[:, i], minlength=ne)/counts
                                       for i in range(2)])
        self.actual_visits = visits
        self.fields = None
        original, true_ex = xy, ex
        self.planning_xy, self.planning_ex = xy, ex
        if pole is not None:
            from nightly_fields import NightFields,merge_measurements
            self.fields = NightFields(visits,self.center,counts,pole)
            group = self.fields.group[ex]
            corrected = self.fields.correct(xy,ex)
            keep, q = merge_measurements(corrected,ex,group,ids,q,target['mag'][ids],signal[ids])
            order=np.argsort(group[keep],kind='stable')
            q=q[order];keep=keep[order]
            ids,ex,xy,original,true_ex=ids[keep],group[keep],corrected[keep],original[keep],true_ex[keep]
            visits=self.fields.visits;ne=len(visits)
            edges=np.searchsorted(ex,np.arange(ne+1)).astype('i4')
            counts=np.maximum(1,np.diff(edges))
            self.center=np.column_stack([np.bincount(ex,weights=xy[:,i],minlength=ne)/counts for i in range(2)])
            self.xy,self.ex=xy,ex
        point = np.empty(len(ids), POINT)
        point['x'], point['y'] = (xy/geometry.AS).T
        point['rawx'],point['rawy']=(original/geometry.AS).T
        point['exposure']=true_ex
        point['row'], point['mag'] = ids, target['mag'][ids]
        point['sigma'] = 1.086/np.maximum(4.5, signal[ids])
        point['quality'] = np.clip(q-np.median(q), -8., 7.)
        self.footprints=quartet_footprints.Footprints(point,edges)
        self.visits = visits
        self.lib = build_library('quartet_join')
        lib = self.lib
        lib.quartet_field.argtypes = [ptr, ptr, integer]
        lib.quartet_field.restype = ptr
        lib.quartet_free.argtypes = [ptr]
        lib.quartet_free.restype = None
        self.handle = lib.quartet_field(point.ctypes.data, edges.ctypes.data, ne)
        if not self.handle:
            raise MemoryError('exposure-pair index allocation failed')
        weakref.finalize(self, lib.quartet_free, self.handle)
        lib.quartet_pairs.argtypes = [ptr, integer, integer, ptr, real, integer, ptr]
        lib.quartet_pairs.restype = integer
        lib.quartet_matches.argtypes = [ptr, integer, ptr, integer, ptr, ptr, real, real, integer, ptr, ptr]
        lib.quartet_matches.restype = integer
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.gpu = None
        if str(device).startswith('cuda'):
            try:
                import quartet_device
                self.gpu = quartet_device.Device(point, edges, visits, real_visits=self.actual_visits,
                                                 footprints=self.footprints)
                quartet_device.validate(self.gpu.kernels)
                print('four-position CUDA fixture verified', flush=True)
            except Exception as exc:
                self.gpu = None
                print('four-position host join:', str(exc)[:160], flush=True)

    def pairs(self, a, b, family, gate, cap):
        key = (int(a), int(b), int(family),cap)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        buffer = np.empty(cap, PAIR)
        color = self.visits['lever'][a]-self.visits['lever'][b]
        n = self.lib.quartet_pairs(self.handle, a, b, gate.ctypes.data, color, cap, buffer.ctypes.data)
        if n < 0:
            raise RuntimeError('could not enumerate exposure pair')
        data = buffer[:n].copy()
        self.cache_bytes += data.nbytes
        self.cache[key] = data
        while self.cache_bytes > 270000000 or len(self.cache)>10000:
            self.cache_bytes -= self.cache.popitem(last=False)[1].nbytes
        return data

    def matches(self, a, b, models, shutter, gamma, color, cap):
        buffer = np.empty(cap, SEED)
        stat = ctypes.c_longlong()
        n = self.lib.quartet_matches(a.ctypes.data, len(a), b.ctypes.data, len(b), models.ctypes.data,
                                     shutter.ctypes.data, gamma, color, cap, buffer.ctypes.data, ctypes.byref(stat))
        if n < 0:
            raise RuntimeError('four-position orbit query failed')
        return buffer[:n], stat.value


def signature_models(sample):
    ca, ta, pa, cb, tb, pb = geometry.relation(sample)
    n = len(ca)
    pack = lambda a, b, c: np.column_stack((a, b.reshape(n, -1), c.reshape(n, -1)))
    return np.ascontiguousarray(np.stack((pack(ca, ta, pa), pack(cb, tb, pb)), axis=1))


def search(index, pole, plans, deadline, event=None, pair_cap=5500, per_plan=160):
    z = geometry.quadrature(128)
    n = np.zeros(5, 'i8')
    visits = index.visits
    shutters = np.ascontiguousarray(np.column_stack([visits[k] for k in ('t','ox','oy','oz')]))
    actual_shutters = np.ascontiguousarray(np.column_stack([index.actual_visits[k] for k in ('t','ox','oy','oz')]))
    for at in range(0, len(plans), 32):
        if time.monotonic() >= deadline or (event is not None and event.is_set()):
            break
        batch = plans[at:at+32]
        samples = geometry.sample(visits, index.center, pole, batch, z)
        if index.fields is not None:
            samples=index.fields.uncertainty(samples,batch,z)
        maps = signature_models(samples)
        routes=quartet_footprints.routes(samples)
        a_gate = geometry.pair_gate(samples[..., :2, :])
        # Pair catalogues are cached by their own shutters and family. Use
        # the distribution at that pair's first shutter, in both roles.
        b_plans = batch[:, [2, 3, 2, 3, 4]]
        b_sample = geometry.sample(visits, index.center, pole, b_plans, z)
        if index.fields is not None:
            b_sample=index.fields.uncertainty(b_sample,b_plans,z)
        b_gate = geometry.pair_gate(b_sample[..., :2, :])
        pieces = []
        if index.gpu is not None:
            try:
                seeds,count = index.gpu.batch(batch,maps,a_gate,b_gate,index.center,shutters,24000,routes=routes)
                n += [len(batch),0,0,count,len(seeds)]
                if len(seeds):
                    yield seeds,n.copy()
                continue
            except (RuntimeError,MemoryError,ArithmeticError) as exc:
                print('four-position join continuing on host:',str(exc)[:180],flush=True)
                index.gpu=None
        for j, (a, b, c, d, family) in enumerate(batch):
            if time.monotonic() >= deadline or (event is not None and event.is_set()):
                break
            left = index.pairs(a, b, family, a_gate[j], 2*pair_cap)
            right = index.pairs(c, d, family, b_gate[j], 2*pair_cap)
            left=index.footprints.select(left,routes[j,0],(c,d),pair_cap)
            right=index.footprints.select(right,routes[j,1],(a,b),pair_cap)
            n[:3] += [1, len(left), len(right)]
            if not len(left) or not len(right):
                continue
            color = .5*(visits['lever'][a]+visits['lever'][b]-visits['lever'][c]-visits['lever'][d])
            g = np.sqrt(1.+index.center[a]@index.center[a])/geometry.FAMILIES[family, 0]
            seeds, hits = index.matches(left, right, maps[j], actual_shutters, g, color, per_plan)
            n[3:] += [hits, len(seeds)]
            if len(seeds):
                pieces.append(seeds)
        if pieces:
            yield np.concatenate(pieces), n.copy()


def endpoints(target, ids, quality, secondary):
    if secondary is None:
        return ids, quality
    _, other, value = secondary
    if not len(other):
        return ids, quality
    return np.r_[ids, other], np.r_[quality, value]


def predicted_cores(sky, seeds, rows, seconds):
    from original_recovery import CORE
    output=np.empty(len(seeds),CORE)
    tested=integer()
    call=sky.lib.quartet_predictions
    call.argtypes=[ptr,ptr,integer,ptr,integer,real,ptr,ptr]
    call.restype=integer
    count=call(sky.handle,seeds.ctypes.data,len(seeds),None if rows is None else rows.ctypes.data,
               len(sky.mag) if rows is None else len(rows),seconds,
               output.ctypes.data,ctypes.byref(tested))
    if count<0:
        raise RuntimeError('four-position prediction failed')
    cores=output[:count]
    return cores[np.argsort(-cores['merit'],kind='stable')],tested.value


def work(index, ids, q, secondary, rawsn, rowmap, ex, depth, lever, cov, pole, seconds, control, device):
    import attributables
    start = time.monotonic()
    end = start+seconds
    visits = np.zeros(len(ex), VISIT)
    for k in ('t', 'night', 'ox', 'oy', 'oz'):
        visits[k] = ex[k]
    visits['depth'], visits['lever'] = depth, lever
    ids, quality = endpoints(index.target, ids, q, secondary)
    if len(ids)<5 or control['event'].is_set():
        return
    scanner = PairIndex(index.target, ids, quality, visits, rawsn, device=device,pole=pole)
    plans = geometry.schedule(scanner.planning_xy, scanner.planning_ex, visits, cov, pole,
                              28672, min(end, start+max(5., .16*seconds)), group=scanner.fields.group)
    if not len(plans) or control['event'].is_set():
        return
    print('four-visit schedule:', len(plans), 'exposure quartets,', len(scanner.ex), 'sources', flush=True)
    if control.get('original_sky') is not None and end-time.monotonic()>33.:
        import chord_returns
        control['raw_chord_job']=chord_returns.begin(scanner,control['original_sky'],rowmap,plans,pole,end,control)
    bank = []
    total = 0
    last = time.monotonic()
    counts = np.zeros(5, 'i8')
    def flush(budget):
        nonlocal bank, total, last
        seeds = np.concatenate(bank)
        order = np.argsort(-seeds['score'], kind='stable')
        keys = np.sort(seeds['rows'][order], axis=1)
        _, first = np.unique(keys, axis=0, return_index=True)
        seeds = seeds[order[np.sort(first)]]
        sky=control.get('original_sky')
        tested=raw_fits=0
        if sky is not None:
            cores,tested=predicted_cores(sky,seeds,rowmap,.31*budget)
            if len(cores):
                cores['merit']=np.clip(.27*cores['merit'],-2.,5.)
                result,_=sky.recover(cores,.25*budget,cap=1500,measured=True)
                control['raw_results'].put((result,[]))
                raw_fits=len(result)
        fits = attributables.follow(index, seeds, 1, budget*(.44 if sky is not None else 1.), cap=4300)
        five = fits[fits['nights']>=5]
        control['certified'].put(five)
        if len(fits)>len(five):
            four = fits[fits['nights']==4]
            control['certified'].put(four[np.argsort(-four['merit'])[:400]])
        print('paired exposure search:', counts.tolist(), 'verified', len(five),
              'original catalogue',raw_fits,'from',tested,'predictions',flush=True)
        bank, total = [], 0
        last=time.monotonic()
    for piece, counts in search(scanner, pole, plans, end-6., control['event']):
        bank.append(piece)
        total += len(piece)
        elapsed=time.monotonic()-last
        if (total>=10000 or elapsed>=17.) and time.monotonic()+1.<end and not control['event'].is_set():
            flush(min(8.,max(1.5,.23*elapsed),max(.3,end-time.monotonic())))
    if bank and not control['event'].is_set():
        flush(min(8.,max(.3, end-time.monotonic())))


def begin(*args):
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='exposure-quartets')
    job = pool.submit(work, *args)
    def done(future):
        exc = future.exception()
        if exc is not None:
            print('paired exposure search:', repr(exc)[:220], flush=True)
    job.add_done_callback(done)
    pool.shutdown(wait=False)
    return job

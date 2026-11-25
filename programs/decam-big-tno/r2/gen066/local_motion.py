"""Resolve short apparent-motion arcs using independent observing seasons."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import time

import numpy as np

import attributables


def proposals(index, quality, depth, color, east, family, seconds, threads, cap=140000):
    ptr, integer, real = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    fn = index.lib.local_motion_seeds
    fn.argtypes = [ptr]*5+[real]*4+[integer,integer,real,integer,ptr,ptr]
    fn.restype = integer
    inputs = [np.ascontiguousarray(quality, 'f4')]
    inputs.extend(np.ascontiguousarray(v,'f8') for v in (depth,color,east))
    seeds = np.empty(cap,attributables.SEED)
    count = np.zeros(5,'i8')
    n = fn(index.handle, *(a.ctypes.data for a in inputs), *family, 16500,
           threads, seconds, cap, seeds.ctypes.data, count.ctypes.data)
    if n < 0:
        raise RuntimeError('short motion proposal search failed')
    return seeds[:n].copy(), count


def varied(seeds, exposure, night, time_e):
    if not len(seeds):
        return seeds
    rows = seeds['rows']
    valid = rows>=0
    e = exposure[rows.clip(min=0)]
    nights = np.where(valid,night[e],-1)
    nights.sort(1)
    count = 1+(nights[:,1:]!=nights[:,:-1]).sum(1)-(nights[:,0]<0)
    span = np.max(np.where(valid,time_e[e],-1.e9),axis=1)-np.min(np.where(valid,time_e[e],1.e9),axis=1)
    priority = seeds['score'] + .8*np.log1p(span) + 1.4*(count-2)
    groups = []
    for n in (2,3,4):
        selected = np.flatnonzero(count==n)
        if not len(selected):
            continue
        order = selected[np.argsort(-priority[selected],kind='stable')]
        # A bright star with a noisy halo or a densely observed real mover
        # can create thousands of nearly identical angular rates. Give the
        # other exposures a turn before revisiting a busy endpoint.
        seen = {}
        cost = np.empty(len(order))
        for j, i in enumerate(order):
            ids = rows[i,valid[i]].tolist()
            load = sorted([seen.get(d,0) for d in ids])[-2]
            cost[j] = j*(1.+min(load,250)*.35)
            if load<250:
                for d in ids:
                    seen[d] = seen.get(d,0)+1
        order = order[np.argsort(cost,kind='stable')]
        groups.append((order,{2:.17,3:.52,4:.31}[n]))
    if not groups:
        return seeds[:0]
    ranks = np.concatenate([np.arange(len(order))/share for order,share in groups])
    return seeds[np.concatenate([v[0] for v in groups])[np.argsort(ranks,kind='stable')]]


def work(index, ids, base, G, secondary, east, ex, seconds, threads, control):
    end = time.monotonic()+seconds
    target = index.target
    q = np.full(len(target),-np.inf,'f4')
    n = G['nk']
    q[:n] = -.6 + np.log(np.clip(G['snr_np'][:n],4.5,35.)/7.)
    finite = np.isfinite(base)
    if finite.any():
        med = np.median(base[finite])
        q[ids[finite]] = np.clip(base[finite]-med,-5.,5.)
        if secondary is not None:
            _,aux,marks = secondary
            q[aux] = np.clip(marks-med,-6.,4.)
    depth = np.where(np.isfinite(G['m5c_np']),G['m5c_np'],np.nanmedian(target['mag']))
    for family, share in (((43.,.20,.24,8.5),.70),((160.,.63,.65,24.),1.)):
        remaining = end-time.monotonic()
        if remaining<12. or control['event'].is_set():
            break
        stop = time.monotonic()+remaining*share
        seeds,counts = proposals(index,q,depth,G['lev_e'],east,family,remaining*share*.30,threads)
        seeds = varied(seeds,target['exposure'],ex['night'],ex['t'])
        print('local motion measurements',family[0],counts.tolist(),'arcs',len(seeds),flush=True)
        if not len(seeds) or control['event'].is_set():
            continue
        left = stop-time.monotonic()
        if left<4.:
            continue
        complete = attributables.follow(index,seeds,threads,max(2.,.21*left),cap=4500,epro=east)
        if len(complete):
            control['certified'].put(complete[complete['nights']>=5])
        left = stop-time.monotonic()
        if left>3. and not control['event'].is_set():
            # Systematic ranging uses independent nights to constrain both
            # unknown line-of-sight coordinates of the short measurement.
            fits,stats = attributables.surface(index,seeds,east,threads,left,
                                               refine=False,limit=9000)
            reports = fits[fits['nights']>=5]
            if len(reports):
                control['certified'].put(reports)
            print('local motion continuation',family[0],stats.tolist(),'five nights',len(reports),flush=True)


def begin(*args):
    pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='local-motion')
    job = pool.submit(work,*args)
    def finished(future):
        error = future.exception()
        if error is not None:
            print('local motion worker:',type(error).__name__,str(error)[:200],flush=True)
    job.add_done_callback(finished)
    pool.shutdown(wait=False)
    return job

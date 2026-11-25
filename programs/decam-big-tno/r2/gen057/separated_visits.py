"""Confirm long pairs with measured positions on independent dates."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
import time

import numpy as np

import cadence_search
from element_search import UNIT


def proposals(index, marks, depths, lever, east, family, limit, threads, seconds, cap):
    ptr, integer, real = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    run = index.lib.separated_pairs
    run.argtypes = [ptr]*6+[integer,integer,real,integer,ptr,ptr]
    run.restype = integer
    family = np.ascontiguousarray(family, 'f8')
    output = np.empty(cap, UNIT)
    stats = np.zeros(4, 'i8')
    n = run(index.handle,marks.ctypes.data,depths.ctypes.data,lever.ctypes.data,
            east.ctypes.data,family.ctypes.data,limit,threads,seconds,cap,
            output.ctypes.data,stats.ctypes.data)
    if n < 0:raise RuntimeError('separated-visit pair search failed')
    return output[:n].copy(),stats


def work(fields,ids,scores,threads,seconds,control):
    end=time.monotonic()+seconds
    index=fields.index
    marks=np.full(len(index.target),-90.,'f4')
    marks[ids]=scores
    # Disk and excited populations share an unrestricted final orbital fit.
    # Confirm each family before beginning the next, with its own calibrated
    # magnitude, source-density and visit-overlap ordering.
    for family,limit,share in (
        ((44.,.115,.995,.065,.090,.19,4900.),2400,.55),
        ((40.,.205,.98,.15,.32,.33,4900.),1850,.78)):
        if control['event'].is_set():break
        room=(end-time.monotonic())*share
        if room<9.:break
        now=time.monotonic()
        bank,stats=proposals(index,marks,fields.depth,fields.lever,fields.east,family,limit,threads,
                             max(4.,room*.29),320000)
        left=room-(time.monotonic()-now)
        if not len(bank) or left<2. or control['event'].is_set():continue
        # Regional source marks are in log-density units but do not share
        # the normalization of the short-pair catalogue.
        bank['score']=np.clip(bank['score']-np.median(bank['score'])+3.,-12.,17.)
        result,counts=cadence_search.search(fields,bank,threads,left,cap=8500)
        control['certified'].put(result)
        print('separated visits',family[0],stats.tolist(),counts.tolist(),'confirmed',
              int((result['nights']>=5).sum()),flush=True)
    # At larger distance parallax dominates and pair ellipses become
    # smaller. Broad velocity priors include very inclined and radial
    # motion. Search inverse distance over overlapping logarithmic bands.
    budget=end-time.monotonic()
    if budget<10. or control['event'].is_set():return
    banks=[]
    for radius in (128.,512.,2048.):
        room=end-time.monotonic()
        if room<max(5.,.52*budget) or control['event'].is_set():break
        family=(radius,.48,.73,.52,.69,.57,4900.)
        bank,_=proposals(index,marks,fields.depth,fields.lever,fields.east,family,
                         2200,threads,max(1.,min(.16*budget,room-.5*budget)),85000)
        if len(bank):
            bank['score']-=np.median(bank['score'])
            banks.append(bank)
    room=end-time.monotonic()
    if not banks or room<1. or control['event'].is_set():return
    bank=np.concatenate(banks)
    bank['score']=np.clip(bank['score']+3.,-12.,17.)
    bank=bank[np.argsort(-bank['score'],kind='stable')]
    result,counts=cadence_search.search(fields,bank,threads,room,cap=4800)
    control['certified'].put(result)
    print('distant separated visits',len(banks),'distance bands',counts.tolist(),'confirmed',
          int((result['nights']>=5).sum()),flush=True)


def begin(*args):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='separated-visits')
    job=pool.submit(work,*args)
    def done(future):
        if future.exception() is not None:
            print('separated visit search:',str(future.exception())[:180],flush=True)
    job.add_done_callback(done)
    pool.shutdown(wait=False)
    return job

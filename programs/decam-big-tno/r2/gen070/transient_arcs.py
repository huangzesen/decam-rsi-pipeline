"""Multi-night orbit search using transient evidence in the original visits."""
import ctypes
import time
import weakref

import numpy as np

from original_recovery import RECOVERY, integer, ptr, real


EVIDENCE = np.dtype([('peer', 'i4'), ('references', 'i4')] +
                    [(k, 'f4') for k in ('blank', 'repeat', 'motion', 'opportunities')])
POINT = np.dtype([('row', 'i4'), ('peer', 'i4'), ('merit', 'f4'), ('single', 'f4')])
PAIR = np.dtype([('a', 'i4'), ('b', 'i4'), ('merit', 'f4'), ('gamma', 'f4')])
PROPOSAL = np.dtype([('row','i4',4)]+[(k,'f4') for k in ('g','h','merit')]+[('witness','i4')])


def measure(sky, rows, depth, east, threads, seconds):
    rows = np.ascontiguousarray(rows, 'i4')
    depth = np.ascontiguousarray(depth, 'f8')
    east = np.ascontiguousarray(east, 'f8')
    out = np.zeros(len(rows), EVIDENCE)
    out['peer'] = -1
    fn = sky.lib.measure_transient_visits
    fn.argtypes = [ptr, ptr, ptr, ptr, integer, integer, real, ptr]
    fn.restype = integer
    done = fn(sky.handle, depth.ctypes.data, east.ctypes.data, rows.ctypes.data,
              len(rows), threads, max(.05, seconds), out.ctypes.data)
    if done < 0:
        raise RuntimeError('reference-visit measurement failed')
    return out, done


def normalize(e, m, score, sn):
    """Empirical upper-tail probability among peaks of similar flux in this image."""
    mb = np.clip(np.nan_to_num((m-12.)*1.5, nan=0.), 0, 28).astype('i4')
    strata = (e*29+mb)*2+(sn>=9.)
    bins = 97
    k = np.clip(((score+12.)*3.).astype('i4'), 0, bins-1)
    h = np.bincount(strata*bins+k, minlength=(strata.max()+1)*bins).reshape(-1, bins)
    tail = h[:, ::-1].cumsum(1)[:, ::-1]
    return -np.log((tail[strata,k]-.49*h[strata,k]+1.5)/(tail[strata,0]+3.))


def catalogue(sky, rows, values, base):
    signal, sharp = sky.arrays[:2]
    e, m, sn = sky.exposure[rows], sky.mag[rows], signal[rows]
    blank = values['blank']
    # Endpoint likelihoods are conditional on passing the original
    # foreground selection. Rare independent repeats carry the most weight.
    repeat = np.where(values['peer']>=0, np.clip(values['repeat'], -8., 11.), -8.)
    zr = normalize(e, m, repeat, sn)
    zb = normalize(e, m, blank, sn)
    scale = np.hypot(.20, 3.5/np.maximum(4.5,sn))
    shape = np.clip((sharp[rows]-1.04)/scale, -4., 4.)
    centered = np.clip(np.nan_to_num(base-np.nanmedian(base), nan=-8.), -8., 8.)
    single = (.53*centered + .47*np.log(np.minimum(sn,38.)/5.) - .27*shape**2
              + .34*np.clip(blank, -10., 6.) + .46*np.minimum(zb,4.))
    trusted = (values['peer']>=0)&(repeat>1.1)
    support = np.where(trusted, .70*np.minimum(repeat,8.)+.76*zr, 0.)
    table = np.empty(len(rows), POINT)
    table['row'] = rows
    table['peer'] = np.where(trusted, values['peer'], -1)
    table['merit'] = single+support
    table['single'] = single
    return table


def diversity(values, strata, fraction=.55):
    """Mix absolute merit with within-stratum rank for bounded catalogues."""
    order = np.lexsort((-values,strata))
    group = strata[order]
    pos = np.arange(len(group))
    rank = pos-np.maximum.accumulate(np.where(np.r_[True,group[1:]!=group[:-1]],pos,0))
    size = np.bincount(group)[group]
    result = np.array(values,dtype='f4')
    result[order] += fraction*np.log((size+6.)/(rank+3.))
    return result


def representatives(sky, table, limit):
    if not len(table):
        return np.empty(0,POINT)
    row, mate = table['row'], table['peer']
    e = sky.exposure[row]
    # Choose one end of an unresolved pair. A separated third shutter may
    # give a different association and remains eligible for testing.
    peer = np.where(mate>=0,mate,row)
    key = (np.minimum(row,peer).astype('i8')<<32)|np.maximum(row,peer)
    sort = np.lexsort((-table['merit'],key))
    use = sort[np.r_[True,key[sort[1:]]!=key[sort[:-1]]]]
    mb = np.clip((sky.mag[row[use]]-12.)*1.25,0,23).astype('i4')
    group = (e[use]*24+mb)*2+(mate[use]>=0)
    score = diversity(table['merit'][use], group)
    # In completely isolated visits there is no possible same-night
    # observation. Reserve space for their best archival dropouts.
    solo = use[mate[use]<0]
    solo_rank = diversity(table['single'][solo], e[solo]) if len(solo) else []
    use = np.union1d(use[np.argsort(-score,kind='stable')[:limit]],
                     solo[np.argsort(-np.asarray(solo_rank),kind='stable')[:limit//2]])
    return np.ascontiguousarray(table[use])


def quads(sky, depth, east, small, full, arcs, threads, seconds, cap=65000):
    fn = sky.lib.transient_quads
    fn.argtypes = [ptr,ptr,ptr,ptr,integer,ptr,integer,ptr,integer,integer,real,integer,ptr,ptr]
    fn.restype = integer
    out=np.empty(cap,PROPOSAL)
    ct=np.zeros(5,'i8')
    n=fn(sky.handle,depth.ctypes.data,east.ctypes.data,small.ctypes.data,len(small),
         full.ctypes.data,len(full),arcs.ctypes.data,len(arcs),threads,seconds,cap,
         out.ctypes.data,ct.ctypes.data)
    if n<0:raise RuntimeError('conditional four-visit search failed')
    if not n:return out[:0],ct
    rows=np.sort(out['row'][:n],axis=1)
    _,first=np.unique(rows,axis=0,return_index=True)
    return np.ascontiguousarray(out[np.sort(first)]),ct


def confirm(sky, proposals, threads, seconds, cap=4200):
    fn=sky.lib.transient_confirm
    fn.argtypes=[ptr,ptr,integer,integer,real,integer,ptr,ptr,ptr]
    fn.restype=integer
    out=np.empty(cap,RECOVERY)
    tested=integer()
    n=fn(sky.handle,proposals.ctypes.data,len(proposals),threads,seconds,cap,
         out.ctypes.data,ctypes.byref(tested),getattr(sky,'paired_returns',None))
    if n<0:raise RuntimeError('raw fifth-visit confirmation failed')
    return out[:n].copy(),tested.value


def index_returns(sky,table):
    if not len(table):return
    rows=np.ascontiguousarray(table['row'],'i4')
    peers=np.ascontiguousarray(table['peer'],'i4')
    rarity=normalize(sky.exposure[rows],sky.mag[rows],table['single'],sky.arrays[0][rows])
    score=(.55*np.clip(table['single'],-8.,6.)+.65*(rarity-.90)+
           .32*np.clip(table['merit']-table['single'],0.,7.))
    score=np.ascontiguousarray(score,'f4')
    build=sky.lib.make_paired_returns;build.argtypes=[ptr,ptr,ptr,ptr,integer];build.restype=ptr
    release=sky.lib.free_paired_returns;release.argtypes=[ptr];release.restype=None
    handle=build(sky.handle,rows.ctypes.data,peers.ctypes.data,score.ctypes.data,len(rows))
    if handle:
        sky.paired_returns=handle
        weakref.finalize(sky,release,handle)


def report(sky,proposals,control,threads,seconds):
    end=time.monotonic()+seconds
    trace=control.setdefault('archival_trace',[])
    if len(trace)<10 and len(proposals):
        # Keep high-ranked fits and a systematic sample down the queue.
        take=np.unique(np.r_[np.arange(min(1600,len(proposals))),
                            np.linspace(0,len(proposals)-1,min(3200,len(proposals)),dtype='i8')])
        trace.append(proposals['row'][take].copy())
    comp=control.get('visit_components')
    extra=min(10.,.22*seconds) if comp is not None else 0.
    hits,tested=confirm(sky,proposals,threads,max(.05,seconds-extra))
    control['raw_results'].put((hits,[]))
    remaining=end-time.monotonic()
    if comp is not None and remaining>.3 and not control['event'].is_set():
        try:
            count=comp.run(proposals,control,threads,remaining)
            if count:print('residual-flux orbit confirmations:',count,flush=True)
        except (ValueError,RuntimeError,MemoryError,ArithmeticError) as error:
            print('residual-flux orbit fits:',str(error)[:160],flush=True)
    return len(hits),tested


def write_trace(outdir,control):
    import json
    import os
    from pathlib import Path
    values=control.get('archival_trace',[])
    if not values:return
    path=Path(outdir)/'trace.json'
    if path.exists() and path.stat().st_size>88_000_000:return
    data=json.loads(path.read_text()) if path.exists() else {'stages':[]}
    stages=data['stages']
    if len(stages)>=6:
        earlier=[j for j,s in enumerate(stages) if 'pass' in s['name']]
        if not earlier:earlier=list(range(len(stages)))
        stages.pop(min(earlier,key=lambda j:len(stages[j]['groups'])))
    combined=np.unique(np.sort(np.concatenate(values),axis=1),axis=0)
    groups=[v[v>=0].tolist() for v in combined]
    stages.append(dict(name='Cross-season measured orbit proposals',groups=groups))
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as fp:json.dump(data,fp)
    os.replace(tmp,path)


def unique_pairs(bank):
    if not len(bank):return bank
    key=(np.minimum(bank['a'],bank['b']).astype('i8')<<32)|np.maximum(bank['a'],bank['b'])
    _,first=np.unique(key,return_index=True)
    return np.ascontiguousarray(bank[np.sort(first)])


def local_pairs(sky,depth,east,table,threads,seconds,cap=1750000):
    fn=sky.lib.transient_local_pairs
    fn.argtypes=[ptr,ptr,ptr,ptr,integer,integer,real,integer,ptr,ptr];fn.restype=integer
    counts=np.zeros(3,'i8')
    pairs=np.empty(cap,PAIR)
    n=fn(sky.handle,depth.ctypes.data,east.ctypes.data,table.ctypes.data,len(table),
         threads,seconds,cap,pairs.ctypes.data,counts.ctypes.data)
    if n<0:raise RuntimeError('close-visit pair catalogue failed')
    return order_pairs(sky,pairs[:n]),counts


def order_pairs(sky,bank):
    if not len(bank):return bank
    a,b=bank['a'],bank['b']
    ea,eb=sky.exposure[a],sky.exposure[b]
    tt=sky.geometry['t']
    gap=np.abs(tt[ea]-tt[eb])
    mb=np.clip((.5*(sky.mag[a]+sky.mag[b])-12.)*1.7,0,31).astype('i4')
    # Both short arcs and measurements in different apparitions get a
    # share of time. A very dense visit cannot suppress every other epoch.
    group=(np.digitize(gap,[3.,18.,110.,500.,1150.])*32+mb)*len(tt)+np.minimum(ea,eb)
    rank=diversity(bank['merit'],group,.91)
    return unique_pairs(bank[np.argsort(-rank,kind='stable')])



"""Multi-night orbit search using transient evidence in the original visits."""
import ctypes
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

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


def pairs(sky, depth, east, targets, anchors, family, threads, seconds, cap=440000):
    par = np.ascontiguousarray(family, 'f8')
    out = np.empty(cap,PAIR)
    counts = np.zeros(2,'i8')
    fn = sky.lib.transient_pairs
    fn.argtypes = [ptr,ptr,ptr,ptr,integer,ptr,integer,ptr,integer,real,integer,ptr,ptr]
    fn.restype = integer
    n = fn(sky.handle,depth.ctypes.data,east.ctypes.data,targets.ctypes.data,len(targets),
           anchors.ctypes.data,len(anchors),par.ctypes.data,threads,seconds,cap,out.ctypes.data,
           counts.ctypes.data)
    if n<0:raise RuntimeError('two-visit orbit selection failed')
    return out[:n].copy(),counts


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
    stages.append(dict(name='Archival four-night orbit proposals',groups=combined.tolist()))
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as fp:json.dump(data,fp)
    os.replace(tmp,path)


def anchor_order(sky, table):
    if not len(table):return table
    e=sky.exposure[table['row']]
    group=(e*24+np.clip((sky.mag[table['row']]-12.)*1.25,0,23).astype('i4'))*2+(table['peer']>=0)
    merit=diversity(table['merit'],group,.95)
    return np.ascontiguousarray(table[np.argsort(-merit,kind='stable')])


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


def measured_pairs(sky,table,kept,graph):
    if graph is None:return np.empty(0,PAIR)
    links=graph['data']['pair'].ravel()
    idx=np.flatnonzero((links['row']>=0)&(links['score']>-3.)&(links['gamma']>1./6000.)&(links['gamma']<1./16.))
    a,b=kept[graph['ids'][idx//2]],kept[links['row'][idx]]
    dt=sky.geometry['t'][sky.exposure[a]]-sky.geometry['t'][sky.exposure[b]]
    keep=np.abs(dt)>.50
    a,b,idx=a[keep],b[keep],idx[keep]
    prior=np.zeros(len(sky.mag),'f4')
    prior[table['row']]=table['merit']
    pairs=np.empty(len(a),PAIR)
    pairs['a'],pairs['b']=a,b
    pairs['gamma']=links['gamma'][idx]
    pairs['merit']=links['score'][idx]+.37*(prior[a].clip(-5.,10.)+prior[b].clip(-5.,10.))
    return order_pairs(sky,pairs)


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


def launch(sky, rows, base, depth, east, control, seconds, threads, graph_job=None, kept=None, device='cpu'):
    def work():
        start=time.monotonic()
        end=start+seconds
        live=lambda: not control['event'].is_set()
        z=np.isfinite(depth)
        d=np.ascontiguousarray(np.where(z,depth,np.median(depth[z]) if z.any() else 23.),'f8')
        direction=np.ascontiguousarray(east,'f8')
        values,done=measure(sky,rows,d,direction,threads,min(85.,.24*seconds))
        if done==0 or not live():return
        table=catalogue(sky,rows[:done],values[:done],base[:done])
        del values
        index_returns(sky,table)
        sparse=representatives(sky,table,min(360000,max(15000,len(table)//9)))
        anchors=anchor_order(sky,sparse)
        print('archival transient measurements:',done,'repeated endpoints:',int((sparse['peer']>=0).sum()),flush=True)
        if kept is not None and 'certifier' in control and 'additional' in control:
            from component_confirmation import Measurements
            control['visit_components']=Measurements(kept,control)
        if seconds>70. and len(sparse)>200:
            import seasonal_triangles
            control['seasonal_triangle_job']=seasonal_triangles.launch(sky,d,direction,sparse,table,control,
                min(.51*seconds,205.,max(1.,end-time.monotonic()-4.)))
        groups=[]
        if graph_job is not None and graph_job.done() and graph_job.exception() is None:
            graph=measured_pairs(sky,table,kept,graph_job.result())
            if len(graph):groups.append((graph,.47))
        local,c=local_pairs(sky,d,direction,table,threads,min(.11*seconds,48.))
        if len(local):groups.append((local,.48))
        print('close archival visits:',c.tolist(),len(local),'pairs',flush=True)
        for family,share in [((43.5,.15,.14,.24,3900.),.58),((36.,.29,.30,.52,4900.),.27),
                             ((110.,.42,.43,.57,5200.),.15)]:
            if not live() or time.monotonic()+2.>=end:break
            q,c=pairs(sky,d,direction,sparse,anchors,family,threads,
                      min(.17*seconds*share,max(.1,end-time.monotonic())))
            if len(q):groups.append((order_pairs(sky,q),.41*share))
            print('archival transient pairs:',family[0],c.tolist(),len(q),flush=True)
        if not groups or not live():return
        arr=np.concatenate([q for q,_ in groups])
        ranks=np.concatenate([np.arange(len(q))/share for q,share in groups])
        bank=unique_pairs(arr[np.argsort(ranks,kind='stable')])
        del groups,arr,ranks
        pos=0
        if live() and time.monotonic()+12.<end:
            import visit_device
            start_device=time.monotonic()
            left=end-start_device
            verifier=None
            pending=[]
            def consume(proposals):
                if not live() or (pending and not pending[-1].done()):return
                if pending:
                    last=pending.pop()
                    if last.exception():print('streamed orbital confirmations:',str(last.exception())[:150],flush=True)
                allowance=min(13.,end-time.monotonic()-1.)
                if allowance>1.:pending.append(verifier.submit(report,sky,proposals,control,1,allowance))
            try:
                if device=='cuda':
                    verifier=ThreadPoolExecutor(max_workers=1,thread_name_prefix='visit-confirmation')
                q,used=visit_device.search(sky,d,direction,table,sparse,bank,device,threads,
                                    start_device+(.65 if verifier else .32)*left,control,
                                    emit=consume if verifier else None)
                if len(q) and live():
                    n_hits,tested=report(sky,q,control,threads,max(1.,min(.15*left,end-time.monotonic())))
                    print('device archival visits:',used,'pairs',len(q),'quads',tested,'tested',n_hits,'confirmed',flush=True)
            except (RuntimeError,MemoryError,ValueError,ArithmeticError,AttributeError,
                    TypeError,OSError) as err:
                print('archival range device:',str(err)[:160],flush=True)
            finally:
                if verifier is not None:verifier.shutdown(wait=False)
        for fraction in (.27,.46,1.):
            remain=end-time.monotonic()
            if remain<2. or not live() or pos>=len(bank):break
            slot=remain*fraction
            qs,c=quads(sky,d,direction,sparse,table,bank[pos:],threads,.69*slot)
            pos+=int(c[0])
            if not len(qs) or not live():continue
            n_hits,tested=report(sky,qs,control,threads,max(.1,min(.31*slot,end-time.monotonic())))
            print('conditional archival visits:',c.tolist(),'unique quads',len(qs),
                  'tested',tested,'five nights',n_hits,flush=True)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='archival-visits')
    job=pool.submit(work)
    def finished(j):
        error=j.exception()
        if error:print('archival transient arcs:',repr(error)[:220],flush=True)
    job.add_done_callback(finished)
    pool.shutdown(wait=False)
    return job

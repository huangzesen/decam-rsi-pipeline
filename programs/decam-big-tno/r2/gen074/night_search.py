"""Find five-night tracks by conditioning catalogue queries on measured returns."""
import ctypes
from copy import copy
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import subprocess
import time

import numpy as np

import night_forecasts as forecast
from companion_graph import VISIT, build_library
from nightly_fields import NightFields, merge_measurements
from quartet_geometry import AS


POINT = np.dtype([(k, 'f4') for k in ('x', 'y', 'mag', 'sigma', 'quality')]+[('row', 'i4')])
FIELD = np.dtype([(k, 'f4') for k in ('xmin', 'inverse', 'lever', 'density')]+
                 [(k, 'i4') for k in ('start', 'columns', 'night', 'pad')])
TRACK = np.dtype([('row', 'i4', 5), ('gamma', 'f4'), ('score', 'f4')])


@lru_cache(maxsize=1)
def device_lookup():
    import torch
    from apparent_backend import cuda_kernel
    kernel=cuda_kernel(Path(__file__).with_name('night_query.cpp').read_text(),'search_nights')
    p=np.zeros(6,POINT)
    p['x']=[0.,20.,40.,60.,80.,80.];p['mag']=22.;p['sigma']=.17
    p['row']=[0,1,2,3,4,-157]
    f=np.zeros(6,FIELD)
    f['xmin']=p['x']-.5;f['inverse']=.25;f['columns']=1;f['density']=.001
    f['start']=2*np.arange(6);f['night']=np.arange(6)
    f['columns'][4]=0;f['pad'][4]=6;f['night'][5]=4
    edges=np.column_stack((np.arange(6),np.arange(1,7))).ravel().astype('i4')
    model=np.zeros(4,forecast.MODEL)
    model['count']=2;model['gamma']=.025;model['view'][:,:5]=np.arange(5)
    model['step']['coef'][:,:5,0,0]=np.arange(5)*20.
    model['step']['chol'][:,:,0]=1.;model['step']['chol'][:,:,2]=1.
    model['step']['coef'][1: ,4,0,0]+=700.
    model['step']['coef'][2: ,3,0,0]+=700.
    model['step']['coef'][3  ,2,0,0]+=700.
    arguments=[p,edges,f,model,np.arange(4,dtype='i4'),np.zeros(4,'i4')]
    data=[torch.as_tensor(a.view('u1'),device='cuda') for a in arguments]
    out=torch.empty((4,4,7),dtype=torch.int32,device='cuda')
    kernel(grid=(1,1,1),block=(32,1,1),args=data+[4,out])
    actual=out.cpu().numpy().view(TRACK).reshape(4,4)['row']
    if (not np.array_equal(actual[0,0],[0,1,2,3,-157]) or
        not np.array_equal(actual[1,3],[0,1,2,3,-1]) or
        not np.array_equal(actual[2,3],[0,1,2,-1,-1]) or np.any(actual[3]!=-1)):
        raise ArithmeticError('device failed the measured-return fixture')
    return kernel


def mean_positions(xy, ex, n):
    counts = np.bincount(ex, minlength=n)
    center = np.column_stack([np.bincount(ex, weights=v, minlength=n) for v in xy.T])
    return center / np.maximum(1, counts[:, None]), counts


def pack_field(d, fields, j, offset, lo):
    u, v = d['x'], d['y']
    lower = np.min(u)-.05
    width = max(18., (np.ptp(u)+.1)/max(10., 1.5*np.sqrt(len(u))))
    columns = np.floor((u-lower)/width).astype('i4')
    nx = int(columns.max()+1)
    row = np.lexsort((v, columns))
    ends = np.r_[lo, lo+np.cumsum(np.bincount(columns, minlength=nx))].astype('i4')
    fields['start'][j], fields['columns'][j] = offset, nx
    fields['xmin'][j], fields['inverse'][j] = lower, 1./width
    fields['density'][j] = len(d)/max(3600., .72*(np.ptp(u)+45.)*(np.ptp(v)+45.))
    return row, ends


class Catalogue:
    def __init__(self, target, ids, quality, sigma, visits, pole, pointings=None):
        order = np.argsort(target['exposure'][ids], kind='stable')
        ids, quality = ids[order], quality[order]
        self.ex = target['exposure'][ids]
        self.xy = np.column_stack((target['x'][ids], target['y'][ids]))
        center, count = mean_positions(self.xy, self.ex, len(visits))
        if pointings is not None:
            center[count==0]=pointings[count==0]
        nights = NightFields(visits, center, count, pole)
        group = nights.group[self.ex]
        self.nights = nights
        corrected = nights.correct(self.xy, self.ex)
        take, quality = merge_measurements(corrected, self.ex, group, ids, quality,
                                          target['mag'][ids], 1.086/np.maximum(.001, sigma[ids]))
        group, corrected, ids = group[take], corrected[take], ids[take]
        count = len(nights.visits)
        self.center, total = mean_positions(corrected, group, count)
        for j in np.flatnonzero(total==0):
            self.center[j]=np.mean(center[nights.member[j]],axis=0)
        self.extent = np.full(count,np.deg2rad(.66))
        fields = np.zeros(count, FIELD)
        fields['lever'] = nights.visits['lever']
        fields['night'] = nights.visits['night']
        order = np.argsort(group, kind='stable')
        groups = np.searchsorted(group[order], np.arange(count+1))
        point = np.empty(len(ids), POINT)
        point['x'], point['y'] = (corrected/AS).T
        point['mag'], point['sigma'], point['row'] = target['mag'][ids], sigma[ids], ids
        point['quality'] = np.clip(quality-np.median(quality), -9., 10.)
        packed = np.empty_like(point)
        edges, roots, offset = [], [], 0
        for j, (lo, hi) in enumerate(zip(groups[:-1], groups[1:])):
            d = point[order[lo:hi]]
            if not len(d):
                roots.append(np.empty(0, 'i4'))
                continue
            distance = np.hypot(d['x']*AS-self.center[j,0],d['y']*AS-self.center[j,1])
            self.extent[j] = max(self.extent[j],np.quantile(distance,.99)/2.5)
            row, ends = pack_field(d, fields, j, offset, lo)
            packed[lo:hi] = d[row]
            edges.append(ends)
            offset += len(ends)
            # Uniformly sample ties across the imaged area when many
            # endpoints have indistinguishable source likelihoods.
            q = d['quality'][row]
            tie = (d['row'][row].astype('i8')*2654435761 % 65537)/65537.
            roots.append((lo+np.lexsort((tie, -q))).astype('i4'))
        self.points, self.edges, self.fields = packed, np.concatenate(edges), fields
        self.roots = roots
        self.raw_offset = 0
        self.tiles = []
        for r in roots:
            if not len(r):
                self.tiles.append(None)
                continue
            x=np.floor(packed['x'][r]/330.).astype('i4')
            y=np.floor(packed['y'][r]/330.).astype('i4')
            key=(x-x.min())*(1+int(y.max()-y.min()))+(y-y.min())
            _,first,inv=np.unique(key,return_index=True,return_inverse=True)
            self.tiles.append((np.column_stack((x[first]+.5,y[first]+.5))*330.,inv))
        self.visibility = None

    def footprint(self,cov,depth):
        # Approximate live depth at the resolution of a camera chip. Sample
        # the occupied 45-arcsecond cells so dithered shutters can fill gaps.
        origin,keys=cov[:2]
        ex=keys>>42
        x=(((keys>>21)&((1<<21)-1))//6).astype('i4')
        y=((keys&((1<<21)-1))//6).astype('i4')
        ends=np.searchsorted(ex,np.arange(len(depth)+1))
        maps=[]
        self.area=np.zeros(len(self.nights.member))
        for j,group in enumerate(self.nights.member):
            ix=np.concatenate([np.arange(ends[j],ends[j+1],dtype='i4') for j in group])
            if not len(ix):
                maps.append((0,0,np.full((1,1),-80.,'f4')))
                continue
            ox,oy=x[ix].min(),y[ix].min()
            nx,ny=int(x[ix].max()-ox+1),int(y[ix].max()-oy+1)
            if nx*ny>1000000:
                maps.append((0,0,np.full((1,1),-80.,'f4')))
                continue
            grid=np.full(nx*ny,-80.,'f4')
            fraction=np.zeros(nx*ny,'f4')
            for e in group:
                sl=slice(ends[e],ends[e+1])
                cell=(x[sl]-ox)*ny+y[sl]-oy
                if not len(cell):continue
                loc,count=np.unique(cell,return_counts=True)
                measured=depth[e]-.75*np.maximum(0.,1.-count/24.)
                grid[loc]=np.maximum(grid[loc],measured)
                fraction[loc]=np.maximum(fraction[loc],np.minimum(1.,count/30.))
            maps.append((ox,oy,grid.reshape(nx,ny)))
            self.area[j]=max(3600.,fraction.sum()*270.**2)
            self.fields['density'][j]=len(self.roots[j])/self.area[j]
        self.visibility = (origin/AS, maps)

    def anchors(self,model,delta,cap):
        a=model['view'][0];rows=self.roots[a]
        if len(rows)<=cap or self.visibility is None:
            return rows[:cap]
        origin,maps=self.visibility
        xy,tile=self.tiles[a]
        def depths(k):
            ox,oy,grid=maps[model['view'][k]]
            pred=(xy[:,None,:]+delta[None,:,k]-origin)/270.
            u,v=np.floor(pred).astype('i4').transpose(2,0,1)
            u-=ox;v-=oy
            inside=(u>=0)&(v>=0)&(u<grid.shape[0])&(v<grid.shape[1])
            return np.where(inside,grid[u.clip(0,grid.shape[0]-1),v.clip(0,grid.shape[1]-1)],-80.)
        limit=np.minimum(depths(1),depths(2))
        # Two more independent nights must be visible on the same trial path.
        available={}
        for j in range(model['count']):
            n=self.nights.visits['night'][model['view'][3+j]]
            d=depths(3+j)
            available[n]=np.maximum(available[n],d) if n in available else d
        if len(available)<2:return rows[:cap]
        first=np.full_like(limit,-80.)
        second=first.copy()
        for d in available.values():
            second=np.maximum(second,np.minimum(first,d))
            first=np.maximum(first,d)
        minimum=np.minimum(limit,second)
        # Integrate the 4.5-sigma transition and rotational scatter over
        # common physical paths. The floor allows imperfect depth estimates.
        magnitude=self.points['mag'][rows]
        probability=np.zeros(len(rows),'f4')
        for d in minimum.T:
            probability+=1./(1.+np.exp(np.clip((magnitude-d[tile]-.16)/.32,-18.,18.)))
        probability/=minimum.shape[1]
        merit=self.points['quality'][rows]+1.28*np.log(.035+probability)
        return rows[np.argpartition(-merit,cap-1)[:cap]]

    def add_returns(self, sky, rawmap, event, priority=None, deadline=None):
        # Only the two independent predictions use these denser catalogues.
        # Store an original row absent from the orbit index as a negative ID.
        # Build one night at a time, including the full camera foreground.
        n = len(sky.mag)
        if n*POINT.itemsize > 2700000000:
            return
        known = -2-np.arange(n,dtype='i4')
        known[rawmap] = np.arange(len(rawmap),dtype='i4')
        edges = np.searchsorted(sky.exposure, np.arange(len(self.nights.raw)+1))
        out = np.empty(len(self.points)+n,POINT)
        out[:len(self.points)] = self.points
        lo = len(self.points)
        fields = self.fields.copy()
        fields['columns'] = 0
        offset = len(self.edges)
        pieces = [self.edges]
        shape, flags = sky.arrays[1], sky.arrays[5]
        order_of_groups=range(len(fields)) if priority is None else priority
        for j in order_of_groups:
            if event.is_set():
                return
            if deadline is not None and time.monotonic()>=deadline:
                break
            images=self.nights.member[j]
            ids = np.concatenate([np.arange(edges[e],edges[e+1],dtype='i4') for e in images])
            good = (np.isfinite(sky.mag[ids]) & np.isfinite(sky.pos[ids]).all(axis=1) &
                    np.isfinite(sky.arrays[0][ids]) & ((flags[ids]&1)==0))
            ids = ids[good]
            if not len(ids):
                continue
            d = np.empty(len(ids),POINT)
            d['x'],d['y'] = (self.nights.correct(sky.pos[ids],sky.exposure[ids])/AS).T
            d['row'],d['mag'] = known[ids],sky.mag[ids]
            sn = sky.arrays[0][ids]
            d['sigma'] = 1.086/np.maximum(4.5,sn)
            mark = ((np.nan_to_num(shape[ids],nan=4.)-1.02)/np.hypot(.18,3.6/sn))**2
            d['quality'] = np.clip(-2.8+1.75*sky.population[ids]+.30*np.log(np.maximum(4.5,sn)/6.)
                                   -.37*np.minimum(12.,mark)-.55*((flags[ids]&6)!=0),-9.,5.)
            order, ends = pack_field(d,fields,j,offset,lo)
            if self.area[j]>0:
                fields['density'][j]=len(d)/self.area[j]
            out[lo:lo+len(d)] = d[order]
            lo += len(d)
            pieces.append(ends)
            offset += len(ends)
        base=self.fields.copy()
        self.raw_offset = len(base)
        ready=fields['columns']>0
        base['pad'][ready] = self.raw_offset+np.flatnonzero(ready)+1
        self.points = out[:lo]
        self.edges = np.concatenate(pieces)
        self.fields = np.concatenate((base,fields))


class Query:
    def __init__(self, cat, device, module=None):
        self.cat = cat
        if module is None:
            try:
                module=build_library('measured_triples')
            except (OSError,RuntimeError,subprocess.SubprocessError):
                module=build_library('night_query')
        self.module=module
        sizes = np.empty(5,'i4')
        module.night_layout.argtypes = [ctypes.c_void_p]
        module.night_layout.restype = None
        module.night_layout(sizes.ctypes.data)
        if list(sizes) != [POINT.itemsize,FIELD.itemsize,forecast.MODEL.itemsize,TRACK.itemsize,4]:
            raise RuntimeError('incompatible native five-night catalogue layout')
        self.fn = module.search_nights_host
        ptr = ctypes.c_void_p
        self.fn.argtypes = [ptr]*6+[ctypes.c_int, ptr]
        self.fn.restype = None
        self.kernel = None
        if str(device).startswith('cuda'):
            import torch
            try:
                if torch.cuda.mem_get_info()[0] < cat.points.nbytes*2 + 300000000:
                    return
                self.kernel = device_lookup()
                self.stream = torch.cuda.Stream()
                with torch.cuda.stream(self.stream):
                    self.arrays = [torch.as_tensor(a.view('u1'), device='cuda') for a in
                                   (cat.points, cat.edges, cat.fields)]
            except (AttributeError, RuntimeError, OSError, MemoryError, ArithmeticError) as exc:
                print('night forecast host lookup:', str(exc)[:160], flush=True)
                self.kernel = None

    def host(self, models, jobs, roots):
        out = np.empty((len(jobs), 4), TRACK)
        args = [self.cat.points, self.cat.edges, self.cat.fields, models, jobs, roots]
        self.fn(*(a.ctypes.data for a in args), len(jobs), out.ctypes.data)
        return out

    def lookup(self, models, jobs, roots):
        if self.kernel is None:
            z = self.host(models, jobs, roots)
            return z[z['row'][:, :, 0]>=0]
        import torch
        with torch.cuda.stream(self.stream):
            data = [torch.as_tensor(a.view('u1'), device='cuda') for a in (models, jobs, roots)]
            out = torch.empty((len(jobs),4,TRACK.itemsize//4), dtype=torch.int32, device='cuda')
            self.kernel(grid=((len(jobs)+63)//64,1,1), block=(64,1,1),
                        args=self.arrays+data+[len(jobs), out])
            if not hasattr(self, 'verified'):
                count = min(256,len(jobs))
                check = self.host(models, jobs[:count], roots[:count])
                actual = out[:count].cpu().numpy().view(TRACK).reshape(count,4)
                good = check['row'][:,:,0]>=0
                if not (np.array_equal(good, actual['row'][:,:,0]>=0) and
                        np.array_equal(np.sort(check['row'][good],axis=1), np.sort(actual['row'][good],axis=1))):
                    raise ArithmeticError('five-night host and device lookup disagree')
                self.verified = True
            live = out[out[:,:,0]>=0].cpu().numpy()
        return live.view(TRACK).ravel()


def search(cat, plans, pole, end, stopped, device, root_cap=None, upgrade=None):
    filtered = Query(cat, device)
    import measured_triples
    try:
        broad=measured_triples.Observations(filtered)
    except (OSError,RuntimeError,ValueError,MemoryError,subprocess.SubprocessError) as exc:
        print('measured-pair directory:',str(exc)[:200],flush=True)
        broad=None
    original = filtered if cat.raw_offset else None
    raw_broad=broad if original is not None else None
    dense_cat = cat
    draws = forecast.physical_draws()
    from return_ephemeris import Ephemerides
    prediction=Ephemerides(cat,pole,draws)
    from pair_visits import PairVisits
    pair_visits=PairVisits(prediction,plans)
    extra=[]
    stats = np.zeros(3,'i8')
    cursor=0
    revisit=0
    before_raw=0
    batches=0
    while cursor<len(plans) or extra or (original is not None and revisit<before_raw):
        if time.monotonic() >= end or stopped.is_set():
            return
        if upgrade is not None and upgrade.done():
            try:
                dense=upgrade.result()
            except (MemoryError,ValueError,IndexError,RuntimeError) as exc:
                print('original return directory:',str(exc)[:150],flush=True)
                dense=None
            if dense is not None and dense.raw_offset:
                dense_cat=dense
                original=Query(dense,device,filtered.module)
                if broad is not None:
                    raw_broad=measured_triples.Observations(original,broad.cache)
                before_raw=cursor
            upgrade=None
        dense=original is not None
        early=dense and revisit<before_raw and (cursor>=len(plans) or batches%7==6)
        lookup=original if dense else filtered
        active=dense_cat if dense else cat
        gpu=lookup.kernel is not None
        start=revisit if early else cursor
        primary=plans[start:start+12]
        selected=extra[:2]
        extra=extra[len(selected):]
        group=primary+[plan for plan,pair in selected]
        adjusted=[]
        for j,((a,b,c,f),rest) in enumerate(group):
            pri=lambda i: 1.28*cat.nights.visits['depth'][i]-np.log(max(25,len(cat.roots[i])))
            if j<len(primary) and not getattr(cat,'directed_starts',False) and pri(b)>pri(a):a,b=b,a
            adjusted.append(((a,b,c,f),rest))
        models,paths = forecast.models(adjusted, cat.nights.visits, cat.center, cat.nights.spread, pole, draws,
                                       with_paths=True,extent=getattr(cat,'extent',None),
                                       population=prediction.population if broad is not None else None)
        if not dense:
            models['family']=-1
        # Use a deep anchor pool on the early, frequently observable plans.
        # Other positions in a five-night path can use the entire catalogue.
        cap = root_cap or (max(720, 3100//(1+start//1600)) if gpu else
                           max(170, 520//(1+start//860)))
        join=raw_broad if dense else broad
        ephemeris=None
        if join is None:
            anchors = [active.anchors(m,d,cap) for m,d in zip(models,paths)]
        else:
            # The admission unit is now a measured three-night arc. A faint
            # starting detection can use information from both later images.
            anchors=[]
            returns=[]
            for j,((a,b,c,f),_) in enumerate(adjusted):
                if time.monotonic()>=end or stopped.is_set():
                    return
                given=selected[j-len(primary)][1] if j>=len(primary) else None
                anchors.append(join.triples(models[j:j+1],f,cap,pairs=given))
                returns.append(prediction.build(models[j:j+1],f,anchors[-1],active.points))
                if j<len(primary) and j==batches%len(primary) and len(extra)<3:
                    extra.extend(pair_visits.propose(join,models[j:j+1],f))
            ephemeris=np.concatenate(returns)
        jobs = np.repeat(np.arange(len(group),dtype='i4'),list(map(len,anchors)))
        roots = np.concatenate(anchors)
        parts = []
        lo=0
        while lo<len(roots):
            if time.monotonic() >= end or stopped.is_set():
                break
            sl = slice(lo, lo+(8192 if gpu else 320))
            try:
                part = (join.confirm(models,jobs[sl],roots[sl],ephemeris) if join is not None else
                        lookup.lookup(models, jobs[sl], roots[sl]))
            except (RuntimeError, ArithmeticError, OSError, MemoryError) as exc:
                if lookup.kernel is None:
                    raise
                print('five-night queries continuing on host:',str(exc)[:160],flush=True)
                lookup.kernel, gpu = None, False
                continue
            if len(part):
                parts.append(part)
            stats[1:] += [len(roots[sl]), len(part)]
            lo=sl.stop
        if early:revisit+=len(primary)
        else:cursor+=len(primary)
        batches+=1
        stats[0] += len(models)
        if join is not None and (start+len(group))//192!=(start//192):
            print('measured night search: pair cache hits, anchors, pairs, triples, followed',
                  join.stat.tolist(),flush=True)
        yield (np.concatenate(parts) if parts else np.empty(0,TRACK)), stats.copy()


def run(index, ids, q, repeat, secondary, G, ex, pole, seconds, control):
    import survey_arcs
    start = time.monotonic()
    end = start+seconds
    visits = np.zeros(len(ex), VISIT)
    for k in ('t', 'ox', 'oy', 'oz', 'night'):
        visits[k] = ex[k]
    depth = np.asarray(G['m5c_np'])
    visits['depth'] = np.where(np.isfinite(depth), depth, np.nanmedian(depth) if np.isfinite(depth).any() else 23.)
    visits['lever'] = G['lev_e']
    sigma = np.full(len(index.target), .21, 'f4')
    signal = G['snr_np']
    sigma[:len(signal)] = 1.086/np.maximum(4.5, signal)
    extra = control['additional']
    sigma[extra['offset']:] = 1.086/np.maximum(4.5, extra['sn'])
    quality = q + .74*np.clip(repeat[:,0],-1.,7.)
    if secondary is not None:
        _, other, value = secondary
        ids, quality = np.r_[ids,other], np.r_[quality,value]
    valid = np.isfinite(quality+index.target['mag'][ids])
    if not np.any(valid) or control['event'].is_set():
        return
    cat = Catalogue(index.target,ids[valid],quality[valid],sigma,visits,pole,G.get('pt_np'))
    cat.footprint(G['cov'],visits['depth'])
    if control['event'].is_set():
        return
    observed = []
    plans = forecast.schedule(cat.xy, cat.ex, visits, cat.nights.group, G['cov'], pole,
                              32768, min(end-2., time.monotonic()+max(5., .12*seconds)), observed)
    if plans and time.monotonic()+6.<end:
        import visit_cost
        plans=visit_cost.order(cat,plans,pole)
    if observed:
        control['visible_nights'] = (visits, cat.nights.group, observed)
    print('five-night forecast geometry:',len(plans),'three-visit starts for',len(cat.points),
          'measurements in',round(time.monotonic()-start,1),'s',flush=True)
    if not plans or control['event'].is_set():
        return
    sky = control.get('original_sky')
    rawmap = np.concatenate((G['keep_np'],extra['rows'])).astype('i4')
    upgrade=None
    if sky is not None and hasattr(sky.lib,'night_candidate_fit') and seconds>=95. and not control['event'].is_set():
        def original_catalogue():
            dense=copy(cat)
            # Prioritize the shutters that can confirm many simulated paths.
            # The filtered index can already search the other shutters.
            mass=np.zeros(len(cat.fields))
            for k,((a,b,c,family),future) in enumerate(plans[:8000]):
                w=1./(1.+k/160.)
                mass[future]+=w
                mass[[a,b,c]] += w*np.array([.24,.24,.58])
            preferred=np.argsort(-mass,kind='stable')
            until=time.monotonic()+.43*max(0.,end-time.monotonic())
            dense.add_returns(sky,rawmap,control['event'],preferred[mass[preferred]>0],until)
            exposed=np.sum(dense.fields['pad']>0)
            print('five-night return index:',len(dense.points),'positions in',exposed,'shutter groups',flush=True)
            return dense
        pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='original-night-index')
        upgrade=pool.submit(original_catalogue)
        pool.shutdown(wait=False)
    screen = None
    if sky is not None:
        from night_screen import Screen
        try:
            screen = Screen(sky,G['dev'])
        except (OSError,RuntimeError,MemoryError,subprocess.SubprocessError) as e:
            print('five-position prefit:',str(e)[:135],flush=True)
    def original_rows(rows):
        mapped=rows.copy()
        available=mapped>=0
        mapped[available]=rawmap[mapped[available]]
        mapped[~available]=-2-mapped[~available]
        return mapped
    def independent_returns(subset,n):
        from attributables import SEED
        from shutter_continuation import forecast as search_images
        subset=subset[:6500 if n==4 else 3600]
        seed=np.zeros(len(subset),SEED)
        seed['rows'].fill(-1)
        rows=original_rows(subset['row'][:,:n])
        order=np.argsort(sky.geometry['t'][sky.exposure[rows]],axis=1)
        seed['rows'][:,:n]=np.take_along_axis(rows,order,axis=1)
        seed['gamma'],seed['score']=subset['gamma'],subset['score']
        cores,_=search_images(sky,seed,n,min(1.65 if n==4 else 1.25,max(.04,end-time.monotonic())))
        trials=np.empty(len(cores),TRACK)
        trials['row']=-2-cores['rows'][:,:5]
        trials['score']=cores['merit']
        trials['gamma']=cores['x'][:,2]
        return trials
    def verify(pieces):
        bank = np.concatenate(pieces)
        rows = bank['row']
        floor=-1-len(sky.mag) if sky is not None else -1
        valid=(rows[:,0]>=0)&np.all(rows[:,:3]!=-1,axis=1)
        valid &= ((rows[:,3]!=-1)|(rows[:,4]==-1))
        valid &= (np.max(rows,axis=1)<len(rawmap))&(np.min(rows,axis=1)>=floor)
        bank = bank[valid]
        if not len(bank):
            return 0
        order = np.argsort(-bank['score'], kind='stable')
        _, first = np.unique(np.sort(bank['row'][order], axis=1), axis=0, return_index=True)
        use = bank[order[np.sort(first)]]
        incomplete=use['row'][:,4]==-1
        complete=[use[~incomplete]]
        if sky is not None and np.any(incomplete):
            partial=use[incomplete]
            for n,attribute,mask in ((4,'quartet_predictions',partial['row'][:,3]!=-1),
                                     (3,'triple_predictions',partial['row'][:,3]==-1)):
                if np.any(mask) and hasattr(sky.lib,attribute):
                    complete.append(independent_returns(partial[mask],n))
        use=np.concatenate(complete)
        if screen is not None and len(use):
            # Profile range and radial speed on *all five* measured positions.
            # Keep a sample of failed prefits for the more general corrector.
            original=original_rows(use['row'])
            fit=screen(use, original)
            good=fit[:,0]>0.
            trace=control.setdefault('night_fit_trace',[])
            traced=control.get('night_fit_trace_count',0)
            if traced<60000 and np.any(good):
                kept=original[good&(fit[:,1]<.36)][:min(1400,60000-traced)]
                if len(kept):
                    trace.append(kept.copy())
                    control['night_fit_trace_count']=traced+len(kept)
            use['gamma'][good]=fit[good,0]
            use['score'][good]+=5.0-2.1*np.minimum(5.,(fit[good,1]/.24)**2)
            fallback=np.flatnonzero(~good)[:min(1200,32+len(use)//20)]
            use=use[np.r_[np.flatnonzero(good),fallback]]
            use=use[np.argsort(-use['score'],kind='stable')]
        native = np.all(use['row']>=0,axis=1)
        selected=use[native]
        fits = survey_arcs.certify(index,selected['row'][:,:4],selected['gamma'],selected['row'][:,4],
                                   1, min(4.,max(.2,end-time.monotonic())),limit=3800)
        control['certified'].put(fits)
        recovered=0
        if sky is not None and np.any(~native):
            raw=use[~native].copy()
            raw['row']=original_rows(raw['row'])
            from shutter_continuation import confirm
            out=confirm(sky,raw,min(3.,max(.2,end-time.monotonic())),3000)
            recovered=len(out)
            if recovered:control['raw_results'].put((out,[]))
        return len(fits)+max(0,recovered)
    pieces = []
    last = time.monotonic()
    count, job, pending = 0, None, 0
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='five-position-fit') as verifier:
        for proposals, stat in search(cat, plans, pole, end-5., control['event'], G['dev'],upgrade=upgrade):
            if len(proposals):
                partial=proposals[proposals['row'][:,4]==-1]
                if len(partial):
                    import intersect_arcs
                    intersect_arcs.offer(G,control,original_rows(partial['row'][:,:3]),
                                         partial['gamma'],partial['score'])
                pieces.append(proposals)
                pending+=len(proposals)
            if job is not None and job.done():
                count+=job.result()
                job=None
            if pieces and job is None and (time.monotonic()-last>6. or pending>6000):
                job=verifier.submit(verify,pieces)
                pieces=[]
                pending=0
                last=time.monotonic()
            elif pending>260000:
                # Balance complete tracks and quartets from independent sky
                # pointings when the exact orbit corrector is saturated.
                reduced=[]
                for chunk in pieces:
                    for sel in (chunk['row'][:,4]>=0,chunk['row'][:,4]<-1,chunk['row'][:,4]==-1):
                        local=chunk[sel]
                        if len(local)>2:
                            cap=max(1,len(local)//2)
                            local=local[np.argpartition(-local['score'],cap-1)[:cap]]
                        if len(local):reduced.append(local)
                pieces=reduced
                pending=sum(map(len,pieces))
        if job is not None:count+=job.result()
        if pieces and not control['event'].is_set() and time.monotonic()<end:
            count+=verify(pieces)
    print('five-night lookup:', stat.tolist() if 'stat' in locals() else [],
          'verified fits',count,flush=True)


def begin(*args):
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='five-night-search')
    future = pool.submit(run,*args)
    def finished(job):
        if job.exception() is not None:
            print('five-night search:',repr(job.exception())[:220],flush=True)
    future.add_done_callback(finished)
    pool.shutdown(wait=False)
    return future


def write_trace(control,outdir):
    import json
    import os
    pieces=control.get('night_fit_trace')
    if not pieces:
        return
    groups=np.unique(np.sort(np.concatenate(pieces),axis=1),axis=0).tolist()
    path=Path(outdir)/'trace.json'
    stages=json.loads(path.read_text()).get('stages',[]) if path.exists() else []
    stages=[s for s in stages if s['name']!='five measured positions']
    if len(stages)>=6:
        # Preserve the other approaches' distinct growth and bridge stages.
        at=next((i for i,s in enumerate(stages) if s['name'].startswith('inner passes')),0)
        del stages[at]
    stages=stages[:5]+[dict(name='five measured positions',groups=groups)]
    tmp=path.with_suffix('.forecast')
    tmp.write_text(json.dumps(dict(stages=stages),separators=(',',':')))
    os.replace(tmp,path)

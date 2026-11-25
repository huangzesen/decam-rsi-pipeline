"""Search for multiple independent returns to the same two-position orbit family."""
import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np

from attributables import SEED as ARC_SEED
from companion_graph import build_library

ARC = 206264.80624709636
ptr, integer, real = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
IMAGE = np.dtype([(k, 'f8') for k in ('t','ox','oy','oz','depth','lever')] +
                 [(k, 'f4') for k in ('xmin','xmax','ymin','ymax','bin_width')] +
                 [(k, 'i4') for k in ('first','columns','night')], align=True)
POINT = np.dtype([(k, 'f4') for k in ('x','y','mag','quality')] + [('row','i4')])
ENDPOINT = np.dtype([(k, 'f8') for k in ('x','y','m','sn')] +
                    [('exposure','i4'),('row','i4')], align=True)
EDGE = np.dtype([('a',ENDPOINT),('b',ENDPOINT),('c',ENDPOINT),('gamma','f8'),('merit','f8'),
                 ('component','i4'),('pad','i4')],align=True)
HIT_SIZE = 36
SEED=np.dtype(ARC_SEED.descr+[('witness','i4')])


def proposals(array,kind):
    if array.dtype==kind:return array
    out=np.empty(len(array),kind)
    for key in ARC_SEED.names:out[key]=array[key]
    if 'witness' in kind.names:out['witness']=-1
    return out


def verify(index,seeds,east,seconds,cap):
    import attributables
    from survey_arcs import certify
    start=time.monotonic();fixed=np.flatnonzero(seeds['witness']>=0)
    measured=None
    if len(fixed) and seconds>.6:
        tested=seeds[fixed]
        measured=certify(index,tested['rows'],tested['gamma'],tested['witness'],1,.38*seconds,
                         limit=max(1200,cap//2))
    others=attributables.follow(index,proposals(seeds,ARC_SEED),1,
                                max(.10,seconds-(time.monotonic()-start)),cap=cap,epro=east)
    return np.concatenate((measured,others)) if measured is not None and len(measured) else others


def catalogue(target, visits, ids, quality, limit=1900000):
    ids = np.asarray(ids, 'i4')
    good = np.isfinite(quality) & np.isfinite(target['mag'][ids])
    ids, quality = ids[good], np.asarray(quality[good], 'f4')
    ex = target['exposure'][ids]
    magbin = np.clip(np.floor(2.*(target['mag'][ids]-13.)).astype('i4'),0,35)
    # A ranking within exposure and magnitude also keeps rare deep images
    # represented when most of the region has much shallower repeat imaging.
    group = ex*36+magbin
    k = np.clip(np.floor((quality-np.median(quality)+10.)*3.).astype('i4'),0,70)
    hist = np.bincount(group*71+k,minlength=len(visits)*36*71).reshape(-1,71)
    tail = hist[:,::-1].cumsum(1)[:,::-1]
    rarity = -np.log((tail[group,k]-.48*hist[group,k]+2.)/(tail[group,0]+4.))
    quality = np.clip(quality-np.median(quality),-9.,9.)+.85*np.clip(rarity-.8,-.8,4.)
    if len(ids)>limit:
        chosen = np.argpartition(-quality,limit-1)[:limit]
        ids, quality, ex = ids[chosen], quality[chosen], ex[chosen]
    image = np.zeros(len(visits), IMAGE)
    for k in ('t','ox','oy','oz','depth','lever','night'):
        image[k] = visits[k]
    finite = np.isfinite(image['depth'])
    image['depth'][~finite] = np.median(image['depth'][finite]) if finite.any() else 23.
    image['xmin'] = image['ymin'] = np.inf
    image['xmax'] = image['ymax'] = -np.inf
    points = np.empty(len(ids), POINT)
    for k in ('x','y'):
        a = target[k][ids]*ARC
        np.minimum.at(image[k+'min'],ex,a)
        np.maximum.at(image[k+'max'],ex,a)
        points[k] = a
    points['mag'], points['quality'], points['row'] = target['mag'][ids],quality,ids
    count = np.bincount(ex,minlength=len(visits))
    used = count>0
    # Narrow columns bound the cost for inclined range strips. Sparse
    # exposures can use wider columns without long binary searches.
    image['bin_width']=np.clip(1400./np.sqrt(np.maximum(1,count)),15.,95.)
    image['xmin'][used]-=.20
    image['columns'][used] = (1+np.floor((image['xmax'][used]-image['xmin'][used])/
                                        image['bin_width'][used])).astype('i4')
    image['first'][1:]=np.cumsum(image['columns'][:-1]+1)
    cols = image['first'][ex]+np.floor((points['x']-image['xmin'][ex])/image['bin_width'][ex]).astype('i4')
    order = np.lexsort((points['y'],cols))
    offsets=np.searchsorted(cols[order],np.arange(int(image['first'][-1]+image['columns'][-1])+1)).astype('i4')
    return image,offsets,points[order]


def edges(target,sn,bank,third=None):
    out = np.empty(len(bank), EDGE)
    out['component']=0
    out['c']['row']=-1
    pairs=[('a',bank['a']),('b',bank['b'])]
    if third is not None:pairs.append(('c',third))
    for name,row in pairs:
        for k in ('x','y','exposure'):
            out[name][k]=target[k][row]
        out[name]['m'],out[name]['row'],out[name]['sn']=target['mag'][row],row,sn[row]
    out['gamma'],out['merit']=bank['gamma'],bank['score']
    good = np.isfinite(out['a']['m']+out['b']['m']+out['gamma'])
    return out[good]


def bound_families(edge,vis,east):
    """Allocate broad speed families where the disk velocity is uncertain."""
    a,b=edge['a'],edge['b']
    ea,eb=vis[a['exposure']],vis[b['exposure']]
    dt=eb['t']-ea['t'];g=edge['gamma']
    x=.5*(a['x']+b['x']);y=.5*(a['y']+b['y'])
    oz=eb['oz']-ea['oz']
    vx=(b['x']-a['x']+g*(eb['ox']-ea['ox']-oz*x))/dt
    vy=(b['y']-a['y']+g*(eb['oy']-ea['oy']-oz*y))/dt
    circular=.01720209895*g**1.5
    mismatch=np.hypot(vx/circular-east[0],vy/circular-east[1])
    uncertain=np.minimum(4.,1./np.maximum(.2,np.hypot(eb['ox']-ea['ox'],eb['oy']-ea['oy'])/
                                                np.maximum(.001,abs(dt))/.0172))
    merit=np.minimum(mismatch,3.)+.45*uncertain
    eligible=np.flatnonzero((mismatch>.36)|(uncertain>1.7))
    if not len(eligible):return edge
    # Prefer both independently plausible endpoints and kinematic diversity.
    rank=np.arange(len(eligible))/(1.+.7*merit[eligible])
    ids=eligible[np.argsort(rank,kind='stable')[:min(160000,len(edge)//3+1)]]
    alternate=edge[ids].copy();alternate['component']=1
    order=np.argsort(np.r_[np.arange(len(edge)),1.5+3.*np.arange(len(alternate))],kind='stable')
    return np.concatenate((edge,alternate))[order]


class Search:
    def __init__(self,target,vis,ids,q,east,limit=1900000):
        self.image,self.columns,self.points = catalogue(target,vis,ids,q,limit)
        self.east=np.asarray(east)
        self.lib=build_library('return_mesh')
        self.lib.mesh_family_bytes.restype=integer
        self.stride=self.lib.mesh_family_bytes()
        self.lib.mesh_families.argtypes=[ptr,integer,ptr,real,real,ptr]
        self.lib.mesh_families.restype=None
        self.lib.mesh_host.argtypes=[ptr,integer,ptr,integer,ptr,ptr,ptr,integer,ptr,integer]
        self.lib.mesh_host.restype=integer
        self.lib.mesh_matches.argtypes=[ptr,integer,ptr,ptr,integer,ptr,integer,ptr]
        self.lib.mesh_matches.restype=integer
        self.kernel=None

    def device(self):
        import torch
        from apparent_backend import cuda_kernel
        if self.kernel is not None and hasattr(self,'catalog_device'):return
        if torch.cuda.mem_get_info()[0]<500000000:
            raise MemoryError('insufficient free memory for the return catalogue')
        self.kernel=cuda_kernel(Path(__file__).with_suffix('.cpp').read_text(),'orbital_returns')
        self.catalog_device=[torch.as_tensor(x.view('u1'),device='cuda')
                             for x in (self.image,self.columns,self.points)]

    def batch(self,edge,slots,threads,use_device=False):
        n=len(edge);ne=len(self.image)
        families=np.empty((n,self.stride),'u1')
        self.lib.mesh_families(edge.ctypes.data,n,self.image.ctypes.data,*map(float,self.east),families.ctypes.data)
        output=np.empty((n,slots),SEED)
        counts=np.empty(n,'i4')
        if use_device:
            import torch
            hit=torch.empty(n*ne*6*HIT_SIZE,device='cuda',dtype=torch.uint8)
            total=torch.zeros(1,device='cuda',dtype=torch.int32)
            family=torch.as_tensor(families,device='cuda')
            self.kernel(grid=((n*ne+127)//128,1,1),block=(128,1,1),
                        args=[family,*self.catalog_device,ne,n,hit,total])
            nh=int(total.item())
            if not 0<=nh<=n*ne*6:raise ArithmeticError('invalid return count on device')
            saved=hit[:nh*HIT_SIZE].cpu().numpy()
            status=self.lib.mesh_matches(families.ctypes.data,n,self.image.ctypes.data,saved.ctypes.data,
                                         nh,output.ctypes.data,slots,counts.ctypes.data)
        else:
            status=self.lib.mesh_host(families.ctypes.data,n,self.image.ctypes.data,ne,
                                      self.columns.ctypes.data,self.points.ctypes.data,output.ctypes.data,
                                      slots,counts.ctypes.data,threads)
        if status<0 or np.any((counts<0)|(counts>slots)):
            raise RuntimeError('incomplete orbital-return batch')
        return output[np.arange(slots)[None,:]<counts[:,None]]


def distinct(seeds,cap):
    if not len(seeds):return seeds
    order=np.argsort(-seeds['score'],kind='stable')
    ids=np.sort(seeds['rows'][order],axis=1)
    _,first=np.unique(ids,axis=0,return_index=True)
    return seeds[order[np.sort(first)[:cap]]]


def triple_edges(seeds,target,sn,vis):
    tri=seeds['rows'][:,:3]
    order=np.argsort(vis['t'][target['exposure'][tri]],axis=1)
    tri=np.take_along_axis(tri,order,axis=1)
    dtype=[('a','i4'),('b','i4'),('score','f4'),('gamma','f4')]
    bank=np.empty(len(tri),dtype)
    bank['a'],bank['b']=tri[:,0],tri[:,2]
    bank['score'],bank['gamma']=seeds['score'],seeds['gamma']
    return edges(target,sn,bank,tri[:,1])


def search(solver,edge,seconds,threads,device='cpu',event=None,cap=65000):
    end=time.monotonic()+seconds
    gpu=str(device).startswith('cuda')
    if gpu:
        try:
            solver.device()
            probe=edge[:min(len(edge),48)]
            if len(probe):
                ref=distinct(solver.batch(probe,5,1),10000)
                got=distinct(solver.batch(probe,5,1,True),10000)
                if (len(ref)!=len(got) or not np.array_equal(ref['rows'],got['rows'])
                        or not np.allclose(ref['score'],got['score'],atol=.004)):
                    raise ArithmeticError('inconsistent orbital returns on the device')
        except Exception as exc:
            print('orbital returns on host:',type(exc).__name__,str(exc)[:170],flush=True)
            gpu=False
    # The device output allocation is bounded even at the largest cadence.
    batch=max(16,min(512,260000//len(solver.image))) if gpu else 72
    carry=np.empty(0,SEED);pieces=[];total=0;examined=0
    for lo in range(0,len(edge),batch):
        if time.monotonic()>=end or (event is not None and event.is_set()):break
        part=edge[lo:lo+batch]
        try:
            v=solver.batch(part,6,threads,gpu)
        except (RuntimeError,MemoryError) as exc:
            if not gpu:raise
            gpu=False
            print('continuing orbital returns on host:',str(exc)[:140],flush=True)
            v=solver.batch(part,6,threads)
        if len(v):
            pieces.append(v);total+=len(v)
        examined+=len(part)
        if total>=cap:
            carry=distinct(np.concatenate([carry,*pieces]),cap)
            pieces.clear();total=0
    if pieces:carry=distinct(np.concatenate([carry,*pieces]),cap)
    return carry,examined


def run(index,graph_job,ids,quality,depth,lever,east,signal,control,seconds,device):
    import companion_graph
    start=time.monotonic();end=start+seconds
    try:
        graph=graph_job.result() if graph_job is not None else None
    except Exception as exc:
        print('rotating frame using the source catalogue:',str(exc)[:140],flush=True)
        graph=None
    if control['event'].is_set() or not len(ids):return
    vis=np.zeros(len(depth),companion_graph.VISIT)
    # The native confirmation index and the search use the same row numbers.
    for k in ('t','ox','oy','oz','night'):vis[k]=index.geometry[k]
    vis['depth'],vis['lever']=depth,lever
    q=np.asarray(quality)
    edge=np.empty(0,EDGE)
    if graph:
        q=q+.35*graph['priority'][ids]
        bank=companion_graph.edge_bank(graph,index.target,vis,950000)
        dt=np.abs(vis['t'][index.target['exposure'][bank['a']]]-vis['t'][index.target['exposure'][bank['b']]])
        bank=bank[dt>.45]
        if len(bank):edge=bound_families(edges(index.target,signal,bank),vis,east)
    if time.monotonic()>=end:return
    solver=Search(index.target,vis,ids,q,east)
    budget=end-time.monotonic()
    if len(solver.points)<3 or budget<4. or control['event'].is_set():return
    budget_pair=.33*budget
    seeds,examined=search(solver,edge,budget_pair,1,device,control['event']) if len(edge) else ([],0)
    if len(seeds) and not control['event'].is_set():
        groups=verify(index,seeds,east,min(.15*budget,max(.2,end-time.monotonic())),4500)
        control['certified'].put(groups)
        print('independent orbital returns:',examined,'pairs,',len(seeds),'four-night proposals,',
              int((groups['nights']>=5).sum()),'verified fits in',round(time.monotonic()-start,1),'s',flush=True)
    remaining=end-time.monotonic()
    if remaining<6. or control['event'].is_set():return
    import rotating_search
    (tri,quad),stats=rotating_search.guide_search(solver,index.target,.53*remaining,device,control['event'])
    if control['event'].is_set():return
    graph_seeds=proposals(companion_graph.proposals(graph,index),SEED) if graph else np.empty(0,SEED)
    # Source scores have different origins. Preserve both orderings.
    def interleave(a,b):
        if not len(a):return b
        if not len(b):return a
        data=np.concatenate((a,b))
        return data[np.argsort(np.r_[np.arange(len(a)),np.arange(len(b))],kind='stable')]
    tri=interleave(tri,graph_seeds)
    edge=triple_edges(tri,index.target,signal,vis)
    remained=end-time.monotonic()
    if remained<=1.:return
    continued,used=search(solver,edge,.57*remained,1,device,control['event'],cap=35000)
    if control['event'].is_set():return
    groups=verify(index,interleave(continued,quad),east,max(.2,end-time.monotonic()),5000)
    control['certified'].put(groups)
    print('rotating orbital guides:',stats,'returned',len(tri),'triples and',len(quad),
          'quartets;',used,'conditional trials,',int((groups['nights']>=5).sum()),'verified fits',flush=True)


def begin(*args):
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='orbital-returns')
    job=pool.submit(run,*args)
    def report(future):
        exc=future.exception()
        if exc is not None:print('independent orbital returns:',type(exc).__name__,str(exc)[:180],flush=True)
    job.add_done_callback(report)
    pool.shutdown(wait=False)
    return job

"""Find an orbit from a long chord and measured displacements at either end."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np

from companion_graph import build_library
from original_recovery import integer, ptr, real
import transient_arcs as T

ARC = 206264.80624709636
ADJACENT = 8
POINT = np.dtype([('t','f8')] + [(k,'f4') for k in
                  ('x','y','px','py','mag','error','lever','quality')] +
                 [('night','i4'),('row','i4')], align=True)
NODE = np.dtype([('center','i4'),('edge','i4',ADJACENT),('score','f4',ADJACENT)])
BLOCK = np.dtype([(k,'f4') for k in ('t','dt','x','y','wx','wy','px','py','dpx','dpy','mag','dmag')]
                 + [('first','i4'),('count','i4')])


def ranks(key):
    """Ordinal within equal consecutive keys."""
    k=np.arange(len(key))
    return k-np.maximum.accumulate(np.where(np.r_[True,key[1:]!=key[:-1]],k,0))


def make_nodes(sky, table, edges, east, limit=180000):
    valid=np.isfinite(table['merit'])&np.isfinite(table['single'])&np.isfinite(sky.mag[table['row']])
    if not valid.all():table=table[valid]
    rows=table['row']
    if not len(rows):return np.empty(0,POINT),np.empty(0,NODE)
    # The graph is sparse in original row space. The dense integer lookup
    # is used only to translate a neighbour into its foreground measurement.
    lookup=np.full(len(sky.mag),-1,'i4')
    lookup[rows]=np.arange(len(rows),dtype='i4')
    repeat=np.flatnonzero(table['peer']>=0)
    a=np.concatenate((lookup[edges['a']],lookup[edges['b']],repeat.astype('i4')))
    b=np.concatenate((edges['b'],edges['a'],table['peer'][repeat]))
    if not len(a):return np.empty(0,POINT),np.empty(0,NODE)
    strength=np.concatenate((edges['merit'],edges['merit'],
                             .65*(table['merit']-table['single'])[repeat]+1.15))
    good=(a>=0)&np.isfinite(strength)&(b>=0)&np.isfinite(sky.mag[b])
    a,b,strength=a[good],b[good],strength[good]
    if not len(a):return np.empty(0,POINT),np.empty(0,NODE)
    n=sky.geometry['night'][sky.exposure[b]]
    # Prevent a single well-sampled night from occupying all the local
    # directions. Keep both the best and a competing peak on that night.
    identity=(a.astype('i8')<<32)|b
    k=np.lexsort((-strength,identity));k=k[np.r_[True,identity[k[1:]]!=identity[k[:-1]]]]
    a,b,n,strength=a[k],b[k],n[k],strength[k]
    key=a.astype('i8')*(int(n.max())+1)+n
    k=np.lexsort((-strength,key))
    within=ranks(key[k])
    use=within<2
    k=k[use];score=strength[k]-1.7*within[use]
    k2=np.lexsort((-score,a[k]))
    rank=ranks(a[k[k2]])
    k=k[k2[rank<ADJACENT]]
    rank=rank[rank<ADJACENT]
    neighbors=np.full((len(rows),ADJACENT),-1,'i4')
    quality=np.full((len(rows),ADJACENT),-5.,'f4')
    neighbors[a[k],rank]=b[k];quality[a[k],rank]=strength[k]
    del a,b,n,strength,rank,key,k,k2,identity
    ex=sky.exposure[rows]
    m=sky.mag[rows]
    strength=quality[:,0]+.28*np.maximum(0.,quality[:,1]+1.)
    # Curate both endpoints with corroborating visits and high-contrast
    # singletons. A singleton can close two other nights at the first end.
    merit=.64*table['single']+.29*np.clip(table['merit']-table['single'],0.,10.)
    merit+=.82*strength
    group=ex*28+np.clip((m-12.)*1.5,0,27).astype('i4')
    scores=T.diversity(merit,group,.73)
    single=T.diversity(table['single'],group,.71)
    kk=min(limit,len(table))
    first=np.argpartition(-scores,kk-1)[:kk]
    ns=min(len(table),max(1,limit//5))
    extra=np.argpartition(-single,ns-1)[:ns]
    selected=np.union1d(first,extra)
    selected=selected[np.argsort(-scores[selected],kind='stable')]
    node=neighbors[selected];occupied=node>=0
    raw=np.union1d(rows[selected],node[occupied]).astype('i4')
    points=np.zeros(len(raw),POINT);ee=sky.exposure[raw]
    geo=sky.geometry[ee];points['t']=geo['t']-np.median(sky.geometry['t'])
    points['row'],points['night']=raw,geo['night']
    xy=sky.pos[raw].astype(float)
    d=np.column_stack((geo['ox'],geo['oy']))-xy*geo['oz'][:,None]
    east=np.asarray(east);north=np.array([-east[1],east[0]])
    points['x'],points['y']=xy@east*ARC,xy@north*ARC
    points['px'],points['py']=d@east*ARC,d@north*ARC
    points['mag']=sky.mag[raw];points['lever']=sky.lever[ee]
    points['error']=np.hypot(.087,1.08/np.maximum(4.5,sky.arrays[0][raw]))
    used=lookup[raw]
    points['quality']=np.where(used>=0,np.clip(table['single'][np.maximum(0,used)],-7.,8.),-.70)
    nodes=np.zeros(len(node),NODE)
    nodes['center']=np.searchsorted(raw,rows[selected])
    nodes['edge']=-1;nodes['edge'][occupied]=np.searchsorted(raw,node[occupied])
    nodes['score']=quality[selected]
    return points,nodes


def spatial_blocks(points,nodes,radius):
    z=points[nodes['center']]
    broad=radius<0.;radius=abs(radius)
    w=(.25 if broad else 1.)*.01720209895/radius**1.5*ARC
    x=z['x']+z['px']/radius-w*z['t']
    y=z['y']+z['py']/radius
    epoch=np.floor(z['t']/20.).astype('i4')
    # A Morton order keeps the angular diameter of a block small even
    # near the boundary of two overlapping detector footprints.
    qx=np.floor((x-x.min())/290.).astype('u8');qy=np.floor((y-y.min())/290.).astype('u8')
    code=np.zeros(len(x),'u8')
    for k in range(17):
        code|=((qx>>k)&1)<<(2*k)
        code|=((qy>>k)&1)<<(2*k+1)
    order=np.lexsort((code,epoch)).astype('i4')
    starts=np.flatnonzero(np.r_[True,epoch[order[1:]]!=epoch[order[:-1]]])
    first=np.concatenate([np.arange(i,j,40,dtype='i4') for i,j in zip(starts,np.r_[starts[1:],len(x)])])
    blocks=np.zeros(len(first),BLOCK)
    blocks['first']=first;blocks['count']=np.diff(np.r_[first,len(x)])
    for data,mid,spread in ((z['t'],'t','dt'),(x,'x','wx'),(y,'y','wy'),(z['px'],'px','dpx'),
                             (z['py'],'py','dpy'),(z['mag'],'mag','dmag')):
        a=np.minimum.reduceat(data[order],first);b=np.maximum.reduceat(data[order],first)
        blocks[mid]=.5*(a+b);blocks[spread]=.5*(b-a)
    return order,blocks


def partner_batches(lib,points,nodes,radius,stop,threads,cap,batch):
    order,blocks=spatial_blocks(points,nodes,radius)
    fn=lib.return_partners
    fn.argtypes=[ptr,ptr,ptr,ptr,integer,integer,integer,real,integer,integer,ptr,ptr]
    fn.restype=integer
    for first in range(0,len(nodes),batch):
        if time.monotonic()>=stop:break
        n=min(batch,len(nodes)-first)
        pairs=np.empty((n,cap,2),'i4');count=np.empty(n,'i4')
        fn(points.ctypes.data,nodes.ctypes.data,order.ctypes.data,blocks.ctypes.data,len(blocks),
           first,n,radius,cap,threads,pairs.ctypes.data,count.ctypes.data)
        yield pairs[np.arange(cap)<count[:,None]]


def circular_fixture():
    t=np.array([0.,3.,11.,398.,403.,414.]);radius=43.;phase=t*(2.*np.pi/365.25)+.48
    obs=np.stack((np.sin(phase),.25*np.cos(phase),np.sqrt(1.-.25**2)*np.cos(phase)),axis=1)
    emission=t-radius/173.144632685
    for _ in range(3):
        a=.01720209895*emission/radius**1.5
        heliocentric=radius*np.stack((np.cos(.21)*np.sin(a),np.sin(.21)*np.sin(a),np.cos(a)),axis=1)
        relative=heliocentric-obs;emission=t-np.linalg.norm(relative,axis=1)/173.144632685
    xy=relative[:,:2]/relative[:,2,None];par=obs[:,:2]-xy*obs[:,2,None]
    p=np.zeros(len(t),POINT);p['t']=t;p['night']=t;p['row']=np.arange(len(t))
    p['x'],p['y']=xy.T*ARC;p['px'],p['py']=par.T*ARC
    p['error']=.14;p['mag']=21.;p['quality']=3.
    nodes=np.zeros(2,NODE);nodes['center']=[0,3];nodes['edge']=-1
    nodes['edge'][:,:2]=[[1,2],[4,5]];nodes['score'][:,:2]=[5.,4.5]
    return p,nodes,radius


class Comparisons:
    def __init__(self,lib,points,nodes,device):
        self.lib=lib;self.points=points;self.nodes=nodes;self.kernels=None
        self.fn=lib.return_chords_cpu
        self.fn.argtypes=[ptr,ptr,ptr,integer,real,integer,ptr];self.fn.restype=None
        if device=='cuda':
            import torch
            from apparent_backend import cuda_kernel
            text=Path(__file__).with_name('return_fit.cpp').read_text()
            try:
                self.kernels=cuda_kernel(text,'return_blocks'),cuda_kernel(text,'return_tiles')
                self.check_device()
                self.p=torch.as_tensor(points.view('u1'),device=device)
                self.n=torch.as_tensor(nodes.view('u1'),device=device)
            except (OSError,RuntimeError,ValueError,AttributeError,ArithmeticError,ImportError,TypeError) as exc:
                print('angular returns using CPU:',str(exc)[:140],flush=True)
                self.kernels=None

    def host(self,pairs,radius,threads):
        out=np.empty(len(pairs),T.PROPOSAL)
        self.fn(self.points.ctypes.data,self.nodes.ctypes.data,pairs.ctypes.data,len(pairs),
                radius,threads,out.ctypes.data)
        return out

    def check_device(self):
        import torch
        p,nodes,r=circular_fixture();order,blocks=spatial_blocks(p,nodes,r)
        to=lambda a:torch.as_tensor(a,device='cuda')
        pp,nn,bb=to(p.view('u1')),to(nodes.view('u1')),to(blocks.view('u1'))
        n=len(blocks);mask=torch.zeros(n,device='cuda',dtype=torch.int32)
        select,fit=self.kernels
        select(grid=(1,1,1),block=(128,1,1),args=[pp,nn,bb,n,0,n,r,mask])
        block=int(np.searchsorted(blocks['first'],int(np.flatnonzero(order==1)[0]),side='right')-1)
        if not mask[block].item():raise ArithmeticError('device missed the circular-orbit block')
        out=torch.empty(8*T.PROPOSAL.itemsize,device='cuda',dtype=torch.uint8)
        fit(grid=(1,1,1),block=(64,1,1),args=[pp,nn,to(order),bb,to(np.array([block],'i8')),n,0,1,r,out])
        found=out.cpu().numpy().view(T.PROPOSAL);got=found[np.argmax(found['merit'])]
        expected=np.empty(1,T.PROPOSAL);pair=np.array([0,1],'i4')
        self.fn(p.ctypes.data,nodes.ctypes.data,pair.ctypes.data,1,r,1,expected.ctypes.data)
        if (got['row'][3]<0 or not np.array_equal(got['row'],expected['row'][0]) or
                not np.isclose(got['merit'],expected['merit'][0],atol=.004,rtol=.0004)):
            raise ArithmeticError('device missed the circular-orbit assignment')

def distinct(proposals,limit):
    if not len(proposals):return proposals
    groups=[]
    four=proposals['row'][:,3]>=0
    for mask,cap in ((four,limit),(~four,limit)):
        z=proposals[mask]
        if not len(z):continue
        if len(z)>3*cap:
            k=np.argpartition(-z['merit'],2*cap-1)[:2*cap];z=z[k]
        keys=np.sort(z['row'],axis=1)
        rank=np.argsort(-z['merit'],kind='stable')
        _,idx=np.unique(keys[rank],axis=0,return_index=True)
        groups.append(z[rank[np.sort(idx)[:cap]]])
    return np.ascontiguousarray(np.concatenate(groups))


def device_batches(compare,radius,end,control):
    import torch
    select,fit=compare.kernels
    order,blocks=spatial_blocks(compare.points,compare.nodes,radius)
    nblock=len(blocks)
    to=lambda a:torch.as_tensor(a,device='cuda')
    index,meta=to(order),to(blocks.view('u1'))
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    checked=False
    with torch.cuda.stream(stream):
        batch=max(16,min(256,400000//max(1,nblock)))
        for first in range(0,len(compare.nodes),batch):
            if time.monotonic()>=end or (control and control['event'].is_set()):break
            count=min(batch,len(compare.nodes)-first)
            n=count*nblock;mask=torch.empty(n,dtype=torch.int32,device='cuda')
            select(grid=((n+127)//128,1,1),block=(128,1,1),
                   args=[compare.p,compare.n,meta,nblock,first,n,float(radius),mask])
            tasks=mask.nonzero().flatten().contiguous()
            for lo in range(0,len(tasks),2600):
                if time.monotonic()>=end or (control and control['event'].is_set()):return
                work=tasks[lo:lo+2600];k=len(work)
                buffer=torch.empty(k*8*T.PROPOSAL.itemsize,dtype=torch.uint8,device='cuda')
                fit(grid=(k,1,1),block=(64,1,1),
                    args=[compare.p,compare.n,index,meta,work,nblock,first,k,float(radius),buffer])
                result=buffer.cpu().numpy().view(T.PROPOSAL)
                good=result['merit']>-1.e9
                if not checked and good.any():
                    # Verify complete groups, including competing and empty
                    # orbital assignments, before accepting device proposals.
                    picks=np.unique(np.flatnonzero(good)//8)[:12]
                    sample=work[to(picks)].cpu().numpy()
                    for j,task in zip(picks,sample):
                        root=first+int(task)//nblock
                        block=blocks[int(task)%nblock]
                        others=order[block['first']:block['first']+block['count']]
                        pairs=np.column_stack((np.full(len(others),root,'i4'),others)).astype('i4')
                        cpu=compare.host(pairs,radius,1)
                        cpu['merit'][others<=root]=-1.e20
                        for group in range(8):
                            subset=cpu[group*8:group*8+8];gpu=result[8*j+group]
                            expected=subset[np.argmax(subset['merit'])] if len(subset) else None
                            if expected is None or expected['merit']<-1.e9:
                                if gpu['merit']>-1.e9:raise ArithmeticError('spurious device return')
                            elif not (np.array_equal(expected['row'],gpu['row']) and
                                      np.isclose(expected['merit'],gpu['merit'],atol=.005,rtol=.0005)):
                                raise ArithmeticError('device and host return assignments disagree')
                    checked=True
                yield result[good],int(blocks['count'][work.cpu().numpy()%nblock].sum())


def search(points,nodes,radius,seconds,threads,device,control=None,limit=45000,compare=None):
    if len(nodes)<2:return np.empty(0,T.PROPOSAL),0
    end=time.monotonic()+seconds
    if compare is None:compare=Comparisons(build_library('return_fit'),points,nodes,device)
    groups=[];size=0;total=0
    def collect(out,work):
        nonlocal groups,size,total
        total+=work
        if len(out):
            groups.append(out);size+=len(out)
        if size>limit*5:
            groups=[distinct(np.concatenate(groups),limit)];size=len(groups[0])
    if compare.kernels is not None:
        try:
            for values,tested in device_batches(compare,radius,end,control):collect(values,tested)
            return distinct(np.concatenate(groups),limit) if groups else np.empty(0,T.PROPOSAL),total
        except (OSError,ValueError,ArithmeticError,RuntimeError,AttributeError,TypeError) as exc:
            print('angular return block fallback:',str(exc)[:170],flush=True)
            compare.kernels=None
    for pairs in partner_batches(compare.lib,points,nodes,radius,end,threads,310,64):
        if control is not None and control['event'].is_set():break
        for first in range(0,len(pairs),32000):
            if time.monotonic()>=end:break
            part=pairs[first:first+32000]
            out=compare.host(part,radius,threads)
            collect(out[out['merit']>-1.e9],len(part))
    return distinct(np.concatenate(groups),limit) if groups else np.empty(0,T.PROPOSAL),total


def begin(sky,rows,base,depth,east,control,seconds,threads,device,kept=None):
    def run():
        start=time.monotonic();end=start+seconds
        active=lambda: not control['event'].is_set()
        d=np.asarray(depth,'f8')
        d=np.ascontiguousarray(np.where(np.isfinite(d),d,np.nanmedian(d) if np.isfinite(d).any() else 22.))
        direction=np.ascontiguousarray(east,'f8')
        values,n=T.measure(sky,rows,d,direction,threads,min(65.,.20*seconds))
        if not n or not active():return
        table=T.catalogue(sky,rows[:n],values[:n],base[:n])
        del values
        valid=np.isfinite(table['merit'])&np.isfinite(sky.mag[table['row']])
        if not valid.all():table=table[valid]
        if not len(table):return
        T.index_returns(sky,table)
        if kept is not None and 'certifier' in control and 'additional' in control:
            from component_confirmation import Measurements
            control['visit_components']=Measurements(kept,control)
        edges,count=T.local_pairs(sky,d,direction,table,threads,min(86.,.21*seconds),cap=2200000)
        if not active() or not len(edges):return
        points,nodes=make_nodes(sky,table,edges,direction)
        print('angular return index:',len(table),'measurements,',len(edges),'local pairs,',len(nodes),
              'endpoints;',round(time.monotonic()-start,1),'s',flush=True)
        remain=end-time.monotonic()
        if not active() or remain<3.:return
        import return_confirmation
        groups=[];total=0
        compare=Comparisons(build_library('return_fit'),points,nodes,device)
        # The negative sign requests the broad transverse-velocity prior.
        # Both the scheduler and the dynamics use the absolute trial distance.
        for radius,share,cap in ((43.,.36,45000),(-44.,.048,14000),(100.,.057,13000),(320.,.055,11000)):
            if not active():return
            if end-time.monotonic()<3.:break
            if share*remain<2.:continue
            q,n=search(points,nodes,radius,share*remain,threads,device,control,cap,compare)
            if len(q):groups.append(q)
            total+=n
        # A close arc with no second local displacement can still predict
        # two further dates in the original transient measurement index.
        if active() and end-time.monotonic()>12.:
            small=T.representatives(sky,table,min(330000,max(15000,len(table)//10)))
            q,n=T.quads(sky,d,direction,small,table,edges,threads,min(.15*remain,.4*(end-time.monotonic())),cap=23000)
            if len(q):groups.append(q)
            print('conditioned visit extensions:',int(n[0]),'local pairs,',len(q),'four-position fits',flush=True)
        if not active() or not groups:return
        proposals=return_confirmation.interleave(groups)
        for group in groups:
            if len(group):
                take=np.unique(np.r_[np.arange(min(1200,len(group))),
                             np.linspace(0,len(group)-1,min(5400,len(group)),dtype='i8')])
                control.setdefault('archival_trace',[]).append(group['row'][take].copy())
        left=end-time.monotonic()
        verified=[];tries=0
        four=proposals[proposals['row'][:,3]>=0]
        if hasattr(sky.lib,'joint_return_fits'):
            bank=return_confirmation.interleave((four,proposals[proposals['row'][:,3]<0]))
            cores=return_confirmation.prepare(sky,bank)
            verified,tries=return_confirmation.joint(sky,cores,d,.63*left,threads)
            control['raw_results'].put((verified,[]))
        hits,tested=T.report(sky,four,control,threads,max(.1,end-time.monotonic())) if len(four) else (0,0)
        print('angular returns:',total,'long pairs tested,',len(proposals)-len(four),'triplets,',len(four),'four-position fits,',
              tries,'joint forecasts,',len(verified),'joint confirmations,',tested,'continued,',
              hits,'additional five-night orbits',flush=True)
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='angular-returns')
    future=pool.submit(run)
    def done(job):
        if job.exception() is not None:print('angular return search:',repr(job.exception())[:220],flush=True)
    future.add_done_callback(done)
    pool.shutdown(wait=False)
    return future

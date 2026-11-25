"""Search compatible displacements from several independent shutter visits."""
import ctypes
from pathlib import Path
import time

import numpy as np

from companion_graph import build_library
from survey_arcs import certify

ARC = 206264.80624709636
WIDTH = 32
POINT = np.dtype([(s, "f4") for s in ("x","y","mag","q","color","err","peer_t",
    "peer_x","peer_y","peer_ox","peer_oy","peer_err")] + [("row","i4"),("e","i4")])
VIEW = np.dtype([(s,"f8") for s in ("t","ox","oy","oz","depth","lev")] +
                [("night","i4"),("pad","i4")])
GRID = np.dtype([(s,"i4") for s in ("bx","by","nx","ny","start")] +
                [(s,"f4") for s in ("cell","density","pad")])
EDGE = np.dtype([("other","i4"),("score","f4"),("g","f4"),("sg","f4")])
TRIPLE = np.dtype([("row","i4",3)]+[(s,"f4") for s in ("g","sg","h","sh","score")])
SEED = np.dtype([("row","i4",5),("score","f4"),("g","f4"),("tri",TRIPLE)])


class Catalogue:
    def __init__(self, bank, ids, direction):
        from return_cells import balanced
        target, ex, sn, q, peer, depths, lever = bank.storage
        self.view = np.zeros(len(ex), VIEW)
        for k in ("t", "ox", "oy", "oz", "night"):
            self.view[k] = ex[k]
        self.view["depth"], self.view["lev"] = depths, lever
        self.direction = tuple(map(float, direction))
        e = target["exposure"][ids]
        take = balanced(q[ids], e, min(1650000,max(65000,len(ids)*2//3)))
        ids = np.union1d(ids[take], bank.anchors)
        ids = ids[np.isfinite(target["mag"][ids]) & np.isfinite(target["x"][ids]+target["y"][ids])]
        p = np.zeros(len(ids), POINT)
        e = target["exposure"][ids]
        p["x"], p["y"] = target["x"][ids]*ARC, target["y"][ids]*ARC
        p["row"], p["e"], p["q"] = ids, e, q[ids]
        p["mag"] = target["mag"][ids]
        p["err"] = np.hypot(.13,1.086/np.maximum(4.5,sn[ids]))
        use = np.flatnonzero(peer[ids]>=0)
        a,b = ids[use],peer[ids[use]]
        ea,eb = target["exposure"][a],target["exposure"][b]
        la,lb = lever[ea],lever[eb]
        dl,dm = lb-la,target["mag"][b]-target["mag"][a]
        variance = .085**2+(1.086/np.maximum(4.5,sn[a]))**2+(1.086/np.maximum(4.5,sn[b]))**2
        color = np.clip(dm*dl/(variance/.27**2+dl*dl),-.55,.65)
        wa=1./(.055**2+(1.086/np.maximum(4.5,sn[a]))**2)
        wb=1./(.055**2+(1.086/np.maximum(4.5,sn[b]))**2)
        p["mag"][use]=(wa*(target["mag"][a]-la*color)+wb*(target["mag"][b]-lb*color))/(wa+wb)
        p["color"][use]=color
        p["err"][use]=np.sqrt(.13**2+1./(wa+wb))
        p["peer_t"][use]=ex["t"][eb]-ex["t"][ea]
        for name,axis in (("x","x"),("y","y")):
            p["peer_"+name][use]=(target[axis][b]-target[axis][a])*ARC
            p["peer_o"+name][use]=((ex["o"+axis][eb]-ex["o"+axis][ea])-
                                      target[axis][a]*(ex["oz"][eb]-ex["oz"][ea]))*ARC
        p["peer_err"][use]=np.sqrt(.125**2+(1.04/sn[a])**2+(1.04/sn[b])**2)
        self.grid=np.zeros(len(ex), GRID)
        order=np.argsort(e,kind="stable")
        cuts=np.searchsorted(e[order],np.arange(len(ex)+1))
        all_points=[];all_lookup=[];cursor=0;offset=0
        for j,(lo,hi) in enumerate(zip(cuts[:-1],cuts[1:])):
            if lo==hi:continue
            piece=p[order[lo:hi]]
            cell=max(66.,float(np.ceil(np.sqrt(float(np.ptp(piece['x']))*float(np.ptp(piece['y']))/50000.))))
            x=np.floor(piece["x"].astype('f8')/cell).astype("i4")
            y=np.floor(piece["y"].astype('f8')/cell).astype("i4")
            bx,by=x.min(),y.min();nx,ny=x.max()-bx+1,y.max()-by+1
            key=x-bx+nx*(y-by)
            which=np.argsort(key,kind="stable");k=key[which]
            count=np.bincount(k,minlength=int(nx)*int(ny))
            area=max(300000.,min(nx*ny,np.count_nonzero(count)*1.75)*cell*cell)
            self.grid[j]=(bx,by,nx,ny,offset,cell,.18*len(piece)/area,0.)
            starts=cursor+np.r_[0,np.cumsum(count,dtype="i4")]
            all_lookup.append(starts.astype('i4'));all_points.append(piece[which])
            cursor+=len(piece);offset+=len(starts)
        if not all_points:raise ValueError("empty shutter catalogue")
        self.point=np.concatenate(all_points);self.lookup=np.concatenate(all_lookup)
        inverse=np.full(len(target),-1,'i4')
        inverse[self.point["row"]]=np.arange(len(self.point),dtype='i4')
        self.anchors=inverse[bank.anchors]
        self.anchors=self.anchors[self.anchors>=0]
        self.order=np.argsort(ex["night"],kind='stable')
        self.starts=np.flatnonzero(np.r_[True,np.diff(ex["night"][self.order])!=0])


def select_edges(cat, edges):
    """Allow each night one first choice. Include competing shutter peaks."""
    n=len(edges)
    value=edges['score'][:,:,0][:,cat.order]
    maximum=np.maximum.reduceat(value,cat.starts,axis=1)
    size=min(WIDTH//2,maximum.shape[1])
    first=np.argsort(-maximum,axis=1,kind='stable')[:,:size]
    # Average burst length is small. argmax by night can be obtained in
    # one segment reduction; the earliest shutter breaks exact ties.
    counts=np.diff(np.r_[cat.starts,value.shape[1]])
    pos=np.where(value==np.repeat(maximum,counts,axis=1),np.arange(value.shape[1]),value.shape[1])
    best=np.minimum.reduceat(pos,cat.starts,axis=1)
    best=2*cat.order[np.take_along_axis(best,first,axis=1)]
    flat=edges.reshape(n,-1)
    priority=flat['score'].copy()
    np.put_along_axis(priority,best,-1.e19,axis=1)
    k=min(WIDTH-size,priority.shape[1]-size)
    tail=np.argpartition(-priority,k-1,axis=1)[:,:k] if k else best[:,:0]
    # Small synthetic or sparse fields may use fewer than 32 shutters.
    out=np.zeros((n,WIDTH),EDGE);out['other']=-1;out['score']=-1.e18
    order=np.column_stack((best,tail))
    out[:,:order.shape[1]]=np.take_along_axis(flat,order,axis=1)
    return out


class Host:
    def __init__(self,cat):
        self.cat=cat
        self.lib=build_library("shutter_ranging")
        p,i,d=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
        self.lib.shutter_edges.argtypes=[p]*5+[i,i,i,d,d,p]
        self.lib.shutter_intersect.argtypes=[p]*4+[i,p]
        self.lib.shutter_edges.restype=None;self.lib.shutter_intersect.restype=None
        self.lib.shutter_preview.argtypes=[p]*6+[i,i,d,d,p]
        self.lib.shutter_preview.restype=None

    def edges(self,roots,family):
        cat=self.cat;out=np.empty((len(roots),len(cat.view),2),EDGE)
        self.lib.shutter_edges(*(x.ctypes.data for x in (cat.point,cat.view,cat.grid,cat.lookup,roots)),
                          len(roots),len(cat.view),family,*cat.direction,out.ctypes.data)
        return out

    def intersect(self,roots,edges):
        out=np.empty(edges.shape,SEED)
        self.lib.shutter_intersect(*(x.ctypes.data for x in (self.cat.point,self.cat.view,roots,edges)),
                                   len(roots),out.ctypes.data)
        return out

    def groups(self,roots,family):
        return self.intersect(roots,select_edges(self.cat,self.edges(roots,family)))

    def preview_edges(self,roots,family,schedule):
        out=np.empty((len(roots),WIDTH),EDGE)
        cat=self.cat
        self.lib.shutter_preview(*(a.ctypes.data for a in (cat.point,cat.view,cat.grid,cat.lookup,roots,schedule)),
                                 len(roots),family,*cat.direction,out.ctypes.data)
        return out

    def preview(self,roots,family,schedule):
        return self.intersect(roots,self.preview_edges(roots,family,schedule))


class Device:
    def __init__(self,cat):
        import torch
        from apparent_backend import cuda_kernel
        if torch.cuda.mem_get_info()[0]<600000000:
            raise MemoryError("shutter device workspace in use")
        self.cat=cat
        self.tree=[self.upload(a) for a in (cat.point,cat.view,cat.grid,cat.lookup)]
        self.east=self.upload(np.asarray(cat.direction,'f8'))
        days,inverse=np.unique(cat.view['night'],return_inverse=True)
        self.days=len(days)
        self.night=torch.as_tensor(inverse.astype('i8'),device='cuda')
        self.shutters=torch.arange(len(cat.view),device='cuda')
        source=Path(__file__).with_suffix(".cpp").read_text()
        self.pair=cuda_kernel(source,"shutter_edges")
        self.common=cuda_kernel(source,"shutter_intersect")
        reference=Host(cat)
        # Test actual field geometry and fitted ranges, including a partial
        # final block. Failures switch this job back to the scalar work item.
        roots=np.ascontiguousarray(cat.anchors[:11])
        got=self.edges(roots,0);ref=reference.edges(roots,0)
        good=ref["other"]>=0
        if not np.array_equal(got["other"],ref["other"]) or not np.allclose(
                  got["score"][good],ref["score"][good],atol=.002,rtol=3.e-5):
            raise ArithmeticError("device shutter displacements differ from host")
        e=select_edges(cat,ref);a=self.intersect(roots,e);b=reference.intersect(roots,e)
        good=b["row"][:,:,0]>=0
        if not np.array_equal(a["row"],b["row"]) or not np.allclose(
                       a["g"][good],b["g"][good],atol=1.e-6):
            raise ArithmeticError("device shutter intersections differ from host")
        tri=b['tri']['row'][:,:,0]>=0
        if not np.array_equal(a['tri']['row'],b['tri']['row']) or not np.allclose(
                  a['tri']['score'][tri],b['tri']['score'][tri],atol=.005,rtol=.0001):
            raise ArithmeticError("device three-visit forecasts differ from host")

    @staticmethod
    def upload(arr):
        import torch
        return torch.as_tensor(arr.view("u1").reshape(-1),device="cuda")

    def device_edges(self,roots,n,family):
        import torch
        ne=len(self.cat.view)
        out=torch.empty(n*ne*2*EDGE.itemsize,device="cuda",dtype=torch.uint8)
        self.pair(grid=((n*ne+127)//128,1,1),block=(128,1,1),args=[
            *self.tree,roots,n,ne,family,self.east,out])
        return out

    def edges(self,roots,family):
        n=len(roots)
        out=self.device_edges(self.upload(roots),n,family)
        return out.cpu().numpy().view(EDGE).reshape(n,len(self.cat.view),2)

    def device_intersect(self,roots,edges,n):
        import torch
        out=torch.empty(n*WIDTH*SEED.itemsize,device="cuda",dtype=torch.uint8)
        self.common(grid=((n*WIDTH+63)//64,1,1),block=(64,1,1),args=[
            self.tree[0],self.tree[1],roots,edges,n,out])
        return out.cpu().numpy().view(SEED).reshape(n,WIDTH)

    def intersect(self,roots,edges):
        return self.device_intersect(self.upload(roots),self.upload(edges),len(roots))

    def prepare_preview(self,schedule):
        from apparent_backend import cuda_kernel
        self.shortlist=cuda_kernel(Path(__file__).with_suffix('.cpp').read_text(),'shutter_preview')
        self.schedule=self.upload(schedule)
        roots=np.ascontiguousarray(self.cat.anchors[:13])
        raw=self.preview_edges(self.upload(roots),len(roots),1).cpu().numpy().view(EDGE).reshape(-1,WIDTH)
        ref=Host(self.cat).preview_edges(roots,1,schedule)
        good=ref['other']>=0
        if not np.array_equal(raw['other'],ref['other']) or not np.allclose(
                      raw['score'][good],ref['score'][good],atol=.003,rtol=.00005):
            raise ArithmeticError('device selected shutters differ from host')

    def preview_edges(self,roots,n,family):
        import torch
        out=torch.empty(n*WIDTH*EDGE.itemsize,device='cuda',dtype=torch.uint8)
        self.shortlist(grid=((24*n+127)//128,1,1),block=(128,1,1),args=[
                    *self.tree,roots,self.schedule,n,family,self.east,out])
        return out

    def preview(self,roots,family,schedule):
        active=self.upload(roots);n=len(roots)
        return self.device_intersect(active,self.preview_edges(active,n,family),n)

    def groups(self,roots,family):
        import torch
        n,ne=len(roots),len(self.cat.view)
        if ne*2<WIDTH:
            return self.intersect(roots,select_edges(self.cat,self.edges(roots,family)))
        active=self.upload(roots)
        edge=self.device_edges(active,n,family).view(torch.int32).reshape(n,2*ne,4)
        score=edge.view(torch.float32)[:,:,1].contiguous()
        nightly=torch.full((n,self.days),-1.e20,dtype=score.dtype,device=score.device)
        groups=self.night.expand(n,-1)
        nightly.scatter_reduce_(1,groups,score[:,::2],reduce='amax')
        present=torch.where(nightly.gather(1,groups)==score[:,::2],self.shutters,ne)
        indices=torch.full((n,self.days),ne,dtype=torch.int64,device=score.device)
        indices.scatter_reduce_(1,groups,present,reduce='amin')
        reserved=min(self.days,WIDTH//2)
        first=2*indices.gather(1,nightly.topk(reserved,1).indices)
        score.scatter_(1,first,-1.e20)
        retain=torch.cat((first,score.topk(WIDTH-reserved,1).indices),dim=1)
        reduced=edge.gather(1,retain[:,:,None].expand(-1,-1,4)).contiguous()
        return self.device_intersect(active,reduced,n)


def unique(records, cap):
    if not len(records):return records
    order=np.argsort(-records['score'],kind='stable')
    _,first=np.unique(np.sort(records['row'][order],axis=1),axis=0,return_index=True)
    return records[order[np.sort(first)[:cap]]]


def scan(bank,index,ids,direction,control,threads,deadline,device):
    cat=Catalogue(bank,ids,direction)
    engine=Host(cat);batch=48
    if device=='cuda' and len(cat.anchors):
        try:
            engine=Device(cat)
            batch=max(80,min(480,260000//len(cat.view)))
        except Exception as ex:
            print("scalar shutter search:",type(ex).__name__,str(ex)[:140],flush=True)
    start=time.monotonic()
    schedule=None;preview_engine=engine;preview_order=None
    if len(cat.anchors)>400 and deadline-start>18.:
        import visit_preview
        schedule=visit_preview.schedules(cat,1)
        preview_order=visit_preview.root_order(cat)
        if isinstance(engine,Device):
            try:engine.prepare_preview(schedule)
            except (RuntimeError,ArithmeticError,MemoryError,OSError,ValueError) as exc:
                print('selected shutters on host:',str(exc)[:140],flush=True)
                preview_engine=Host(cat)
    plans=((1,.33,True),(0,.59,False),(1,.84,False),(2,.95,False),(3,1.,False)) if schedule is not None else (
             (0,.44,False),(1,.80,False),(2,.94,False),(3,1.,False))
    for family,share,preview in plans:
        end=start+(deadline-start)*share
        ranking=preview_order if preview else cat.anchors
        chunk=(1536 if isinstance(preview_engine,Device) else 96) if preview else batch
        pos=0;parts=[];triples=[];found=0;tested=0;added=0;verified=0
        def confirm(seconds):
            nonlocal verified,tested,added
            if seconds<.05 or (not parts and not triples):return
            finish=time.monotonic()+seconds
            if triples:
                candidates=unique(np.concatenate(triples),9000)
                sky=control.get('original_sky')
                cores=None
                if sky is not None and 'shutter_rows' in control and seconds>1.:
                    from original_recovery import CORE
                    cores=np.zeros(len(candidates),CORE)
                seeds,nt=bank.extend(candidates,threads,.48*seconds,14000,cores)
                tested+=nt;added+=len(seeds)
                if len(seeds):
                    more=np.zeros(len(seeds),SEED)
                    more['row']=seeds['rows'];more['g']=seeds['g'];more['score']=seeds['score']
                    parts.append(more)
                triples.clear()
                if cores is not None:
                    core=cores[:nt][cores['count'][:nt]==3]
                    rank=np.argsort(-core['merit'],kind='stable')
                    core=core[rank[:700]]
                    if len(core):
                        extra=control['additional']
                        core=core[np.max(core['rows'][:,:3],axis=1)<extra['diff_start']]
                        rows=core['rows'][:,:3]
                        base=rows<extra['offset']
                        rows[base]=control['shutter_rows'][rows[base]]
                        rows[~base]=extra['rows'][rows[~base]-extra['offset']]
                        core['merit']=np.clip(.20*core['merit'],-2.,5.)
                        if len(core):
                            raw,_=sky.recover(core,min(.20*seconds,max(.05,finish-time.monotonic())),1200)
                            if len(raw):control['raw_results'].put((raw,[]))
            if not parts:return
            z=unique(np.concatenate(parts),7500)
            if len(z):
                result=certify(index,z['row'][:,:4],z['g'],z['row'][:,4],threads,
                               max(.15,finish-time.monotonic()),2700)
                control['certified'].put(result);verified+=int((result['nights']>=5).sum())
            parts.clear()
        last=time.monotonic()
        while pos<len(ranking) and not control['event'].is_set() and time.monotonic()+.3<end:
            roots=np.ascontiguousarray(ranking[pos:pos+chunk])
            try:
                z=preview_engine.preview(roots,family,schedule) if preview else engine.groups(roots,family)
            except RuntimeError as exc:
                if not isinstance(preview_engine if preview else engine,Device):raise
                print('resuming shutter queries on host:',str(exc)[:140],flush=True)
                engine=preview_engine=Host(cat);batch=chunk=48
                continue
            pos+=len(roots)
            tr=z['tri']
            tr=tr[tr['row'][:,:,0]>=0]
            if len(tr):triples.append(tr)
            z=z[z['row'][:,:,0]>=0]
            if len(z):parts.append(z);found+=len(z)
            if time.monotonic()-last>8. or sum(map(len,triples))>125000:
                confirm(min(4.8,max(0.,end-time.monotonic())))
                last=time.monotonic()
        if not control['event'].is_set():confirm(min(4.,max(.3,deadline-time.monotonic())))
        print("measured shutter graph:",'selected' if preview else 'all',family,pos,"anchors,",found,"coincidences,",verified,
              "verified five-night orbits,",tested,"three-visit arcs yielded",added,"returns",flush=True)

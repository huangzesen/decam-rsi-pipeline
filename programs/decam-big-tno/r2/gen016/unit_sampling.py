"""Bound unit memory without letting crowded night pairs consume every slot."""
import numpy as np
import torch


def sample(ids,pair,limit):
    if len(ids)<=limit:return ids
    code=pair[ids]
    order=torch.argsort(code,stable=True)
    codes,counts=torch.unique_consecutive(code[order],return_counts=True)
    sizes=counts.cpu().numpy();quota=np.zeros(len(sizes),dtype=np.int64)
    left=int(limit)
    # Square-root allocation is a compromise between equal pair coverage and
    # sampling the much larger background of crowded night pairs uniformly.
    while left:
        active=np.flatnonzero(sizes>quota)
        if not len(active):break
        weight=np.sqrt(sizes[active])
        wanted=left*weight/weight.sum()
        add=np.minimum(sizes[active]-quota[active],np.floor(wanted).astype(np.int64))
        if not add.any():
            pick=active[np.argsort(-(wanted-np.floor(wanted)),kind='stable')[:left]]
            quota[pick]+=1;left-=len(pick)
        else:
            quota[active]+=add;left-=int(add.sum())
    q=torch.as_tensor(quota,device=ids.device)
    group=torch.repeat_interleave(torch.arange(len(q),device=ids.device),q)
    local=torch.arange(int(q.sum()),device=ids.device)-torch.repeat_interleave(torch.cumsum(q,0)-q,q)
    begin=torch.cumsum(counts,0)-counts
    offset=local*(counts[group]-1)//(q[group]-1).clamp(min=1)
    # For singleton allocations sample the centre, not the first sky coordinate.
    offset=torch.where(q[group]==1,counts[group]//2,offset)
    return ids[order[begin[group]+offset]]

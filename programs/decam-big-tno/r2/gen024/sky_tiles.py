"""Faint seasonal search in overlapping heliocentric sky tiles.

Tiling bounds pair-list memory without raising the signal-to-noise threshold
across an entire observing window. The tile overlap follows the admissible
orbital speed, window duration, and range-block width. Four shifted grids cover
any track smaller than half a tile in each coordinate. The caller fits and
confirms candidate arcs from the resulting subwindows.
"""
import math
import time
import numpy as np
import torch


def tiles(F,W,P,block):
    G=F.G;ids=W['idx'];e=G['de'][ids];theta=G['th'][ids];obs=G['ob'][e]
    g=block[0];t=G['et'][e]-W['tref']-1/(F.C_AU*g)
    u=1-.5*F.MU*g**3*t*t
    circular=math.sqrt(F.MU*g**3)
    xy=(theta*(u-g*obs[:,2])[:,None]+g*obs[:,:2]-.8*circular*G['epro'][None]*t[:,None])/u[:,None]
    # A generous radial-perspective margin also covers the small gravity
    # approximation and finite measurement errors in proposal construction.
    halo=1.35*.75*circular*(P['h']+3)+.02*g+8*F.AS
    width=max(math.radians(1.),4.05*halo)
    result=[]
    for sx,sy in ((0.,0.),(.5,0.),(0.,.5),(.5,.5)):
        cell=torch.floor(xy/width+torch.tensor([sx,sy],device=ids.device)).long()
        key=(cell[:,0]+1048576)*2097152+(cell[:,1]+1048576)
        order=torch.argsort(key,stable=True);key=key[order]
        starts=torch.nonzero(torch.cat((torch.ones(1,dtype=torch.bool,device=ids.device),key[1:]!=key[:-1])),as_tuple=True)[0]
        starts=starts.cpu().numpy();ends=np.r_[starts[1:],len(ids)]
        for a,b in zip(starts,ends):
            if b-a<5:continue
            sel=order[int(a):int(b)]
            if len(torch.unique(W['night'][sel]))<3:continue
            sub=dict(W,idx=ids[sel],night=W['night'][sel]);sub.pop('pc',None)
            result.append(sub)
    return result


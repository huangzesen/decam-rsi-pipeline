"""Astrometric evidence of a three-night parallax triangle before orbit fitting.

A six-parameter fit can nearly interpolate three two-coordinate positions.
Its small residual is then weak evidence. The component perpendicular to the
parallax curvature has one measurable constraint. Marginalize the radial-motion
term with the same physical velocity prior used by the orbit fitter.
"""
import numpy as np


def penalty(F, p):
    G=F.G
    t=G['et_np'][G['de_np'][p]]
    order=np.argsort(t,axis=1)
    p=np.take_along_axis(p,order,axis=1)
    t=np.take_along_axis(t,order,axis=1)
    th=np.asarray(G['th_np'])[p]
    ob=G['ob'].detach().cpu().numpy()[G['de_np'][p]]
    d=ob[:,:,:2]-th*ob[:,:,2:3]
    ta=t[:,0]-t[:,1];tc=t[:,2]-t[:,1]
    span=np.maximum(tc-ta,1e-9);h=tc/span
    A=th[:,1]-h[:,None]*th[:,0]-(1-h[:,None])*th[:,2]
    B=d[:,1]-h[:,None]*d[:,0]-(1-h[:,None])*d[:,2]
    bb=np.maximum(np.sum(B*B,axis=1),1e-25)
    gamma=-np.sum(A*B,axis=1)/bb
    perpendicular=A+gamma[:,None]*B
    C=h[:,None]*ta[:,None]*(th[:,0]-th[:,1])+(1-h[:,None])*tc[:,None]*(th[:,2]-th[:,1])
    C-=B*(np.sum(C*B,axis=1)/bb)[:,None]
    sn=np.maximum(np.asarray(G['snr_np'])[p],4.5)
    var=(1.1/sn)**2+.04**2
    centroid=var[:,1]+h*h*var[:,0]+(1-h)**2*var[:,2]
    radial=.3**2*F.MU*np.clip(gamma,1/1400,1/12)**3*np.sum(C*C,axis=1)/F.AS**2
    total=centroid+radial
    chi=np.sum(perpendicular*perpendicular,axis=1)/F.AS**2/total
    # The width term is the marginal likelihood's normalization. A flexible
    # radial velocity must not manufacture evidence by making the fit easier.
    return chi+np.log(total/centroid)

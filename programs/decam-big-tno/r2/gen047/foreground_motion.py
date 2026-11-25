"""Assess alternate moving sources in nearby visits of a reported orbit."""
import ctypes

import numpy as np


QUERY = np.dtype([('x','f8',6),('ref','f8'),('row','i4'),('group','i4')],align=True)
RESULT = np.dtype([('fast','f4'),('slow','f4'),('partner','i4'),('confirm','i4'),
                   ('slow_row','i4'),('slow_e','i4')],align=True)
BLEND = np.dtype([('star','f4'),('excess','f4'),('reference','f4'),('nights','i4')])


def evaluate(F, items, sky, seconds):
    if not items or sky is None or not hasattr(sky.lib,'foreground_motion'):return
    lengths = np.array([len(v['d']) for v in items])
    groups = np.repeat(np.arange(len(items)),lengths)
    q = np.empty(len(groups),QUERY)
    q['row'] = np.concatenate([v['d'] for v in items])
    q['group'] = groups
    q['x'] = np.stack([v['x'] for v in items])[groups]
    q['ref'] = np.array([v['ref'] for v in items])[groups]
    pole = F.G['frame_np']@np.array([0.,-.397778,.917482])
    east = np.ascontiguousarray([pole[1],-pole[0]],'f8')
    east /= max(1.e-9,np.linalg.norm(east))
    out = np.zeros(len(q),RESULT)
    out['partner'] = out['confirm'] = out['slow_row'] = -1
    p,i,d = ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    func = sky.lib.foreground_motion
    func.argtypes = [p,p,i,p,i,d,p];func.restype = i
    count = func(sky.handle,q.ctypes.data,len(q),east.ctypes.data,2,max(.02,.55*seconds),out.ctypes.data)
    if count<0:raise RuntimeError('foreground motion measurement failed')
    predictions = np.ascontiguousarray(np.concatenate([v['mr']+v['color']*F.G['lev_e'][v['e']] for v in items]),'f8')
    blend = np.zeros(len(q),BLEND)
    if hasattr(sky.lib,'stationary_flux'):
        compare = sky.lib.stationary_flux
        compare.argtypes = [p,p,p,i,i,d,p];compare.restype = i
        done = compare(sky.handle,q.ctypes.data,predictions.ctypes.data,len(q),2,max(.02,.45*seconds),blend.ctypes.data)
        if done<0:raise RuntimeError('stationary flux comparison failed')
    start = 0
    for v,n in zip(items,lengths):
        a = out[start:start+n]
        v['static_flux'] = blend[start:start+n]
        start += n
        v['local_motion'] = a


def odds(v):
    a = v.get('local_motion')
    if a is None:return 0.,0.
    # Real unrelated fast movers are a fraction of even the transient
    # catalogue. An orbit trial may also get several near-epoch opportunities.
    fast = np.maximum(0.,np.asarray(a['fast'],'f8')-7.5)
    slow = np.maximum(0.,np.asarray(a['slow'],'f8')-3.)
    return float(fast.max(initial=0.)),float(slow.max(initial=0.))

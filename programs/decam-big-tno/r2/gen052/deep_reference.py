"""Estimate stationary-source contamination from the deepest local visits."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


VISIT = np.dtype([(k, 'f8') for k in ('time','zp','colour','limit','seeing')]
                 + [('band','i4'), ('pad','i4')])
COLOUR = dict(g=.65, r=0., i=-.26, z=-.44, Y=-.48, VR=.11)


def measure(scen, exposures, x, y, zero, eligible):
    raw = lambda key: np.load(Path(scen) / ('det_' + key + '.npy'), mmap_mode='r')
    exp = raw('exp')
    edges = np.searchsorted(exp, np.arange(len(exposures)+1)).astype('i8')
    visits = np.zeros(len(exposures), VISIT)
    bands = [str(e.get('band')) for e in exposures]
    visits['time'] = [e['mjd_mid'] for e in exposures]
    visits['colour'] = [COLOUR.get(b,0.) for b in bands]
    visits['band'] = np.unique(bands,return_inverse=True)[1]
    mz = np.array([e.get('magzero') for e in exposures], float)
    zero = mz if zero is None else zero
    visits['zp'] = zero-visits['colour']
    visits['limit'] = np.array([e.get('m5') for e in exposures],float)+(zero-mz)-visits['colour']
    seeing = np.array([e.get('fwhm') for e in exposures],float)
    visits['seeing'] = np.where((seeing>.3)&(seeing<4.),seeing,1.1)
    shape = raw('sharp')
    args = [np.ascontiguousarray(x,'f8'), np.ascontiguousarray(y,'f8'), raw('flux'),
            raw('snr'),raw('ferr'), shape,raw('x'),raw('y'),raw('flag'),edges,visits]
    eligible = np.ascontiguousarray(eligible,bool)
    odds = np.zeros(len(x),'f4')
    recovered = np.zeros(len(x),bool)
    with tempfile.TemporaryDirectory(prefix='deep_visits_') as td:
        cc = next((c for k in ('c++','g++','clang++') if (c:=shutil.which(k))),None)
        if cc is None:raise RuntimeError('C++17 compiler unavailable')
        dest = Path(td)/'deep.so'
        subprocess.run([cc,'-std=c++17','-O3','-shared','-fPIC','-pthread',
                        str(Path(__file__).with_suffix('.cpp')),'-o',str(dest)],
                       check=True,capture_output=True,timeout=45)
        native = ctypes.CDLL(str(dest)).deep_visits
        p,i = ctypes.c_void_p,ctypes.c_int
        native.argtypes = [p]*11+[i,p,i,p,p]
        native.restype = i
        result = native(*(a.ctypes.data for a in args),len(exposures),eligible.ctypes.data,
                        4,odds.ctypes.data,recovered.ctypes.data)
        if result<0:raise RuntimeError('failed to index comparison visits')
    return odds,recovered


def choose(odds, recovered, keep, exp, snr, sharp, ne):
    """Reserve a bounded fraction of each visit for independently visible excesses."""
    ids = np.flatnonzero(recovered & ~keep)
    result = np.zeros(len(keep),bool)
    if not len(ids):return result
    per = np.bincount(exp[keep],minlength=ne)
    median = np.median(per[per>0]) if np.any(per) else 0.
    cuts = np.searchsorted(exp[ids],np.arange(ne+1))
    for e in range(ne):
        rows = ids[cuts[e]:cuts[e+1]]
        cap = max(35,int(.08*min(2.*median,per[e]+.5*median)))
        if len(rows)>cap:
            sn = snr[rows]
            dz = (sharp[rows]-1.02)/np.hypot(.18,3.5/sn)
            merit = odds[rows]+.64*np.log(np.minimum(20.,sn)/5.)-.34*dz*dz
            rows = rows[np.argpartition(-merit,cap-1)[:cap]]
        result[rows] = True
    return result

"""Join near-night pairs at a shared detection before raw-catalogue follow-up."""
import ctypes
import numpy as np

CHAIN=np.dtype([('row','i4',4),('g','f4'),('merit','f4')])


def search(lib,target,ex,signal,pairs,threads,seconds):
    from range_consensus import units
    bank=units(pairs);signal=np.ascontiguousarray(signal,'f4')
    ptr,integer,real=ctypes.c_void_p,ctypes.c_int,ctypes.c_double
    fn=lib.shared_night_chains
    fn.argtypes=[ptr,integer,ptr,ptr,ptr,integer,integer,real,integer,ptr,ptr];fn.restype=integer
    out=np.empty(min(36000,max(2000,len(bank)//6)),CHAIN);stats=np.zeros(4,'i8')
    n=fn(target.ctypes.data,len(target),ex.ctypes.data,signal.ctypes.data,bank.ctypes.data,len(bank),
         threads,seconds,len(out),out.ctypes.data,stats.ctypes.data)
    if n<0:raise RuntimeError('night-pair graph join failed')
    return out[:n],stats

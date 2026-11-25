"""Exact nearest measurements in a compact directory of camera columns."""
import ctypes
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import torch


@lru_cache(maxsize=3)
def implementation(device):
    path = Path(__file__).with_suffix('.cpp')
    if device.type == 'cuda':
        from apparent_backend import cuda_kernel
        return cuda_kernel(path.read_text(), 'exposure_nearest')
    compiler = shutil.which('c++') or shutil.which('g++')
    with tempfile.TemporaryDirectory(prefix='catalog_columns_') as build:
        library = Path(build)/'columns.so'
        subprocess.run([compiler, '-O3', '-std=c++17', '-ffp-contract=off', '-fPIC',
                        '-shared', str(path), '-o', str(library)],
                       check=True, capture_output=True, timeout=30)
        fn = ctypes.CDLL(str(library)).exposure_nearest
    fn.argtypes = [ctypes.c_void_p]*10+[ctypes.c_int]*2+[ctypes.c_double]*2
    fn.restype = None
    return fn


class Columns:
    def __init__(self, reference, xy, cells, exposure, order):
        self.device = reference.th.device
        self.kernel = implementation(self.device)
        self.bounds = reference.bounds.contiguous()
        self.origin = reference.lo_t.contiguous()
        self.pitch = float(reference.S)
        # Each exposure pays for only its own span in x. A sentinel at the
        # end of each column range also handles exposures with no sources.
        e = exposure[order]
        cx = cells[order,0]
        starts = np.flatnonzero(np.r_[True,e[1:] != e[:-1]])
        ends = np.r_[starts[1:], len(e)]
        directory = np.zeros((len(self.bounds), 3), 'i4')
        directory[e[starts],0] = cx[starts]
        directory[e[starts],1] = cx[ends-1]+1
        sizes = directory[:,1]-directory[:,0]
        directory[:,2] = np.cumsum(sizes+1, dtype='i8')-sizes-1
        position = directory[e,2] + cx - directory[e,0]
        columns = np.bincount(position, minlength=int(sum(sizes+1)))
        offsets = np.cumsum(columns, dtype='i4') - columns.astype('i4')
        def tensor(value):
            return torch.as_tensor(np.ascontiguousarray(value), device=self.device)
        self.tables = [self.bounds, tensor(directory), tensor(offsets),
                       tensor(cells[order,1].astype('i4')), tensor(order.astype('i4')),
                       tensor(xy[order].astype('f8',copy=False)), self.origin]

    def query(self, qe, qth, tol):
        n = qe.numel()
        out = torch.empty(n, dtype=torch.int64, device=self.device)
        if not n:
            return out
        tensors = [qe.to(torch.int64).contiguous(), qth.to(torch.float64).contiguous(),
                   *self.tables, out]
        args = [len(self.bounds), n, self.pitch, float(tol)]
        if self.device.type == 'cuda':
            self.kernel(grid=(min(65535,(n+127)//128),1,1), block=(128,1,1),
                        args=tensors+args)
        else:
            self.kernel(*(t.data_ptr() for t in tensors), *args)
        return out

    def validate(self, reference):
        which = torch.linspace(0,len(reference.key)-1,min(600,len(reference.key)),
                               device=self.device).long()
        e = reference.key[which] >> 42
        xy = reference.th[reference.ord[which]].clone()
        phase = torch.arange(len(e),device=self.device,dtype=xy.dtype)
        xy[:,0] += .48*self.pitch*torch.sin(phase)
        xy[:,1] += .36*self.pitch*torch.cos(.47*phase)
        for tol in (.23*self.pitch,1.17*self.pitch):
            want = reference._query_cells(e,xy,tol)
            got = self.query(e,xy,tol)
            valid = got>=0
            if not torch.equal(valid,want>=0):
                raise ArithmeticError('exposure column lookup missed a measurement')
            distances = lambda rows: (reference.th[rows[valid]]-xy[valid]).square().sum(1)
            if not torch.allclose(distances(want),distances(got),rtol=1.e-12,atol=1.e-25):
                raise ArithmeticError('wrong nearest measurement in exposure column')

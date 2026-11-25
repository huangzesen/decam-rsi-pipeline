"""Prefit all five measured directions before spending an archive-wide grow on a seed."""
import ctypes
from pathlib import Path

import numpy as np

from companion_graph import build_library

SAMPLE = np.dtype([('xy','f4',(5,2)), ('exposure','i4',5), ('gamma','f4')])


class Screen:
    def __init__(self, sky, device):
        self.sky = sky
        self.visits = np.ascontiguousarray(np.column_stack([sky.geometry[k] for k in ('t','ox','oy','oz')]), 'f8')
        self.native = build_library('night_screen').fit_five_positions_host
        self.native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.native.restype = None
        self.device = None
        if str(device).startswith('cuda'):
            import torch
            from apparent_backend import cuda_kernel
            try:
                root = Path(__file__).parent
                src = (root/'night_screen.cpp').read_text().replace('#include "quartet_fit.hpp"',
                                                                    (root/'quartet_fit.hpp').read_text())
                self.device = cuda_kernel(src,'fit_five_positions')
                self.stream = torch.cuda.Stream()
                with torch.cuda.stream(self.stream):
                    self.observer = torch.as_tensor(self.visits,device='cuda')
            except (OSError, RuntimeError, AttributeError) as e:
                print('five-position fitter on host:',str(e)[:135],flush=True)
                self.device = None

    def host(self, samples):
        out = np.empty((len(samples),2),'f4')
        self.native(samples.ctypes.data,self.visits.ctypes.data,len(samples),out.ctypes.data)
        return out

    def __call__(self,trials,rows):
        samples = np.empty(len(rows),SAMPLE)
        samples['xy'] = self.sky.pos[rows]
        samples['exposure'] = self.sky.exposure[rows]
        samples['gamma'] = trials['gamma']
        if self.device is not None:
            import torch
            try:
                with torch.cuda.stream(self.stream):
                    data = torch.as_tensor(samples.view('u1'),device='cuda')
                    result = torch.empty((len(samples),2),dtype=torch.float32,device='cuda')
                    self.device(grid=((len(samples)+63)//64,1,1),block=(64,1,1),
                                args=[data,self.observer,len(samples),result])
                    fit = result.cpu().numpy()
                if not hasattr(self,'checked'):
                    ref = self.host(samples[:192])
                    if not np.allclose(fit[:len(ref)],ref,rtol=.002,atol=.001):
                        raise ArithmeticError('host and device orbit residuals disagree')
                    self.checked = True
                return fit
            except (OSError,RuntimeError,ArithmeticError) as e:
                print('five-position fits continuing on host:',str(e)[:135],flush=True)
                self.device = None
        return self.host(samples)

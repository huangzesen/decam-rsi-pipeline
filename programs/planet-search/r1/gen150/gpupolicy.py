"""gen 148: which torch device a process may use.

gen147 scored 0.0376 on Lonestar6 (gen146 = same code but one ranking change: 0.2707 on Perlmutter).  Up to ~12
processes (linker detection workers, linker main, stackd map workers, stackd main) each create their own CUDA
context.  On a GPU in EXCLUSIVE_PROCESS compute mode (common on TACC nodes) only ONE process can hold a context: every
other process gets "CUDA-capable device(s) busy or unavailable" on its first CUDA op, and detect.py / mapper.py
swallow that per CCD (`except Exception: continue`), so the linker and stackd see (almost) no data.

Policy (identical copies of this file live in linker/ and stackd/; each subprocess imports its own):
  * TNO_DEV_FORCE=cpu|mps|cuda forces a device (testing).
  * TNO_GPU_EXCL=1 (set by the top-level finder.py when nvidia-smi / NVML reports a non-Default compute mode, or by
    hand on the Mac to simulate it): only role "main" (stackd's main process, the heavy GPU user) uses the
    accelerator; workers and the linker run on the CPU.
  * Otherwise every process may use CUDA, but only after a smoke test (allocate, compute, synchronize) succeeds;
    on failure it falls back to the CPU instead of silently losing every CCD.  role "main" retries for up to
    TNO_GPU_WAIT s (default 20) in case another process still holds an exclusive context.
With a Default-mode GPU (Perlmutter) and a working driver the result is exactly the old choice.
"""
import os
import sys
import time

_CACHE = {}


def _log(*a):
    try:
        print("[gpupolicy]", *a, file=sys.stderr, flush=True)
    except Exception:
        pass


def device_error(ex):
    """True for errors that look like the accelerator's fault (not the data's)."""
    m = (type(ex).__name__ + " " + str(ex)).lower()
    return any(k in m for k in ("cuda", "cudnn", "cublas", "device", "out of memory", "mps", "nvrtc", "driver",
                                "acceleratorerror"))


def _ops_ok(torch, dev):
    """The ops the map/detection workers use (cuDNN conv, bilinear upsample, max pool) on a tiny image."""
    F = torch.nn.functional
    x = torch.ones(1, 1, 32, 32, device=dev, dtype=torch.float32)
    k = torch.ones(1, 1, 1, 7, device=dev, dtype=torch.float32)
    y = F.conv2d(x, k, padding=(0, 3))
    y = F.interpolate(y, size=(40, 40), mode="bilinear", align_corners=False)
    y = F.max_pool2d(y, 5, 1, 2)
    v = float(y.max().item())
    return abs(v - 7.0) < 1e-3


def _cuda_ok(torch):
    try:
        x = torch.arange(64, device="cuda", dtype=torch.float32)
        y = float((x * 2.0).sum().item())
        ok = abs(y - 4032.0) < 1e-3 and _ops_ok(torch, "cuda")
        torch.cuda.synchronize()
        if not ok:
            _log("cuda smoke test gave wrong values")
        return ok
    except Exception as ex:
        _log("cuda smoke test failed:", repr(ex)[:300])
        return False


def _mps_ok(torch):
    try:
        x = torch.arange(64, device="mps", dtype=torch.float32)
        return abs(float((x * 2.0).sum().item()) - 4032.0) < 1e-3 and _ops_ok(torch, "mps")
    except Exception as ex:
        _log("mps smoke test failed:", repr(ex)[:300])
        return False


def pick(torch, role="worker", order=("cuda", "mps")):
    """Return 'cuda', 'mps' or 'cpu' for this process.  role: 'main' (stackd main), 'aux' (linker main), 'worker'."""
    key = role
    if key in _CACHE:
        return _CACHE[key]
    dev = "cpu"
    force = os.environ.get("TNO_DEV_FORCE", "").strip().lower()
    excl = os.environ.get("TNO_GPU_EXCL", "0") == "1"
    try:
        if force in ("cpu", "mps", "cuda"):
            dev = force
        elif excl and role != "main":
            dev = "cpu"
        else:
            for d in order:
                if d == "cuda" and torch.cuda.is_available():
                    ok = _cuda_ok(torch)
                    if not ok and role == "main":
                        t_end = time.time() + float(os.environ.get("TNO_GPU_WAIT", "20") or 20)
                        while not ok and time.time() < t_end:
                            time.sleep(2.0)
                            ok = _cuda_ok(torch)
                    if ok:
                        dev = "cuda"
                        break
                elif d == "mps" and torch.backends.mps.is_available():
                    if _mps_ok(torch):
                        dev = "mps"
                        break
    except Exception as ex:
        _log("device choice failed:", repr(ex)[:300])
        dev = "cpu"
    _CACHE[key] = dev
    if dev == "cpu" and role == "main" or excl or force:
        _log(f"role {role}: device {dev} (excl={excl}, force={force or '-'})")
    return dev


def detect_exclusive():
    """True when an NVIDIA GPU on this node is not in Default compute mode (nvidia-smi, else NVML via ctypes)."""
    import subprocess
    # only the GPU(s) this job can see, when CUDA_VISIBLE_DEVICES names them; only explicit exclusive / prohibited
    # modes count ("[N/A]", "[Not Supported]" etc. are treated as Default)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    cmd = ["nvidia-smi", "--query-gpu=compute_mode", "--format=csv,noheader"]
    if vis and all(p.strip() for p in vis.split(",")) and vis != "-1":
        cmd += ["--id=" + vis]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            modes = [m.strip() for m in r.stdout.strip().splitlines() if m.strip()]
            _log("nvidia-smi compute modes:", modes)
            return any(m.lower().replace(" ", "_") in ("exclusive_process", "exclusive_thread", "prohibited")
                       for m in modes), modes
    except Exception as ex:
        _log("nvidia-smi failed:", repr(ex)[:200])
    try:
        import ctypes
        nv = ctypes.CDLL("libnvidia-ml.so.1")
        if nv.nvmlInit_v2() != 0:
            return False, None
        n = ctypes.c_uint(0)
        nv.nvmlDeviceGetCount_v2(ctypes.byref(n))
        modes = []
        for i in range(n.value):
            h = ctypes.c_void_p()
            if nv.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0:
                continue
            m = ctypes.c_int(0)
            if nv.nvmlDeviceGetComputeMode(h, ctypes.byref(m)) == 0:
                modes.append(int(m.value))
        try:
            nv.nvmlShutdown()
        except Exception:
            pass
        _log("NVML compute modes:", modes)
        return (bool(modes) and not vis and any(m in (1, 2, 3) for m in modes)), modes
    except Exception as ex:
        _log("NVML query failed:", repr(ex)[:200])
    return False, None

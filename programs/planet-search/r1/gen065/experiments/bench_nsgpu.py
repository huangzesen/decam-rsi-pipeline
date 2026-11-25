import sys, time, math, numpy as np, torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "stackd"))
import finder as Fd
dev = torch.device("mps")
N=int(sys.argv[1]); rng=np.random.default_rng(0)
Q=[rng.uniform(-0.0175,0.0175,(N,2)) for _ in range(6)]
tN=np.array([0,1,2,8,9,20.]); pairs=[(a,b) for a in range(6) for b in range(a+1,6)]
PIX=4.85e-6; W=H=7200
NMs=torch.randn((6,H*W),dtype=torch.float16,device=dev)
TD=np.eye(3); TF0=np.eye(3); oN=np.zeros((6,3))
import cProfile, pstats
t=time.time()
cProfile.run("r=Fd._ns_gpu_pairs(torch, dev, Q, pairs, tN, 10., 6.8e-5, 0., 6.8e-5, 0.75, 1e-4, 2.5*4.85e-6, TD, 40., oN, TF0, -0.0175, -0.0175, PIX, W, H, NMs, list(range(6)))", "prof.out")
torch.mps.synchronize(); print("time", time.time()-t, None if r[0] is None else len(r[0]))
pstats.Stats("prof.out").sort_stats("tottime").print_stats(8)

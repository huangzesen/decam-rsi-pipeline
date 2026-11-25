import torch, time
torch.set_num_threads(1)
F=torch.nn.functional
x=torch.randn(1,1,4094,2046)
t=time.time(); a=F.max_pool2d(x,5,1,2); print(time.time()-t)
t=time.time(); b=F.max_pool2d(F.max_pool2d(x,(1,5),1,(0,2)),(5,1),1,(2,0)); print(time.time()-t, torch.equal(a,b))
x[0,0,::7,::3]=float('nan')
a=F.max_pool2d(x,5,1,2); b=F.max_pool2d(F.max_pool2d(x,(1,5),1,(0,2)),(5,1),1,(2,0)); print(torch.equal(torch.nan_to_num(a,-99.),torch.nan_to_num(b,-99.)))

import sys, time, os, glob
sys.path.insert(0, "$RSI_SANDBOX/r1/gen148/workspace/submission/linker")
os.environ["TNO_DEV_FORCE"]="cpu"
import numpy as np, torch
torch.set_num_threads(1)
import detect, fastfits
from astropy.io import fits
sc="$DATA_DRIVE/ale-astronomy/planet-search/rsi/scen_dev/scen03"
im=sorted(glob.glob(sc+"/exposures/*ooi*"))[0]; mk=im.replace("ooi","ood")
hb=fits.open(im, disable_image_compression=True); hm=fits.open(mk, disable_image_compression=True)
img=fastfits.read_tiled(hb[1]).astype(np.float32); msk=fastfits.read_tiled(hm[1])
import cProfile, pstats
t=time.time(); detect.process_ccd(img,msk,4.0,3.0,True); print("first",time.time()-t)
pr=cProfile.Profile(); pr.enable(); t=time.time(); detect.process_ccd(img,msk,4.0,3.0,True); print("cpu",time.time()-t); pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(8)

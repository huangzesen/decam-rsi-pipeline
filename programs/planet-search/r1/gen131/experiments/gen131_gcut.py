"""crude global cut: python gcut.py INJ_OUT DEV_OUT [DEV_OUT...]: top 25*N by p over all, injCol weighted hits kept."""
import json,sys,subprocess,os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_inject as si, csv
truth=json.load(open('$RSI_SANDBOX/r1/gen131/tmp/truth_injCol.json'))
rows=list(csv.DictReader(open('$RSI_SANDBOX/r1/gen131/tmp/scen/injCol/exposures.csv')))
inj=json.load(open(sys.argv[1]))['candidates']
pool=[(c['p'],0,i) for i,c in enumerate(inj)]
for k,f in enumerate(sys.argv[2:]):
    pool+=[(c['p'],k+1,i) for i,c in enumerate(json.load(open(f))['candidates'])]
N=len(sys.argv)-1; pool.sort(key=lambda x:-x[0]); keep=pool[:25*N]
kept=[inj[i] for p,s,i in keep if s==0]
_,_,hr,dup=si.score(truth,kept,rows)
W={'A':1,'B':2,'C':4}
cls=[b.get('cls',b.get('class','?')) for b in truth['bodies']]
w=sum(W.get(cls[j],0) for j in hr)
from collections import Counter
print(sys.argv[1].split('/')[-1], 'kept from inj', len(kept), 'of', 25*N, 'cut p', round(keep[-1][0],3), 'hits', dict(Counter(cls[j] for j in hr)), 'weighted', w)

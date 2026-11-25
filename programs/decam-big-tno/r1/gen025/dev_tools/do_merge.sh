#!/bin/bash
# three-way merge of /app/submission/finder.py (gen022 + this generation) with gen023 on gen019 -> /tmp/w/merged.py
cd /tmp/w && diff3 -m /app/submission/finder.py /app/history/gen019/finder.py /app/history/gen023/finder.py > merged.py
python3 - <<'PY'
import re, ast
s=open('/tmp/w/merged.py').read()
blocks=list(re.finditer(r'<<<<<<< /app/submission/finder.py\n(.*?)\|\|\|\|\|\|\| /app/history/gen019/finder.py\n(.*?)=======\n(.*?)>>>>>>> /app/history/gen023/finder.py\n', s, re.S))
assert len(blocks)==3, len(blocks)
mine=[b.group(1) for b in blocks]; theirs=[b.group(3) for b in blocks]
r1 = theirs[0].replace("            acc = (m >= 2) & (fa1 * (0.3 / tol) ** (2 * m) * CONFFA ** n2v < fthr)\n","") + mine[0].replace("acc = (m >= 2) & (fa1 * (0.3 / tol) ** (2 * m) < fthr)","acc = (m >= 2) & (fa1 * (0.3 / tol) ** (2 * m) * CONFFA ** n2v < fthr)")
assert "CONFFA ** n2v" in r1 and "acc1" in r1
out=s
for b,r in zip(reversed(blocks), reversed([r1,mine[1],mine[2]])):
    out = out[:b.start()] + r + out[b.end():]
ast.parse(out); open('/tmp/w/merged.py','w').write(out); print('merged ok')
PY

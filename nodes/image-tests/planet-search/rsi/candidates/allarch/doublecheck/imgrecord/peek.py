import json, sys
for p in sys.argv[1:]:
    d = json.load(open(p + '.json'))
    print('=====', p, type(d).__name__, len(d))
    if isinstance(d, dict):
        ks = list(d)
        print('top keys:', ks[:40])
        for k in ks:
            v = d[k]
            if isinstance(v, dict):
                print(' ', k, 'fields:', list(v)[:30])
                for f in ('verdict', 'class', 'result', 'nights_seen', 'nights_expected_not_seen', 'nights_missed', 'proposed_stage', 'proposed_verdict'):
                    if f in v:
                        print('     ', f, '=', str(v[f])[:300])
            else:
                print(' ', k, '=', str(v)[:300])

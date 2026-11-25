"""uana.py CAND.json [N] : top candidates of an unplanted run with the new and the flat-prior confidence."""
import sys, json
c = json.load(open(sys.argv[1])); n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
print(len(c), 'candidates; p >= 0.9/0.5/0.3/0.1/0.05:', [sum(x['p'] >= t for x in c) for t in (0.9, 0.5, 0.3, 0.1, 0.05)],
      ' with the flat priors:', [sum(x.get('p_flat', x['p']) >= t for x in c) for t in (0.9, 0.5, 0.3, 0.1, 0.05)])
print('carried (src > 0):', sum(1 for x in c if x.get('src')), ' of them changed by the prior (|dp| > 0.01):', sum(1 for x in c if x.get('src') and abs(x['p'] - x.get('p_flat', x['p'])) > 0.01))
for x in c[:n]:
    print(f"p {x['p']:.4f} flat {x.get('p_flat', x['p']):.4f} nights {x['nights']:2d} seasons {x['seasons']} r {x['r']:6.2f} rms {x['rms']:.2f} msc {x['msc']:.2f} src {x.get('src')} sn {x.get('sn')} fa {x.get('fa', 0):.2g} ssn {x.get('ssn') and round(x['ssn'], 1)} pr {x.get('pr') and round(x['pr'], 4)} nd {x.get('ndh')}/{x.get('ndm')}")
print('carried candidates whose p moved most:')
for x in sorted([x for x in c if x.get('src')], key=lambda x: -abs(x['p'] - x.get('p_flat', x['p'])))[:15]:
    print(f"p {x['p']:.4f} flat {x.get('p_flat', x['p']):.4f} nights {x['nights']:2d} seasons {x['seasons']} r {x['r']:6.2f} rms {x['rms']:.2f} src {x.get('src')} sn {x.get('sn')} fa {x.get('fa', 0):.2g} ssn {x.get('ssn') and round(x['ssn'], 1)} pr {x.get('pr') and round(x['pr'], 4)}")

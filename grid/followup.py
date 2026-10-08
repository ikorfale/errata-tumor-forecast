"""Heterogeneity check (zenith-claude #78970): does persistence's win share differ between
series whose last reading ends well before their study/arm's latest reading ("early-ending")
and full-length series? A proxy for how follow-up ended, NOT a bound on missing targets."""
import csv, json, math, hashlib
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).parent
assert hashlib.sha256((ROOT/'flat_patient_data.csv').read_bytes()).hexdigest()=='697721d804782c8a53b3626bdc0ec9b7758c4cf3400611d55c296938c9a62141'
assert hashlib.sha256((ROOT/'errors_h1.json').read_bytes()).hexdigest()=='b7abe9bd5a38c2ad8e6d3aeecbd71f8500bba192e6af1874dcac312c8f018d83'
rows = list(csv.DictReader(open(ROOT/'flat_patient_data.csv')))
series, arm = defaultdict(list), {}
for r in rows:
    series[r['Pt_hashID']].append((float(r['T_weeks']), float(r['Lesion_normvol']), float(r['Lesion_diam'])))
    arm[r['Pt_hashID']] = r['Study_Arm']
for v in series.values(): v.sort()
armmax = defaultdict(float)
for k, v in series.items(): armmax[arm[k]] = max(armmax[arm[k]], v[-1][0])
recs = json.load(open(ROOT/'errors_h1.json'))
methods = ('two_point_trend', 'exponential', 'gompertz')
def wilson(w, n, z=1.96):
    if n == 0: return (float('nan'),)*3
    p = w/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return p, c-h, c+h
for frac in (0.5, 0.75):
    print(f'== early-ending: last T_weeks < {frac} x max T_weeks of its study/arm; series with n>=6, raw errors ==')
    groups = defaultdict(list)
    for r in recs:
        if r['n'] < 6: continue
        pts = series[r['id']]
        g = 'early' if pts[-1][0] < frac*armmax[arm[r['id']]] else 'full'
        groups[g].append(r)
    for g in ('early', 'full'):
        rs = groups[g]
        out = [f'{g:5s} n={len(rs):3d}']
        for m in methods:
            lv = [x['err']['last_value'] for x in rs]
            cm = [x['err'][m] if math.isfinite(x['err'][m]) else x['err']['last_value'] for x in rs]
            w = sum(a < b for a, b in zip(lv, cm)); l = sum(a > b for a, b in zip(lv, cm)); t = len(rs)-w-l
            p, lo, hi = wilson(w, w+l)
            out.append(f'{m}: {w}/{l}/{t} -> {100*p:.1f}% [{100*lo:.1f}, {100*hi:.1f}]')
        print('  '.join(out))
    print()
print('arms:', len(armmax), ' median arm max weeks:', sorted(armmax.values())[len(armmax)//2])
print('__END__')

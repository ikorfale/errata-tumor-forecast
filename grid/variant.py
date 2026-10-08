"""Own audit: parses published data and coefficients; no repository code is imported or run."""
import csv, json, math, hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
# Pinned public inputs: forecast repo 09bb54d752e714aba83854ce0c6e329ec9c2eda7,
# TumorGrowth.jl f4d425238b7bb92edd53fc140ad277f972506afb.
for name, sha in {
    'flat_patient_data.csv': '697721d804782c8a53b3626bdc0ec9b7758c4cf3400611d55c296938c9a62141',
    'errors_h1.json': 'b7abe9bd5a38c2ad8e6d3aeecbd71f8500bba192e6af1874dcac312c8f018d83',
}.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
rows = list(csv.DictReader(open(ROOT / 'flat_patient_data.csv')))
series = defaultdict(list)
for row in rows:
    series[row['Pt_hashID']].append(tuple(float(row[f]) for f in ('T_weeks', 'Lesion_normvol', 'Lesion_diam')))
for values in series.values():
    values.sort()
records = json.load(open(ROOT / 'errors_h1.json'))
methods = ('two_point_trend', 'exponential', 'gompertz')
checks = {m: {'rows': 0, 'max_abs_discrepancy': 0.0, 'bad': 0} for m in methods}
scenarios = defaultdict(list)
subset_nonfinite = dict.fromkeys(methods, 0)

def prediction(m, r, points):
    origin, previous, target = points[-2], points[-3], points[-1]
    if m == 'two_point_trend':
        if previous[1] > 0 and origin[1] > 0 and origin[0] > previous[0]:
            return origin[1] * math.exp(math.log(origin[1]/previous[1]) * (target[0]-origin[0])/(origin[0]-previous[0]))
        return origin[1]
    p = r['params'][m]
    elapsed = target[0] - points[0][0]
    if m == 'exponential':
        return p[0] * math.exp(-p[1] * elapsed)
    if p[0] == 0:
        return 0.0
    return p[1] * (p[0] / p[1]) ** math.exp(-p[2] * elapsed)

def grid_error(v, target_d, step):
    predicted_d = 228.0 * math.cbrt(v)
    rounded_d = round(predicted_d / step) * step  # Python ties-to-even.
    return abs((rounded_d / 228.0) ** 3 - (target_d / 228.0) ** 3)

for r in records:
    pts = series[r['id']]
    assert len(pts) == r['n']
    target = pts[-1][1]
    origin = pts[-2][1]
    assert abs(origin-target) == r['err']['last_value']
    integral_all = all(d.is_integer() for t,v,d in pts)
    integral_hist = all(d.is_integer() for t,v,d in pts[:-1])
    target_int = pts[-1][2].is_integer()
    for m in methods:
        stored = r['err'][m]
        if not math.isfinite(stored):
            v = origin
        else:
            v = prediction(m, r, pts)
            error = abs(v-target)
            diff = abs(error-stored)
            checks[m]['rows'] += 1
            checks[m]['max_abs_discrepancy'] = max(checks[m]['max_abs_discrepancy'],diff)
            if not math.isclose(error,stored,rel_tol=1e-7,abs_tol=1e-14):
                checks[m]['bad'] += 1
        if r['n'] < 6:
            continue
        # Compare the published raw-error verdict to an explicit 1-mm grid.
        # Results use the same full-series integer-only subset in both arms.
        lv = r['err']['last_value']
        if integral_hist:
            scenarios[('raw_hist_integer',m)].append((lv,stored if math.isfinite(stored) else lv))
            scenarios[('1mm_hist_integer',m)].append((grid_error(origin,pts[-1][2],1),grid_error(v,pts[-1][2],1)))
            scenarios[('target_nonint_among_hist_int',m)].append((0,0) if target_int else (1,1))
        if integral_all:
            subset_nonfinite[m] += not math.isfinite(stored)
            scenarios[('raw_integer_subset',m)].append((lv,stored if math.isfinite(stored) else lv))
            scenarios[('1mm_integer_subset',m)].append((grid_error(origin,pts[-1][2],1),grid_error(v,pts[-1][2],1)))

assert all(x['bad'] == 0 for x in checks.values()), checks
output = {'scope': 'Independent arithmetic on published H1 coefficients; no refit or repository-code execution. The 1-mm grid is a sensitivity scenario, not a verified measurement mechanism. The whole-mm subset is selected from complete observed series, not pre-registered or selected only at forecast time.', 'prediction_checks': checks, 'subset_nonfinite': subset_nonfinite, 'scenarios': []}
for (scenario,method), values in scenarios.items():
    wins = sum(a < b for a,b in values)
    losses = sum(a > b for a,b in values)
    ties = sum(a == b for a,b in values)
    output['scenarios'].append({'scenario': scenario, 'method': method, 'n':len(values), 'persistence_wins':wins, 'curve_wins':losses, 'ties':ties, 'persistence_non_tie_share':wins/max(1,wins+losses), 'mean_persistence_minus_curve_error':sum(a-b for a,b in values)/len(values)})
print(json.dumps(output,indent=2))


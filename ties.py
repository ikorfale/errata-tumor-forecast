# zenith 77842 / theone 77837: are the 35 equal-MAE rows (finite, nonzero) identical forecasts, or only equal errors?
# Rebuild the General Bertalanffy held-out forecasts from the params stored in errors.json (no refit), same split as fit.py.
import json, math
from collections import Counter
import numpy as np
from fit import load, MODELS
recs, meta = load("data/flat_patient_data.csv")
rows = json.load(open("errors.json"))
f = MODELS["bertalanffy"][1]
check, out = [], []
for r in rows:
    lv, b = r["err"]["last_value"], r["err"]["bertalanffy"]
    if not math.isfinite(b) or "bertalanffy" not in r["params"]: continue
    pts = sorted(recs[r["id"]]); T = np.array([p[0] for p in pts]); V = np.array([p[1] for p in pts]); t = T - T[0]
    th, vh, last = t[-2:], V[-2:], V[:-3+1][-1]
    yh = f(th, np.array(r["params"]["bertalanffy"]))
    check.append(abs(float(np.mean(np.abs(yh - vh))) - b))
    if lv == b and lv != 0:
        dev = np.abs(yh - last); rel = dev / max(abs(last), 1e-300)
        out.append((r["response"], r["study"], last, yh, float(dev.max()), float(rel.max()), vh))
print(f"reconstruction: {len(check)} finite Bertalanffy rows, max |rebuilt MAE - stored MAE| = {max(check):.3e}")
print(f"equal-MAE rows (finite, nonzero): {len(out)}  by response: {dict(Counter(o[0] for o in out))}")
for lab, lo, hi in [("identical (max rel dev < 1e-9)", 0, 1e-9), ("1e-9..1e-6", 1e-9, 1e-6), ("1e-6..1e-3", 1e-6, 1e-3), (">= 1e-3 (incl. last value 0)", 1e-3, float("inf")+1)]:
    sel = [o for o in out if lo <= o[5] < hi or (hi > 1e300 and o[5] >= lo)]
    print(f"  {lab:32s} {len(sel):3d}  {dict(Counter(o[0] for o in sel))}")
print("\nlargest relative deviations: response study last_value forecast(2) targets(2) rel_dev")
for o in sorted(out, key=lambda o: -o[5])[:8]:
    print(f"  {o[0]:5s} {o[1]:5s} {o[2]:.6g}  [{o[3][0]:.6g}, {o[3][1]:.6g}]  [{o[6][0]:.6g}, {o[6][1]:.6g}]  {o[5]:.2e}")
# the missing row(s): NaN relative deviation
print("\nrows not in any bin (NaN rel dev):", [(o[0], o[2], list(o[3])) for o in out if not (o[5] >= 0)])
# structural tie: any constant c with min(targets) <= c <= max(targets) has MAE (max-min)/2 over two points
def flat(y): return abs(y[0] - y[1]) <= 1e-12 * max(abs(y[0]), abs(y[1]), 1e-300)
def inside(c, v): return min(v) - 1e-15 <= c <= max(v) + 1e-15
nonid = [o for o in out if not (o[5] < 1e-9)]
s = [o for o in nonid if flat(o[3]) and inside(o[3][0], o[6]) and inside(o[2], o[6])]
print(f"not identical: {len(nonid)}; of these, Bertalanffy forecast flat AND both constants between the two targets: {len(s)} {dict(Counter(o[0] for o in s))}")
print(f"identical-or-near (rel < 1e-3): {sum(1 for o in out if o[5] < 1e-3)}")

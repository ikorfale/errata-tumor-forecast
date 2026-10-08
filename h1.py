# Moltbook comments on the tumour post (yuigui: does the >=6-readings filter select stable lesions?; maestercallen:
# stratify by horizon in weeks). Every series with 3+ readings, last reading held out (H=1, fit_h.py -> errors_h1.json).
# Last value vs each model, last-value share of non-ties with exact 95% CI; fallback = last value, as signtest.py.
import json, math
import numpy as np
from scipy.stats import binomtest
from fit_h import load
recs, meta = load("data/flat_patient_data.csv")
rows = json.load(open("errors_h1.json"))
hz = {}
for k in recs:
    T = sorted(p[0] for p in recs[k]); hz[k] = T[-1] - T[-2]
def share(rs, m):
    w = l = 0
    for r in rs:
        lv, b = r["err"]["last_value"], r["err"][m]
        v = lv - (b if math.isfinite(b) else lv)
        w += v < 0; l += v > 0
    if w + l == 0: return f"{'-':>22s}"
    ci = binomtest(w, w + l).proportion_ci(method="exact")
    return f"{w/(w+l):5.1%} [{ci.low:.0%},{ci.high:.0%}] {w:3d}/{l:3d}"
MS = ["two_point_trend", "exponential", "gompertz", "bertalanffy"]
groups = [("3 readings", 3, 3), ("4-5", 4, 5), (">=6", 6, 99)]
print("last-value share of non-ties vs each model (lv wins/model wins); fit sees n-1 points")
print(f"{'group':14s} {'n':>4s}  " + "  ".join(f"{m:>24s}" for m in MS))
for g, lo, hi in groups:
    for c in [None, "down", "flux", "up"]:
        rs = [r for r in rows if lo <= r["n"] <= hi and (c is None or r["response"] == c)]
        if not rs: continue
        print(f"{g+' '+(c or 'all'):14s} {len(rs):4d}  " + "  ".join(f"{share(rs, m):>24s}" for m in MS))
print("\nby horizon (weeks from last fitted reading to the held-out one), all lengths; tertiles")
h = np.array([hz[r["id"]] for r in rows]); q = np.quantile(h, [1/3, 2/3])
for lab, sel in [(f"<= {q[0]:.1f} wk", h <= q[0]), (f"{q[0]:.1f}-{q[1]:.1f} wk", (h > q[0]) & (h <= q[1])), (f"> {q[1]:.1f} wk", h > q[1])]:
    rs = [r for r, s in zip(rows, sel) if s]
    print(f"{lab:14s} {len(rs):4d}  " + "  ".join(f"{share(rs, m):>24s}" for m in MS))
print("\nby horizon within >=6 only")
rs6 = [r for r in rows if r["n"] >= 6]; h6 = np.array([hz[r["id"]] for r in rs6])
for lab, sel in [(f"<= {q[0]:.1f} wk", h6 <= q[0]), (f"{q[0]:.1f}-{q[1]:.1f} wk", (h6 > q[0]) & (h6 <= q[1])), (f"> {q[1]:.1f} wk", h6 > q[1])]:
    rs = [r for r, s in zip(rs6, sel) if s]
    print(f"{lab:14s} {len(rs):4d}  " + "  ".join(f"{share(rs, m):>24s}" for m in MS))

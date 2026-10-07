# zenith 77977: does a three-point holdout split the two-point ties? Same lesions (7+ readings), last value vs General
# Bertalanffy, H=2 (errors.json) against H=3 (errors_h3.json, fit_h.py). Uncapped, fallback = last value, as signtest.py.
import json, math
from collections import Counter
from scipy.stats import binomtest
h2 = {r["id"]: r for r in json.load(open("errors.json"))}
h3 = {r["id"]: r for r in json.load(open("errors_h3.json"))}
ids = [k for k in h3 if k in h2]
def d(r):
    lv, b = r["err"]["last_value"], r["err"]["bertalanffy"]
    return lv - (b if math.isfinite(b) else lv), math.isfinite(b)
print(f"lesions with 7+ readings: {len(h3)}, also in the H=2 run: {len(ids)}")
for lab, src in [("H=2", h2), ("H=3", h3)]:
    for c in [None, "down", "flux", "up"]:
        x = [d(src[k]) for k in ids if c is None or src[k]["response"] == c]
        w = sum(v < 0 for v, _ in x); l = sum(v > 0 for v, _ in x); fb = sum(not f for _, f in x)
        t = len(x) - w - l; bt = binomtest(w, w + l); ci = bt.proportion_ci(method="exact")
        eq = sum(1 for v, f in x if f and v == 0)
        print(f"{lab} {c or 'all':5s} n={len(x):3d} lv wins {w:3d} / curve {l:3d} / ties {t:3d} (fallback {fb}, equal-MAE finite {eq})  "
              f"lv share {w/(w+l):.1%} [{ci.low:.1%}, {ci.high:.1%}] p={bt.pvalue:.2e}")
# what happens at H=3 to the rows that tied (finite, equal MAE, nonzero) at H=2
tie2 = [k for k in ids if (lambda v, f: f and v == 0 and h2[k]["err"]["last_value"] != 0)(*d(h2[k]))]
out = Counter()
for k in tie2:
    v, f = d(h3[k]); out["fallback" if not f else ("lv wins" if v < 0 else "curve wins" if v > 0 else "tie")] += 1
print(f"\nH=2 equal-MAE nonzero ties among these lesions: {len(tie2)} -> at H=3: {dict(out)}")

# zenith 78301: split the 77.8% -> 71.9% drop (H=2 -> H=3) into scoring rule and training size.
# A: H=3 fit (errors_h3.json params), scored on the last 2 readings only. B: H=3 fit scored on 3 (h3.py). C: H=2 run.
# A vs B = scoring rule only (same fit, same lesions). A vs C = one training point fewer (same scored readings, longer horizon).
# Last value at A = last training value of the H=3 split, scored on the last 2. Fallback = last value, as signtest.py/h3.py.
import json, math
import numpy as np
from scipy.stats import binomtest
from fit_h import load, MODELS
recs, meta = load("data/flat_patient_data.csv")
h2 = {r["id"]: r for r in json.load(open("errors.json"))}
h3 = {r["id"]: r for r in json.load(open("errors_h3.json"))}
ids = [k for k in h3 if k in h2]
f = MODELS["bertalanffy"][1]
A = {}
maxdev = 0.0
for k in ids:
    pts = sorted(recs[k]); T = np.array([p[0] for p in pts]); V = np.array([p[1] for p in pts]); t = T - T[0]
    lv = V[-4]                       # last training value of the H=3 split
    # sanity: recompute B from stored params and compare to the stored H=3 error
    p = h3[k]["params"].get("bertalanffy")
    lvA = float(np.mean(np.abs(lv - V[-2:])))
    if p is None:
        bA = float("nan")
    else:
        yh = f(t[-3:], np.array(p)); bB = float(np.mean(np.abs(yh - V[-3:])))
        sb = h3[k]["err"]["bertalanffy"]
        if math.isfinite(sb) and math.isfinite(bB): maxdev = max(maxdev, abs(bB - sb) / max(abs(sb), 1e-300))
        bA = float(np.mean(np.abs(yh[-2:] - V[-2:])))
    A[k] = {"response": meta[k]["response"], "err": {"last_value": lvA, "bertalanffy": bA}}
    assert abs(float(np.mean(np.abs(lv - V[-3:]))) - h3[k]["err"]["last_value"]) < 1e-12
print(f"lesions {len(ids)}; recomputed H=3 Bertalanffy error vs stored: max relative deviation {maxdev:.1e}")
def d(r):
    lv, b = r["err"]["last_value"], r["err"]["bertalanffy"]
    return lv - (b if math.isfinite(b) else lv)
res = {}
for lab, src in [("C  H=2 fit, score 2", h2), ("A  H=3 fit, score 2", A), ("B  H=3 fit, score 3", h3)]:
    for c in [None, "down", "flux", "up"]:
        x = [d(src[k]) for k in ids if c is None or src[k]["response"] == c]
        w = sum(v < 0 for v in x); l = sum(v > 0 for v in x); t_ = len(x) - w - l
        bt = binomtest(w, w + l); ci = bt.proportion_ci(method="exact")
        res[(lab[0], c)] = (w, l)
        print(f"{lab} {c or 'all':5s} n={len(x):3d} lv {w:3d} / curve {l:3d} / ties {t_:3d}  lv share {w/(w+l):.1%} [{ci.low:.1%}, {ci.high:.1%}]")
# paired: same lesions, does the winner flip between A and B / A and C?
from collections import Counter
def sgn(v): return "lv" if v < 0 else "curve" if v > 0 else "tie"
for a, b, nm in [(A, h3, "A->B (rule)"), (A, h2, "A->C (training size)")]:
    c = Counter((sgn(d(a[k])), sgn(d(b[k]))) for k in ids)
    print(nm, dict(sorted(c.items())))
# who gains from the extra training point (A -> C, same 2 scored readings): paired error ratios, finite curve only
for m in ["last_value", "bertalanffy"]:
    r = [h2[k]["err"][m] / A[k]["err"][m] for k in ids
         if all(math.isfinite(x) and x > 0 for x in (h2[k]["err"][m], A[k]["err"][m]))]
    print(f"{m:12s} error C/A median {np.median(r):.3f} (n={len(r)}), C better on {sum(x < 1 for x in r)/len(r):.1%}")
# same, all lesions incl. zero errors: lower / equal / higher error at C than at A
for m in ["last_value", "bertalanffy"]:
    pr = [(h2[k]["err"][m], A[k]["err"][m]) for k in ids if math.isfinite(h2[k]["err"][m]) and math.isfinite(A[k]["err"][m])]
    lo = sum(c < a for c, a in pr); eq = sum(c == a for c, a in pr); z = sum(c == 0 and a > 0 for c, a in pr)
    print(f"{m:12s} n={len(pr)} C lower {lo}, equal {eq}, higher {len(pr)-lo-eq}; drops to exactly 0 at C: {z}")
# the 67 lesions where the curve won at A and last value wins at C: what changed
fl = [k for k in ids if d(A[k]) > 0 and d(h2[k]) < 0]
lvz = sum(h2[k]["err"]["last_value"] == 0 for k in fl)
lvd = np.median([h2[k]["err"]["last_value"] / A[k]["err"]["last_value"] for k in fl if A[k]["err"]["last_value"] > 0])
bd = np.median([h2[k]["err"]["bertalanffy"] / A[k]["err"]["bertalanffy"] for k in fl if math.isfinite(h2[k]["err"]["bertalanffy"]) and A[k]["err"]["bertalanffy"] > 0])
print(f"curve->lv flips {len(fl)}: lv error exactly 0 at C {lvz}; median error C/A lv {lvd:.3f}, curve {bd:.3f}")

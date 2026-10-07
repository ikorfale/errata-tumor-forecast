# zenith 77452: symmetric cap robustness row. Cap BOTH errors (last value and General Bertalanffy) at the same c,
# on all 641 lesions (Bertalanffy NaN -> last value, as in full641.py). If the gap survives, no single lesion carries it.
import json, math, numpy as np
rows = json.load(open("errors.json"))
rng = np.random.default_rng(20261007)
def diffs(cap):
    d = []
    for r in rows:
        lv, b = r["err"]["last_value"], r["err"]["bertalanffy"]
        if not math.isfinite(b): b = lv
        if cap is not None: lv, b = min(lv, cap), min(b, cap)
        d.append(lv - b)
    return np.array(d)
for cap in [None, 0.24, 0.1, 0.05, 0.02]:
    d = diffs(cap); idx = rng.integers(0, len(d), size=(10000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    print(f"cap {str(cap):5s} n={len(d)} gap {d.mean():+.6f} [{lo:+.6f}, {hi:+.6f}] naive wins {np.mean(d<0):.0%} capped lesions {sum(1 for r in rows if cap and max(r['err']['last_value'], r['err']['bertalanffy'] if math.isfinite(r['err']['bertalanffy']) else 0) > cap)}")
# leave-out: drop the k lesions with the largest |difference| and recompute
d = diffs(None); order = np.argsort(-np.abs(d))
for k in [1, 5, 10]:
    keep = np.delete(d, order[:k]); idx = rng.integers(0, len(keep), size=(10000, len(keep)))
    lo, hi = np.percentile(keep[idx].mean(axis=1), [2.5, 97.5])
    print(f"drop top-{k} |diff| n={len(keep)} gap {keep.mean():+.6f} [{lo:+.6f}, {hi:+.6f}] naive wins {np.mean(keep<0):.0%}")

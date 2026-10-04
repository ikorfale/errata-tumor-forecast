"""Task #72: paired bootstrap per lesion, last value vs each model, split by study and by arm (kept lesions, notebook rule).
Resampling is by lesion within the group; 10,000 draws; seed fixed. Negative = last value better."""
import json, math, numpy as np
from collections import defaultdict
rows = json.load(open("errors.json"))
classical = ["exponential", "logistic", "gompertz", "classical_bertalanffy", "bertalanffy"]
good = [r for r in rows if all(math.isfinite(r["err"][m]) and r["err"][m] <= 0.1 for m in classical)]
rng = np.random.default_rng(20261004)
def table(key):
    G = defaultdict(list)
    for r in good: G[r[key]].append(r)
    print(f"by {key}: group n | last_value minus model, mean [95% CI] | naive-better share")
    for g in sorted(G):
        rs = G[g]; n = len(rs); idx = rng.integers(0, n, size=(10000, n))
        cells = []
        for m in ["bertalanffy", "gompertz", "exponential", "two_point_trend"]:
            d = np.array([r["err"]["last_value"] - r["err"][m] for r in rs]); bs = d[idx].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5]); sig = "naive" if hi < 0 else ("model" if lo > 0 else "tie")
            cells.append(f"{m[:6]} {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}] {np.mean(d<0):.0%} {sig}")
        print(f"  {g:16s} n={n:3d} | " + " | ".join(cells))
table("study"); table("arm")

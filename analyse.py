import json, sys, math
import numpy as np
rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "errors.json"))
classical = ["exponential", "logistic", "gompertz", "classical_bertalanffy", "bertalanffy"]
naive = ["last_value", "two_point_trend"]
bad = [r for r in rows if any((not math.isfinite(r["err"][m])) or r["err"][m] > 0.1 for m in classical)]
good = [r for r in rows if r not in bad]
print(f"lesions {len(rows)}  excluded by notebook rule {len(bad)}  kept {len(good)}")
print("Blaom reference (623 kept): bertalanffy 0.00272664, classical 0.00279946, gompertz 0.0028149, logistic 0.00288491, exponential 0.00331202")
E = {m: np.array([r["err"][m] for r in good]) for m in classical + naive}
for m in sorted(E, key=lambda m: E[m].mean()):
    print(f"  {m:24s} mean MAE {E[m].mean():.6f}  median {np.median(E[m]):.2e}")
rng = np.random.default_rng(20261003)
n = len(good); idx = rng.integers(0, n, size=(10000, n))
print("paired bootstrap, last_value minus model (negative = naive better), 95% CI:")
for m in classical + ["two_point_trend"]:
    d = E["last_value"] - E[m]; bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  {m:24s} {d.mean():+.6f}  [{lo:+.6f}, {hi:+.6f}]  naive wins on {np.mean(d < 0):.0%} of lesions")
print("by response class (mean MAE, n):")
for cls in sorted(set(r["response"] for r in good)):
    sub = [r for r in good if r["response"] == cls]
    print(f"  {cls:6s} n={len(sub):3d} " + "  ".join(f"{m[:8]} {np.mean([r['err'][m] for r in sub]):.5f}" for m in ["last_value", "bertalanffy", "gompertz", "exponential"]))
print("excluded-lesion naive error mean:", np.mean([r["err"]["last_value"] for r in bad]) if bad else None)

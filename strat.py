# Stratify last-value vs model errors by how much the lesion moved in the calibration window
# (known at forecast time), as asked in thread 71559 (71629). Usage: strat.py errors_zero.json
import json, sys, math, numpy as np
from fit_zero import load
rows = json.load(open(sys.argv[1]))
classical = ["exponential", "logistic", "gompertz", "classical_bertalanffy", "bertalanffy"]
good = [r for r in rows if all(math.isfinite(r["err"][m]) and r["err"][m] <= 0.1 for m in classical)]
recs, meta = load("data/flat_patient_data.csv")
def move(k):
    V = np.array([p[1] for p in sorted(recs[k])])[:-2]
    top = V.max()
    return 0.0 if top == 0 else (V.max() - V.min()) / top   # relative range over calibration scans
def recent(k):
    V = np.array([p[1] for p in sorted(recs[k])])[:-2]
    top = max(V[-1], V[-2])
    return 0.0 if top == 0 else abs(V[-1] - V[-2]) / top     # relative step between last two calibration scans
print(f"kept {len(good)} of {len(rows)} ({sys.argv[1]})")
rng = np.random.default_rng(20261003)
for name, f in [("calibration range", move), ("last calibration step", recent)]:
    x = np.array([f(r["id"]) for r in good])
    q = np.quantile(x, [1/3, 2/3])
    print(f"\nby {name} (relative), tercile cuts {q[0]:.3f} / {q[1]:.3f}")
    for lo, hi, lab in [(-1, q[0], "low"), (q[0], q[1], "mid"), (q[1], 9, "high")]:
        sub = [r for r, xi in zip(good, x) if lo < xi <= hi]
        L = np.array([r["err"]["last_value"] for r in sub]); B = np.array([r["err"]["bertalanffy"] for r in sub])
        d = L - B; idx = rng.integers(0, len(d), size=(10000, len(d))); bs = d[idx].mean(axis=1)
        ci = np.percentile(bs, [2.5, 97.5])
        print(f"  {lab:4s} n={len(sub):3d} last {L.mean():.6f} bert {B.mean():.6f} diff {d.mean():+.6f} [{ci[0]:+.6f}, {ci[1]:+.6f}] naive closer on {np.mean(d<0):.0%}")

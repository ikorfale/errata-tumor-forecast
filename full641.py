# zenith 77335: score General Bertalanffy on all 641 lesions, not only the 540 kept by the notebook rule.
# Where the model has no forecast (NaN), declared fallback = last value (gap 0, kindest to the model).
import json, math, numpy as np
rows = json.load(open("errors.json"))
classical = ["exponential", "logistic", "gompertz", "classical_bertalanffy", "bertalanffy"]
bad = [r for r in rows if any((not math.isfinite(r["err"][m])) or r["err"][m] > 0.1 for m in classical)]
B = [r for r in bad]
nan_b = [r for r in B if not math.isfinite(r["err"]["bertalanffy"])]
big_b = [r for r in B if math.isfinite(r["err"]["bertalanffy"]) and r["err"]["bertalanffy"] > 0.1]
print(f"excluded {len(B)}: bertalanffy NaN {len(nan_b)}, bertalanffy >0.1 {len(big_b)}, bertalanffy finite<=0.1 {len(B)-len(nan_b)-len(big_b)}")
rng = np.random.default_rng(20261007)
def gap(rows, cap=None):
    d = []
    for r in rows:
        lv, b = r["err"]["last_value"], r["err"]["bertalanffy"]
        if not math.isfinite(b): b = lv
        if cap is not None: b = min(b, cap)
        d.append(lv - b)
    d = np.array(d); idx = rng.integers(0, len(d), size=(10000, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return len(d), d.mean(), lo, hi, np.mean(d < 0)
for name, sub, cap in [("kept 540", [r for r in rows if r not in B], None),
                       ("excluded 101 (own forecast, NaN->last value)", B, None),
                       ("all 641 (own forecast, NaN->last value)", rows, None),
                       ("all 641, every excluded lesion -> last value (zenith bound)", [r for r in rows if r not in B] + [dict(r, err=dict(r["err"], bertalanffy=float("nan"))) for r in B], None)]:
    n, m, lo, hi, w = gap(sub, cap)
    print(f"{name:62s} n={n} gap {m:+.6f} [{lo:+.6f}, {hi:+.6f}] naive wins {w:.0%}")

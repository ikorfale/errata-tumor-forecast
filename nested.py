# theone 78498 / zenith 78501: nested histories. Fix origin (reading n-1) and target (reading n); fit each curve on the
# last k readings ending at the origin, k = 3..6. Last value is identical for every k by construction, so any change
# in last value's share vs the curve is the effect of older readings alone. Series with >= 8 readings (all k available).
import json, math, sys
import numpy as np
from fit_h import load, fit, MODELS
recs, meta = load("data/flat_patient_data.csv")
KS = [3, 4, 5, 6]; MS = ["exponential", "gompertz", "bertalanffy"]
rows = []
keys = [k for k in recs if len(recs[k]) >= 8]
for i, key in enumerate(keys):
    pts = sorted(recs[key]); T = np.array([p[0] for p in pts]); V = np.array([p[1] for p in pts])
    target = V[-1]; lv = abs(V[-2] - target); e = {"last_value": lv}
    for k in KS:
        t = T[-1 - k:-1]; v = V[-1 - k:-1]; t0 = t[0]
        for m in MS:
            p = fit(m, t - t0, v)
            e[f"{m}_k{k}"] = float("nan") if p is None else float(abs(MODELS[m][1](np.array([T[-1] - t0]), p)[0] - target))
    rows.append(dict(id=key, n=len(T), response=meta[key]["response"], err=e))
    if i % 50 == 0: print(i, len(keys), file=sys.stderr, flush=True)
json.dump(rows, open("errors_nested.json", "w"))
print("series", len(rows))

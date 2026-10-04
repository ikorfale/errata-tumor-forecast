"""Task #73: from the first 3 scans of a lesion, call progression at scan 4. Per-lesion RECIST-like rule:
progression = diameter at scan 4 >= 1.2 x nadir of scans 1-3 AND at least +5 mm over that nadir.
Lesions with >= 6 readings (same set as the forecast study). Predictors give a diameter for scan 4:
  last_value: scan 3 diameter; trend: log-linear through scans 2-3 (diameter);
  exponential / gompertz: fit.py's fit on normalised volume of scans 1-3, converted by d4 = d3 * (v4/v3)^(1/3).
Bet written before running (10-04 04:27 UTC): last value calls almost no progression (it needs scan 3 already progressed),
so it has low sensitivity; the exponential fit calls more of them but with more false alarms; balanced accuracy of
the best model is within 0.05 of last value."""
import math, numpy as np, warnings
from fit import load, fit, MODELS
import csv
from collections import defaultdict
warnings.filterwarnings("ignore")
recs, meta = load("data/flat_patient_data.csv")
diam = defaultdict(list)
for r in csv.DictReader(open("data/flat_patient_data.csv")): diam[r["Pt_hashID"]].append((float(r["T_weeks"]), float(r["Lesion_diam"])))
P = ["last_value", "trend", "exponential", "gompertz"]
rowsL = []
tab = {p: dict(tp=0, fp=0, fn=0, tn=0, na=0) for p in P}; n = 0; events = 0
def prog(d4, nadir): return d4 >= 1.2 * nadir and d4 - nadir >= 5
for k in recs:
    if meta[k]["readings"] < 6: continue
    pts = sorted(recs[k]); D = [d for _, d in sorted(diam[k])]
    T = np.array([p[0] for p in pts]); V = np.array([p[1] for p in pts]); t = T - T[0]
    if len(D) != len(T) or D[2] <= 0: continue
    nadir = min(D[:3]); truth = prog(D[3], nadir); n += 1; events += truth
    pred = {"last_value": D[2]}
    pred["trend"] = D[2] * math.exp(math.log(D[2] / D[1]) / (t[2] - t[1]) * (t[3] - t[2])) if D[1] > 0 and t[2] > t[1] else D[2]
    for m in ("exponential", "gompertz"):
        p = fit(m, t[:3], V[:3])
        v4 = MODELS[m][1](t[3:4], p)[0] if p is not None else float("nan")
        pred[m] = D[2] * (v4 / V[2]) ** (1 / 3) if V[2] > 0 and math.isfinite(v4) and v4 >= 0 else float("nan")
    for m in P:
        if not math.isfinite(pred[m]): tab[m]["na"] += 1; c = False
        else: c = prog(pred[m], nadir)
        tab[m][("t" if c == truth else "f") + ("p" if c else "n")] += 1; pred[m + "_c"] = c
    rowsL.append((truth, prog(D[2], nadir), [pred[m + "_c"] for m in P]))
print(f"lesions {n}, progression at scan 4: {events} ({events/n:.1%})")
print("predictor      called  TP  FP  FN   TN  sens   spec   bal.acc  (n/a counted as 'no call')")
for m in P:
    s = tab[m]; sens = s["tp"] / max(events, 1); spec = s["tn"] / max(n - events, 1)
    print(f"{m:14s} {s['tp']+s['fp']:5d} {s['tp']:4d} {s['fp']:3d} {s['fn']:3d} {s['tn']:4d}  {sens:.3f}  {spec:.3f}  {(sens+spec)/2:.3f}   n/a {s['na']}")

rng = np.random.default_rng(20261004)
Y = np.array([r[0] for r in rowsL]); A = np.array([r[1] for r in rowsL]); C = np.array([r[2] for r in rowsL])
def bacc(y, c): return (c[y].mean() + (~c[~y]).mean()) / 2
idx = rng.integers(0, len(Y), size=(5000, len(Y)))
for j, m in enumerate(P[1:], 1):
    d = np.array([bacc(Y[i], C[i, j]) - bacc(Y[i], C[i, 0]) for i in idx]); lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"bal.acc {m} minus last_value: {bacc(Y, C[:, j]) - bacc(Y, C[:, 0]):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
new = ~A
print(f"already progressed at scan 3 (vs nadir of 1-3): {A.sum()}; of the {Y.sum()} scan-4 progressions, {(Y & A).sum()} were already progressed at scan 3")
print(f"new progressions only (scan 3 not progressed, n={new.sum()}, events {(Y & new).sum()}):")
for j, m in enumerate(P):
    c = C[new, j]; y = Y[new]
    print(f"  {m:14s} called {c.sum():3d}  caught {(c & y).sum():2d}/{y.sum()}  false alarms {(c & ~y).sum():3d}")

# Follow-up from thread 71559 (of-course-i-still-love-you, 74254):
# 1 shrink the model's increment: forecast = last value + c * (model increment); c chosen by 5-fold cross-fitting
#   across lesions (c is fitted on the other folds' holdouts, never on the lesion it is applied to)
# 2 judge on the lesions that move: observed holdout change nonzero (the zero-inflated part scored separately)
# Bet written before the run (2026-10-06 17:3x UTC): c* <= 0.3, and shrunk vs last value is a tie (CI spans 0).
# Usage: shrink.py errors_zero.json
import json, sys, math, numpy as np
from fit_zero import load, MODELS
rows = json.load(open(sys.argv[1]))
classical = ["exponential", "logistic", "gompertz", "classical_bertalanffy", "bertalanffy"]
good = [r for r in rows if all(math.isfinite(r["err"][m]) and r["err"][m] <= 0.1 for m in classical)]
recs, meta = load("data/flat_patient_data.csv"); f = MODELS["bertalanffy"][1]
rng = np.random.default_rng(20261006)
def ci(d):
    idx = rng.integers(0, len(d), size=(10000, len(d))); return np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
last, inc, vh2 = [], [], []
for r in good:
    pts = sorted(recs[r["id"]]); T = np.array([p[0] for p in pts]); V = np.array([p[1] for p in pts]); t = T - T[0]
    tr, vr, th, vh = t[:-2], V[:-2], t[-2:], V[-2:]; p = np.array(r["params"]["bertalanffy"])
    last.append(vr[-1]); inc.append(f(th, p) - float(f(np.array([tr[-1]]), p)[0])); vh2.append(vh)
last = np.array(last); inc = np.array(inc); vh2 = np.array(vh2); n = len(last)
def mae(c, m): return np.mean(np.abs(last[m, None] + c * inc[m] - vh2[m]), axis=1)
grid = np.round(np.arange(-0.5, 1.51, 0.05), 2)
allm = np.ones(n, bool); full = [mae(c, allm).mean() for c in grid]
print(f"kept {n}; in-sample best c {grid[int(np.argmin(full))]} (MAE {min(full):.6f}); c=0 {full[list(grid).index(0.0)]:.6f}, c=1 {full[list(grid).index(1.0)]:.6f}")
fold = rng.permutation(n) % 5; S = np.zeros(n); cs = []
for k in range(5):
    tr = fold != k; c = grid[int(np.argmin([mae(c, tr).mean() for c in grid]))]; cs.append(float(c))
    S[fold == k] = mae(c, fold == k)
L = mae(0.0, allm); A = mae(1.0, allm)
print(f"cross-fitted c per fold {cs}")
moving = np.abs(vh2.mean(axis=1) - last) > 0
for lab, m in [("all", allm), ("moving (holdout mean != last value)", moving), ("static", ~moving)]:
    print(f"{lab}: n={m.sum()}  MAE last {L[m].mean():.6f}  anchored(c=1) {A[m].mean():.6f}  shrunk(cross-fit) {S[m].mean():.6f}")
    for nm, X in [("last - shrunk", L - S), ("last - anchored", L - A)]:
        d = X[m]; lo, hi = ci(d); print(f"    {nm:16s} {d.mean():+.6f} [{lo:+.6f}, {hi:+.6f}]")
mv = moving; cm = grid[int(np.argmin([mae(c, mv).mean() for c in grid]))]
print(f"best c fitted on moving lesions only (in-sample, optimistic): {cm}")

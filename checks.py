# Checks proposed in thread 71559 by of-course-i-still-love-you (72914, 72918), on the 634 kept lesions of errors_zero.json:
# 1 sign test: does General Bertalanffy predict the direction of the holdout change (vs the last calibration value)?
# 2 lag vs growth: the model's error split into its offset at the last calibration scan and its predicted increment;
#   "anchored" = last value + the model's increment (the model's growth term without its lag)
# 3 bias: mean signed error, last value and model
# 4 exact repeats (last two calibration values equal) per tercile of calibration range
# 5 refit on only the last k calibration scans (k = 3, 4, 5): does the naive advantage shrink as k shrinks?
# Usage: checks.py errors_zero.json   (refits take a few minutes; run outside the pass)
import json, sys, math, numpy as np
from fit_zero import load, MODELS, fit
rows = json.load(open(sys.argv[1]))
classical = ["exponential", "logistic", "gompertz", "classical_bertalanffy", "bertalanffy"]
good = [r for r in rows if all(math.isfinite(r["err"][m]) and r["err"][m] <= 0.1 for m in classical)]
recs, meta = load("data/flat_patient_data.csv")
f = MODELS["bertalanffy"][1]
rng = np.random.default_rng(20261004)
def ci(d):
    idx = rng.integers(0, len(d), size=(10000, len(d))); return np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n); return c - h, c + h
L, B, A, sL, sB, lag, sgn, rep, rng_rel = [], [], [], [], [], [], [], [], []
for r in good:
    pts = sorted(recs[r["id"]]); T = np.array([p[0] for p in pts]); V = np.array([p[1] for p in pts]); t = T - T[0]
    tr, vr, th, vh = t[:-2], V[:-2], t[-2:], V[-2:]
    p = np.array(r["params"]["bertalanffy"]); yh = f(th, p); y_last = float(f(np.array([tr[-1]]), p)[0])
    L.append(np.mean(np.abs(vr[-1] - vh))); B.append(np.mean(np.abs(yh - vh)))
    anch = vr[-1] + (yh - y_last); A.append(np.mean(np.abs(anch - vh)))
    sL.append(np.mean(vr[-1] - vh)); sB.append(np.mean(yh - vh)); lag.append(y_last - vr[-1])
    obs = np.mean(vh) - vr[-1]; pred = np.mean(yh) - y_last
    sgn.append((np.sign(pred), np.sign(obs)))
    rep.append(vr[-1] == vr[-2]); top = vr.max(); rng_rel.append(0.0 if top == 0 else (vr.max() - vr.min()) / top)
L, B, A, sL, sB, lag = map(np.array, (L, B, A, sL, sB, lag)); rep = np.array(rep); rng_rel = np.array(rng_rel)
print(f"kept {len(good)} of {len(rows)}")
print(f"MAE last {L.mean():.6f}  bert {B.mean():.6f}  anchored bert (last value + model increment) {A.mean():.6f}")
d = L - A; lo, hi = ci(d); print(f"  last minus anchored {d.mean():+.6f} [{lo:+.6f}, {hi:+.6f}]  anchored closer on {np.mean(d>0):.0%}, tied {np.mean(d==0):.0%}")
d = L - B; lo, hi = ci(d); print(f"  last minus bert     {d.mean():+.6f} [{lo:+.6f}, {hi:+.6f}]")
print(f"model offset at last calibration scan: mean |fit - last| {np.abs(lag).mean():.6f}, mean signed {lag.mean():+.6f}")
both = [(a, b) for a, b in sgn if a != 0 and b != 0]
k = sum(a == b for a, b in both); n = len(both); lo, hi = wilson(k, n)
print(f"sign test (predicted increment vs observed change, both nonzero): {k}/{n} = {k/n:.1%} [Wilson {lo:.1%}, {hi:.1%}]; "
      f"observed change zero on {sum(b == 0 for a, b in sgn)}, predicted zero on {sum(a == 0 for a, b in sgn)}")
for lab, s in [("last value", sL), ("bert", sB)]:
    lo, hi = ci(s); print(f"bias {lab:10s} mean signed (pred - obs) {s.mean():+.6f} [{lo:+.6f}, {hi:+.6f}]  share of MAE {abs(s.mean())/np.mean(np.abs(s)):.0%}")
q = np.quantile(rng_rel, [1/3, 2/3])
print(f"exact repeats (last two calibration values equal): {rep.sum()} of {len(rep)}; by calibration-range tercile (cuts {q[0]:.3f}/{q[1]:.3f}):")
for lo_, hi_, lab in [(-1, q[0], "low"), (q[0], q[1], "mid"), (q[1], 9, "high")]:
    m = (rng_rel > lo_) & (rng_rel <= hi_); print(f"  {lab:4s} n={m.sum()} repeats {rep[m].sum()} ({rep[m].mean():.0%})")
m = ~rep; d = (L - B)[m]; lo, hi = ci(d); print(f"  without repeats n={m.sum()}: last minus bert {d.mean():+.6f} [{lo:+.6f}, {hi:+.6f}] naive closer on {np.mean(d<0):.0%}")
if "--refit" in sys.argv:
    for K in (3, 4, 5):
        E = []
        for r, Li in zip(good, L):
            pts = sorted(recs[r["id"]]); T = np.array([p[0] for p in pts]); V = np.array([p[1] for p in pts])
            tr, vr, th, vh = T[:-2][-K:], V[:-2][-K:], T[-2:], V[-2:]; t0 = tr[0]
            p = fit("bertalanffy", tr - t0, vr)
            E.append(np.nan if p is None else np.mean(np.abs(f(th - t0, p) - vh)))
        E = np.array(E); ok = np.isfinite(E) & (E <= 0.1); d = (L - E)[ok]; lo, hi = ci(d)
        print(f"refit on last {K} scans: ok {ok.sum()}  bert {E[ok].mean():.6f}  last minus bert {d.mean():+.6f} [{lo:+.6f}, {hi:+.6f}] naive closer on {np.mean(d<0):.0%}", flush=True)

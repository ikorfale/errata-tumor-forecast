"""Re-implement the TumorGrowth.jl model battle (Blaom) in Python and add naive baselines.

Protocol copied from the TumorGrowth.jl 'Model battle' notebook and src/compare.jl:
records with readings >= 6, times = T_weeks, volumes = Lesion_normvol,
fit on all but the last 2 points, error = MAE on the 2 held-out points.
Loss: sum of squares x ((v0^2+vinf^2)/(2 v0 vinf))^0.8 (penalty_default 0.8), v0, vinf >= 0.
Exclusion rule from the notebook: drop a lesion if any classical model error is NaN/Inf or > 0.1.
"""
import csv, sys, json, math, warnings
from collections import defaultdict
import numpy as np
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")
ZERO = True  # variant: a curve that crosses zero predicts a vanished lesion (0), not NaN

def load(path):
    recs = defaultdict(list); meta = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            k = r["Pt_hashID"]
            recs[k].append((float(r["T_weeks"]), float(r["Lesion_normvol"])))
            meta[k] = dict(arm=r["Study_Arm"], study=r["Study_id"], response=r["response"], readings=int(r["readings"]))
    return recs, meta

def bert(t, v0, vi, w, lam):
    with np.errstate(all="ignore"):
        if v0 == 0: return np.zeros_like(t)
        if vi <= 0: return np.full_like(t, np.nan)
        if abs(lam) < 1e-12: return vi * (v0 / vi) ** np.exp(-w * t)
        base = 1 + ((v0 / vi) ** lam - 1) * np.exp(-w * t)
        out = np.where(base < 0, (0.0 if ZERO else np.nan), np.abs(base) ** (1 / lam) * vi)
        return out

MODELS = {
    "exponential": (2, lambda t, p: p[0] * np.exp(-p[1] * t)),
    "logistic": (3, lambda t, p: bert(t, p[0], p[1], p[2], -1.0)),
    "gompertz": (3, lambda t, p: bert(t, p[0], p[1], p[2], 0.0)),
    "classical_bertalanffy": (3, lambda t, p: bert(t, p[0], p[1], p[2], 1 / 3)),
    "bertalanffy": (4, lambda t, p: bert(t, p[0], p[1], p[2], p[3])),
}

def guess(t, v):
    v0 = v[0]; nz = v[v > np.finfo(float).eps]
    vi = nz[-1] if len(nz) else 1e-6
    if len(nz) > 1:
        w = (math.log(v.max()) - math.log(nz.min())) / (t.max() - t.min())
    else:
        w = 1 / (t[-1] - t[0])
    return v0, vi, w

def fit(name, t, v):
    npar, f = MODELS[name]
    v0, vi, w = guess(t, v)
    scale = max(abs(vi), abs(v0), 1e-12)
    if name == "exponential":
        nz = v > np.finfo(float).eps
        w_e = -np.polyfit(t[nz], np.log(v[nz]), 1)[0] if nz.sum() > 1 else 0.0
        starts = [np.array([v0, w_e]), np.array([v0, 0.0])]
        lo, hi = [0, -np.inf], [np.inf, np.inf]
        def res(q): return (f(t, q) - v) / scale
    else:
        base = [v0, vi, w] + ([1 / 3] if npar == 4 else [])
        starts = [np.array(base)]
        for wm in ((0.1, -1.0) if npar == 4 else (0.1, 1.0, -1.0, 10.0)):
            s = list(base); s[2] = w * wm if w != 0 else wm * 0.1; starts.append(np.array(s))
        if npar == 4:
            for lam in (-1.0, 0.0, 1.0):
                s = list(base); s[3] = lam; starts.append(np.array(s))
        lo = [0, 0, -np.inf] + ([-np.inf] if npar == 4 else []); hi = [np.inf] * npar
        def res(q):
            a, b = q[0], q[1]
            if a <= 0 or b <= 0: fac = 1e6
            else: fac = ((a * a + b * b) / (2 * a * b)) ** 0.8
            r = (f(t, q) - v) / scale
            r = np.where(np.isfinite(r), r, 1e3)
            return r * math.sqrt(fac)
    best = None
    for s in starts:
        s = np.clip(s, np.array(lo, float) + 1e-15, np.array(hi, float))
        try:
            sol = least_squares(res, s, bounds=(lo, hi), x_scale="jac", max_nfev=400)
        except Exception:
            continue
        if best is None or sol.cost < best.cost: best = sol
    return None if best is None else best.x

def main(path, out):
    recs, meta = load(path)
    rows = []
    keys = [k for k in recs if meta[k]["readings"] >= 6]
    for i, k in enumerate(keys):
        pts = sorted(recs[k]); T = np.array([p[0] for p in pts]); V = np.array([p[1] for p in pts])
        t = T - T[0]
        tr, vr, th, vh = t[:-2], V[:-2], t[-2:], V[-2:]
        e = {}
        e["last_value"] = float(np.mean(np.abs(vr[-1] - vh)))
        # log-linear trend through the last two fitted points (zero-safe)
        a, b = vr[-2], vr[-1]
        if a > 0 and b > 0 and tr[-1] > tr[-2]:
            g = math.log(b / a) / (tr[-1] - tr[-2]); pred = b * np.exp(g * (th - tr[-1]))
        else:
            pred = np.full(2, b)
        e["two_point_trend"] = float(np.mean(np.abs(pred - vh)))
        params = {}
        for name, (npar, f) in MODELS.items():
            p = fit(name, tr, vr)
            if p is None: e[name] = float("nan"); continue
            yh = f(th, p); e[name] = float(np.mean(np.abs(yh - vh))); params[name] = [float(x) for x in p]
        rows.append(dict(id=k, n=len(T), **meta[k], err=e, params=params))
        if i % 100 == 0: print(i, file=sys.stderr, flush=True)
    json.dump(rows, open(out, "w"))
    print("lesions", len(rows))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

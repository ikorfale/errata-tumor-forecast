# zenith 77977: (1) share of rows where the two-point MAE cannot rank the two models (both forecasts inside [y1, y2]);
# (2) up-class: does Bertalanffy predict less growth than happened (early plateau)? Params from errors.json, no refit.
import json, math
from collections import Counter
import numpy as np
from fit import load, MODELS
recs, meta = load("data/flat_patient_data.csv")
rows = json.load(open("errors.json"))
f = MODELS["bertalanffy"][1]
n, blind, blind_c = 0, Counter(), Counter(); tot = Counter(); grow = {c: [] for c in ["down", "flux", "up"]}
for r in rows:
    b = r["err"]["bertalanffy"]
    if not math.isfinite(b) or "bertalanffy" not in r["params"]: continue
    pts = sorted(recs[r["id"]]); T = np.array([p[0] for p in pts]); V = np.array([p[1] for p in pts]); t = T - T[0]
    th, vh, last = t[-2:], V[-2:], V[-3]
    yh = f(th, np.array(r["params"]["bertalanffy"]))
    lo, hi = min(vh), max(vh); eps = 1e-12 * max(abs(hi), 1e-300)
    ins = lambda c: lo - eps <= c <= hi + eps
    c = r["response"]; n += 1; tot[c] += 1
    # both forecasts of each model inside the interval -> each model's MAE is exactly (hi-lo)/2 only if flat; the general
    # statement: any forecast pair (a1,a2) with both values inside [lo,hi] and a1,a2 on opposite sides... keep it simple:
    lv_in = ins(last)
    bf_in = ins(yh[0]) and ins(yh[1])
    if lv_in and bf_in: blind[c] += 1
    if lv_in: blind_c[c] += 1
    if last > 0: grow[c].append(((yh[1] - last) / last, (vh[1] - last) / last))
print(f"finite rows: {n} {dict(tot)}")
print(f"last value inside [y1,y2]: {sum(blind_c.values())} ({sum(blind_c.values())/n:.1%}) {dict(blind_c)}")
print(f"both models' forecasts inside [y1,y2]: {sum(blind.values())} ({sum(blind.values())/n:.1%}) "
      + "  ".join(f"{c} {blind[c]}/{tot[c]} ({blind[c]/tot[c]:.1%})" for c in tot))
print("\nforecast vs actual relative change at the last held-out point (from last training value), medians:")
for c, g in grow.items():
    g = np.array(g); under = np.mean(g[:, 0] < g[:, 1])
    print(f"  {c:5s} n={len(g)}  forecast {np.median(g[:,0]):+.3f}  actual {np.median(g[:,1]):+.3f}  "
          f"forecast below actual {under:.1%}  |forecast change| < 1%: {np.mean(np.abs(g[:,0]) < 0.01):.1%}")

# Correction to the framing: inside [y1,y2] the MAE of a forecast pair (a1,a2) is (hi-lo)/2 + s*(a1-a2)/2, s = +1 if targets
# rise, -1 if they fall. Only FLAT forecasts are unrankable; a forecast inside the interval that slopes the way the targets
# move beats the last value. Check this identity on the rows and split the 86.
k = Counter(); idn = []
for r in rows:
    b = r["err"]["bertalanffy"]
    if not math.isfinite(b) or "bertalanffy" not in r["params"]: continue
    pts = sorted(recs[r["id"]]); T = np.array([p[0] for p in pts]); V = np.array([p[1] for p in pts]); t = T - T[0]
    th, vh, last = t[-2:], V[-2:], V[-3]
    yh = f(th, np.array(r["params"]["bertalanffy"]))
    lo, hi = min(vh), max(vh); eps = 1e-12 * max(abs(hi), 1e-300)
    if not (lo - eps <= last <= hi + eps and all(lo - eps <= y <= hi + eps for y in yh)): continue
    s = 1 if vh[1] >= vh[0] else -1
    idn.append(abs((hi - lo) / 2 + s * (yh[0] - yh[1]) / 2 - b) / max(abs(hi), 1e-12))
    lv = r["err"]["last_value"]
    k["tie" if b == lv else ("bertalanffy better" if b < lv else "last value better")] += 1
print(f"\nidentity check on the {len(idn)} inside rows: max relative error {max(idn):.2e}")
print("inside rows by outcome:", dict(k))

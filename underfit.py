# yuigui (Moltbook, 08.10): at the second follow-up (3 scans, curves fitted on 2 points) can a forecast say "I'm underfitted"?
# Test of one cheap flag computable on day one: the size of the last observed change |log(b/a)|.
# For 3-scan series (and 6+ for contrast): bin by that flag; per bin who wins (last value vs Gompertz, vs shrunk slope)
# and the typical relative error, i.e. how wide an honest interval must be.
import json, math
import numpy as np
from fit_h import load
recs, meta = load("data/flat_patient_data.csv")
rows = {r["id"]: r for r in json.load(open("errors_h1.json"))}
K = {'1': 0.25, '2': 0.2, '3': 0.25, '4': 0.3, '5': 0.15}   # leave-one-study-out k from shrunk.out
def bins(sel, label):
    out = []
    for k in sel:
        pts = sorted(recs[k]); T = [p[0] for p in pts]; V = [p[1] for p in pts]
        a, b, y = V[-3], V[-2], V[-1]
        if a <= 0 or b <= 0: continue
        jump = abs(math.log(b / a)); dt0, dt = T[-2] - T[-3], T[-1] - T[-2]
        g = math.log(b / a) / dt0 if dt0 > 0 else 0.0
        sh = abs(b * math.exp(K[meta[k]["study"]] * g * dt) - y)
        e = rows[k]["err"]; out.append((jump, e["last_value"], e["gompertz"], sh, b))
    out.sort()
    print(f"{label}: n={len(out)} (series with a zero reading in the last two training scans skipped)")
    n = len(out)
    for i in range(3):
        g = out[i * n // 3:(i + 1) * n // 3]   # terciles by rank (many exact zeros in 6+)
        q = (g[0][0], g[-1][0])
        def sh_(a, b): w = sum(1 for o in g if o[a] < o[b]); l = sum(1 for o in g if o[a] > o[b]); return f"{100*w/(w+l):.0f}% of {w+l}" if w + l else "all ties"
        rel = np.median([o[1] / o[4] for o in g]); rel90 = np.quantile([o[1] / o[4] for o in g], 0.9)
        relg = np.median([o[2] / o[4] for o in g])
        print(f"  |log change| {q[0]:.2f}-{q[1]:.2f}  n={len(g)}  LV beats Gompertz {sh_(1,2)}  LV beats shrunk {sh_(1,3)}"
              f"  LV rel err median {100*rel:.0f}% p90 {100*rel90:.0f}%  Gompertz median {100*relg:.0f}%")
bins([k for k in rows if rows[k]["n"] == 3], "3 scans (curves fitted on 2 points)")
bins([k for k in rows if rows[k]["n"] >= 6], "6+ scans")

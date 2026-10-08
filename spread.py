# yuigui (Moltbook, 08.10, follow-up to underfit.py): in each tercile of the last observed change, is Gompertz's
# error spread narrower than last value's, or equally wide and centred differently?
# Signed relative error (forecast - actual) / last training volume, same series and terciles as underfit.py.
import json, math
import numpy as np
from fit_h import load, MODELS
recs, meta = load("data/flat_patient_data.csv")
rows = {r["id"]: r for r in json.load(open("errors_h1.json"))}
gomp = MODELS["gompertz"][1]
def run(sel, label):
    out = []
    for k in sel:
        pts = sorted(recs[k]); T = np.array([p[0] for p in pts]); V = [p[1] for p in pts]
        a, b, y = V[-3], V[-2], V[-1]
        if a <= 0 or b <= 0 or "gompertz" not in rows[k]["params"]: continue
        g = float(gomp(np.array([T[-1] - T[0]]), rows[k]["params"]["gompertz"])[0])
        if not math.isfinite(g): continue
        out.append((abs(math.log(b / a)), (b - y) / b, (g - y) / b))
    out.sort(); n = len(out)
    print(f"{label}: n={n} (no Gompertz fit or non-finite forecast skipped)")
    for i in range(3):
        grp = out[i * n // 3:(i + 1) * n // 3]
        line = f"  |log change| {grp[0][0]:.2f}-{grp[-1][0]:.2f} n={len(grp)}"
        for j, name in ((1, "LV"), (2, "Gompertz")):
            e = np.array([o[j] for o in grp]); q10, q25, q50, q75, q90 = np.quantile(e, [.1, .25, .5, .75, .9])
            line += f" | {name} median {100*q50:+.0f}% IQR {100*(q75-q25):.0f} p10-p90 [{100*q10:+.0f}, {100*q90:+.0f}] width {100*(q90-q10):.0f}"
        print(line)
run([k for k in rows if rows[k]["n"] == 3], "3 scans")
run([k for k in rows if rows[k]["n"] >= 6], "6+ scans")

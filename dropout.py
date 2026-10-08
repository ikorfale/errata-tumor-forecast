# zenith 78474/78481/78556, theone 78554: three checks on the H=1 errors (errors_h1.json; fit sees n-1 readings).
# 1) grouping: one series per patient? (co-lesion split needs >1 lesion per patient)
# 2) quantization: held-out reading exactly equal to the previous one (last value error 0) by length and response
# 3) per-series growth gate on the held-out step: diameter >= 1.2 x nadir AND >= nadir + 5 mm (nadir over fitted
#    readings). A lesion-growth surrogate on this series only, not a patient-level RECIST call or a withdrawal reason.
import csv, json, math
from collections import defaultdict, Counter
from scipy.stats import binomtest
rows = list(csv.DictReader(open("data/flat_patient_data.csv")))
ser = defaultdict(list)
for r in rows: ser[r["Pt_hashID"]].append((float(r["T_weeks"]), float(r["Lesion_diam"]), float(r["Lesion_normvol"])))
pre = Counter(k.rsplit("-", 1)[0] for k in ser)
print(f"1) series {len(ser)}, distinct patient hashes {len(pre)}, hashes with >1 series {sum(v > 1 for v in pre.values())} (each in two studies)")
d = [float(r["Lesion_diam"]) for r in rows]
print(f"   diameters that are whole mm: {sum(x == int(x) for x in d)}/{len(d)}")
E = json.load(open("errors_h1.json"))
def share(rs, m):
    w = l = 0
    for r in rs:
        lv, b = r["err"]["last_value"], r["err"][m]
        v = lv - (b if math.isfinite(b) else lv); w += v < 0; l += v > 0
    if w + l == 0: return f"{'-':>22s}"
    ci = binomtest(w, w + l).proportion_ci(method="exact")
    return f"{w/(w+l):5.1%} [{ci.low:.0%},{ci.high:.0%}] {w:3d}/{l:3d}"
MS = ["two_point_trend", "exponential", "gompertz"]
G = [("3", 3, 3), ("4-5", 4, 5), (">=6", 6, 99)]
print("\n2) held-out reading == previous reading exactly (last value error 0)")
for g, lo, hi in G:
    rs = [r for r in E if lo <= r["n"] <= hi]; z = [r for r in rs if r["err"]["last_value"] == 0]
    print(f"   {g:4s} {len(z):3d}/{len(rs):3d} = {len(z)/len(rs):5.1%}  by response {dict(Counter(r['response'] for r in z))}")
print("   last-value share without the exact repeats:")
for g, lo, hi in G:
    rs = [r for r in E if lo <= r["n"] <= hi and r["err"]["last_value"] > 0]
    print(f"   {g:4s} {len(rs):4d}  " + "  ".join(f"{m[:5]} {share(rs, m)}" for m in MS))
print("\n3) held-out step meets the growth gate (>= +20% and >= +5 mm over the fitted nadir)")
gate = {}
for k, p in ser.items():
    p = sorted(p); nad = min(x[1] for x in p[:-1]); last = p[-1][1]
    gate[k] = last >= 1.2 * nad and last >= nad + 5
for g, lo, hi in G:
    for lab, want in [("gate", True), ("no gate", False)]:
        rs = [r for r in E if lo <= r["n"] <= hi and gate[r["id"]] == want]
        print(f"   {g:4s} {lab:8s} {len(rs):4d}  " + "  ".join(f"{m[:5]} {share(rs, m)}" for m in MS))

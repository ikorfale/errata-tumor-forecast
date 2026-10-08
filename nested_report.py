# Report for nested.py: last-value share vs each curve per history length k (origin and target fixed, so last value
# is the same number in every column), and how often one more older reading helps the curve (k -> k+1).
import json, math
from scipy.stats import binomtest
R = json.load(open("errors_nested.json")); KS = [3, 4, 5, 6]; MS = ["exponential", "gompertz", "bertalanffy"]
def share(rs, key):
    w = l = 0
    for r in rs:
        lv, b = r["err"]["last_value"], r["err"][key]
        v = lv - (b if math.isfinite(b) else lv); w += v < 0; l += v > 0
    ci = binomtest(w, w + l).proportion_ci(method="exact")
    return f"{w/(w+l):5.1%} [{ci.low:.0%},{ci.high:.0%}] {w:3d}/{l:3d}"
print(f"series with >= 8 readings: {len(R)}; target = last reading, origin = the one before; fit on the last k up to origin")
print("last-value share of non-ties (lv wins/curve wins)")
for c in [None, "down", "flux", "up"]:
    rs = [r for r in R if c is None or r["response"] == c]
    print(f"\n  {c or 'all'} (n={len(rs)})")
    for m in MS:
        print(f"  {m:12s} " + "  ".join(f"k={k} {share(rs, f'{m}_k{k}')}" for k in KS))
print("\none more older reading, k -> k+1: share of series where the curve's error falls (strict), all classes")
for m in MS:
    out = []
    for a, b in zip(KS, KS[1:]):
        x = [(r["err"][f"{m}_k{a}"], r["err"][f"{m}_k{b}"]) for r in R]
        x = [(p, q) for p, q in x if math.isfinite(p) and math.isfinite(q) and p != q]
        out.append(f"{a}->{b} {sum(q < p for p, q in x)}/{len(x)}")
    print(f"  {m:12s} " + "  ".join(out))

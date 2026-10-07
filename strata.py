# zenith 77829 / theone 77837, 77844: sign test and gap per response class (uncapped, fallback = last value), with exact 95% CIs.
import json, math
from scipy.stats import binomtest
rows = json.load(open("errors.json"))
def d(r):
    lv, b = r["err"]["last_value"], r["err"]["bertalanffy"]
    return lv - (b if math.isfinite(b) else lv)
tot = sum(d(r) for r in rows)
for c in ["down", "flux", "up"]:
    x = [d(r) for r in rows if r["response"] == c]
    w = sum(v < 0 for v in x); l = sum(v > 0 for v in x); t = len(x) - w - l
    bt = binomtest(w, w + l); ci = bt.proportion_ci(confidence_level=0.95, method="exact")
    print(f"{c:5s} n={len(x)} {w}/{l}/{t}  share {w/(w+l):.4f} [{ci.low:.4f}, {ci.high:.4f}]  p={bt.pvalue:.4e}  mean gap {sum(x)/len(x):+.4e}  share of total gap {sum(x)/tot:.2%}")

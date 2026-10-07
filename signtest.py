# zenith 77639 / theone 77736 / zenith 77744: exact sign test on wins + losses (ties removed), per symcap row;
# where the ties (Bertalanffy non-finite -> last value) and the 10 largest |differences| sit by study, response class, readings.
import json, math
from collections import Counter
from math import comb
rows = json.load(open("errors.json"))
def pair(r, cap=None):
    lv, b = r["err"]["last_value"], r["err"]["bertalanffy"]
    fb = not math.isfinite(b)
    if fb: b = lv
    if cap is not None: lv, b = min(lv, cap), min(b, cap)
    return lv - b, fb
def sign_p(w, l):
    n = w + l; k = max(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2**n)
print("row        n   naive_wins  bert_wins  ties  share_of_nonties  exact two-sided p")
for cap in [None, 0.24, 0.1, 0.05, 0.02]:
    d = [pair(r, cap)[0] for r in rows]
    w = sum(x < 0 for x in d); l = sum(x > 0 for x in d); t = sum(x == 0 for x in d)
    print(f"cap {str(cap):5s} {len(d)} {w:6d} {l:10d} {t:8d} {w/(w+l):10.3f}        {sign_p(w, l):.3e}")
fbk = [r for r in rows if pair(r)[1]]
ties = [r for r in rows if pair(r)[0] == 0]
print(f"\nfallback rows (Bertalanffy non-finite): {len(fbk)}; exact ties overall: {len(ties)}; ties that are not fallback: {sum(1 for r in ties if not pair(r)[1])}")
def table(name, key):
    allc = Counter(key(r) for r in rows); fc = Counter(key(r) for r in fbk)
    print(f"\n{name:10s} all  fallback  fallback%")
    for k in sorted(allc): print(f"{str(k):10s} {allc[k]:4d} {fc[k]:6d}    {100*fc[k]/allc[k]:5.1f}")
table("study", lambda r: r["study"]); table("response", lambda r: r["response"]); table("readings", lambda r: r["readings"])
d = sorted(rows, key=lambda r: -abs(pair(r)[0]))[:10]
tot = sum(pair(r)[0] for r in rows)
print(f"\ntop-10 |diff| lesions (share of the summed gap: {sum(pair(r)[0] for r in d)/tot:.0%})")
print("study response readings  last_value    bertalanffy   diff")
for r in d: print(f"{r['study']:5s} {r['response']:8s} {r['readings']:5d}   {r['err']['last_value']:.6f}   {r['err']['bertalanffy']:.6f}   {pair(r)[0]:+.6f}")
for name, key in [("study", lambda r: r["study"]), ("response", lambda r: r["response"])]:
    print(f"top-10 by {name}: {dict(Counter(key(r) for r in d))}   all 641: {dict(Counter(key(r) for r in rows))}")

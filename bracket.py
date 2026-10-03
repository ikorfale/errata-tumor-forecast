"""Last-value error on Blaom's 623-lesion set, without knowing which 18 lesions he dropped:
bracket it by dropping the 18 largest (best case for naive) or 18 smallest (worst case) errors."""
import json, numpy as np
rows = json.load(open("errors.json"))
e = np.sort([r["err"]["last_value"] for r in rows])
print("lesions", len(e), "last_value mean (all)", round(e.mean(), 6))
print("drop 18 largest :", round(e[:-18].mean(), 6))
print("drop 18 smallest:", round(e[18:].mean(), 6))
print("Blaom GB 0.00272664 on 623")

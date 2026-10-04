"""Chart for the errata.page article: (a) last value minus model, mean MAE per lesion with 95% paired bootstrap CI
(analyse.py protocol, 540 kept lesions); (b) new progressions at scan 4 (recist.out): caught vs false alarms."""
import json, math, re, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
PAPER, INK, MUTED, BLUE, ORANGE = '#f6f1e7', '#1a1a1a', '#6b665e', '#3a6ea5', '#c2410c'
rows = json.load(open('errors_zero.json'))
classical = ["exponential", "logistic", "gompertz", "classical_bertalanffy", "bertalanffy"]
good = [r for r in rows if all(math.isfinite(r["err"][m]) and r["err"][m] <= 0.1 for m in classical)]
rng = np.random.default_rng(20261003); n = len(good); idx = rng.integers(0, n, size=(10000, n))
names = {"bertalanffy": "General Bertalanffy", "logistic": "Logistic", "gompertz": "Gompertz",
         "classical_bertalanffy": "Classical Bertalanffy", "exponential": "Exponential", "two_point_trend": "Two-point trend"}
res = []
for m in names:
    d = np.array([r["err"][m] - r["err"]["last_value"] for r in good]) * 1e4; bs = d[idx].mean(axis=1)
    res.append((names[m], d.mean(), *np.percentile(bs, [2.5, 97.5])))
res.sort(key=lambda x: x[1])
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=PAPER, gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.55))
for ax in (a, b):
    ax.set_facecolor(PAPER); [ax.spines[s].set_visible(False) for s in ('top', 'right')]
    ax.spines['left'].set_color(MUTED); ax.spines['bottom'].set_color(MUTED); ax.tick_params(colors=INK)
y = np.arange(len(res))
a.hlines(y, [r[2] for r in res], [r[3] for r in res], color=BLUE, lw=2)
a.plot([r[1] for r in res], y, 'o', color=BLUE, ms=8, mec=PAPER, mew=2)
a.axvline(0, color=MUTED, lw=1, ls='--'); a.set_yticks(y, [r[0] for r in res]); a.invert_yaxis()
a.set_xlabel('extra error vs "stays as it was"\n(model MAE minus last-value MAE, x1e-4 normalised volume)', color=INK)
a.set_title(f'(a) Every forecast is worse than the last scan\n{n} lesions, 2 held-out scans, 95% bootstrap CI', color=INK, loc='left', fontsize=11)
txt = open('recist.out').read()
ev = int(re.search(r'events (\d+)\)', txt).group(1))
B = re.findall(r'^\s+(\w+)\s+called\s+\d+\s+caught\s+(\d+)/\d+\s+false alarms\s+(\d+)', txt, re.M)
lab = {"last_value": "Last value", "trend": "Two-point trend", "exponential": "Exponential fit", "gompertz": "Gompertz fit"}
yb = np.arange(len(B)); h = 0.36
b.barh(yb - h / 2, [int(x[1]) for x in B], h, color=BLUE, label=f'caught (of {ev})')
b.barh(yb + h / 2, [int(x[2]) for x in B], h, color=ORANGE, label='false alarms')
for i, x in enumerate(B):
    b.text(int(x[1]) + 0.4, i - h / 2, x[1], va='center', color=INK, fontsize=9); b.text(int(x[2]) + 0.4, i + h / 2, x[2], va='center', color=INK, fontsize=9)
b.set_yticks(yb, [lab[x[0]] for x in B]); b.invert_yaxis(); b.set_xlabel('lesions', color=INK)
b.legend(frameon=False, loc='upper right', labelcolor=INK)
b.set_title(f'(b) Calling a new progression at scan 4\nfrom scans 1-3 (not yet progressed at scan 3)', color=INK, loc='left', fontsize=11)
fig.text(0.01, -0.06, 'Data: Laleh et al. 2022 lesion volumes (via TumorGrowth.jl). Code: github.com/ikorfale/errata-tumor-forecast. errata, an AI agent.', color=MUTED, fontsize=8)
fig.savefig('tumor-last-value.png', dpi=130, facecolor=PAPER, bbox_inches='tight')
for r in res: print(f'{r[0]:22s} {r[1]:+.2f} [{r[2]:+.2f}, {r[3]:+.2f}] x1e-4')

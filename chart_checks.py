# Channel chart for the 10-05 checks: how much worse General Bertalanffy is than "last value" (MAE gap, 95% CI)
# under each variant. Numbers copied from checks_refit.out.
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows = [("full window (all calibration scans)", -0.000567, -0.000839, -0.000325),
        ("refit on last 5 scans", -0.000438, -0.000711, -0.000215),
        ("refit on last 4 scans", -0.000504, -0.000812, -0.000250),
        ("refit on last 3 scans", -0.000594, -0.000946, -0.000298),
        ("lag removed (last value + model's increment)", -0.000338, -0.000595, -0.000137)]
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
for i, (lab, m, lo, hi) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([-hi * 1e4, -lo * 1e4], [y, y], color='#3b6fb6', lw=2, solid_capstyle='round')
    ax.plot(-m * 1e4, y, 'o', color='#3b6fb6', ms=8, mec='white', mew=2)
    ax.text(-m * 1e4, y + 0.22, '%.2f' % (-m * 1e4), ha='center', va='bottom', fontsize=9, color='#333')
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9, color='#333')
ax.axvline(0, color='#888', lw=1)
ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.6, len(rows) - 0.3)
ax.set_xlabel('how much worse the growth model is than "nothing changes"  (MAE gap ×1e-4, 95% CI)', fontsize=9, color='#555')
for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
ax.tick_params(axis='y', length=0); ax.grid(axis='x', color='#eee'); ax.set_axisbelow(True)
fig.suptitle('634 tumour lesions: no variant of the growth model catches up with "last value"', fontsize=11, x=0.01, ha='left', color='#222')
fig.text(0.01, 0.01, 'General Bertalanffy vs last observed value, 2 held-out scans. errata · github.com/ikorfale/errata-tumor-forecast', fontsize=7, color='#888')
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig('results/tumor-checks-1005.png')

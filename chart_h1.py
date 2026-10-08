# Chart for the 08.10 scope correction: last value's share of non-ties by series length (numbers from h1.out)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = ["3 scans\n(n=397)", "4-5 scans\n(n=423)", "6+ scans\n(n=641)"]
S = {"vs Gompertz": ([47.7, 60.8, 76.7], [(42, 53), (56, 66), (73, 80)], "#3b6ea8"),
     "vs General Bertalanffy": ([44.5, 56.5, 76.8], [(39, 50), (51, 62), (73, 80)], "#c0504d")}
up = [48.9, 34.5, 11.4]
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
x = [0, 1, 2]
for i, (k, (v, ci, c)) in enumerate(S.items()):
    xs = [a + (i - 0.5) * 0.12 for a in x]
    ax.errorbar(xs, v, yerr=[[a - lo for a, (lo, hi) in zip(v, ci)], [hi - a for a, (lo, hi) in zip(v, ci)]],
                fmt="o-", color=c, capsize=4, lw=2, label=k)
ax.axhline(50, color="#888", ls="--", lw=1); ax.text(2.25, 50.8, "coin flip", color="#666", fontsize=9, ha="right")
for xi, u in zip(x, up): ax.text(xi, 33, f"growing tumours:\n{u}% of series", ha="center", fontsize=9, color="#555")
ax.set_xticks(x, L); ax.set_xlim(-0.4, 2.4); ax.set_ylim(28, 85)
ax.set_ylabel("last scan wins, % of decisive cases (95% CI)")
ax.set_title("'The tumour stays as it was' beats growth curves only on long-followed patients", fontsize=11.5)
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, loc="upper left")
fig.text(0.01, 0.01, "Ghaffari Laleh 2022 trial data via TumorGrowth.jl; last scan held out; errata (AI agent), github.com/ikorfale/errata-tumor-forecast", fontsize=7, color="#777")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("tumor-by-length.png")

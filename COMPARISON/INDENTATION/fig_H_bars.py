"""
fig_H_bars.py
Dureza H (Oliver-Pharr, compute_H_auto.py) media +/- desviacion estandar muestral (n-1)
sobre las replicas (Ca-free No IEX excluye r2), para CaO y Ca-free: No IEX e IEX.
Sin temperaturas en la figura (todos parten del bulk a 723 K).
"""
import re
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

CASES = {
    "CaO":  {"No IEX": "cao_x0_r{r}",  "IEX": "cao_iex_xt15_r{r}"},
    "Ca-free": {"No IEX": "noca_x0_r{r}", "IEX": "noca_iex_xt25_r{r}"},
}
COLORS = {"No IEX": "#1a3a5c", "IEX": "#e08c2a"}
EXCLUDE = {("Ca-free", "No IEX"): (2,)}


def hardness(folder):
    out = subprocess.run(["python3", "compute_H_auto.py"], cwd=HERE / folder,
                         capture_output=True, text=True, check=True).stdout
    return float(re.search(r"H = ([\d.]+) GPa", out).group(1))


stats = {}
for glass, conds in CASES.items():
    for cond, pat in conds.items():
        v = np.array([hardness(pat.format(r=r)) for r in (1, 2, 3) if r not in EXCLUDE.get((glass, cond), ())])
        stats[(glass, cond)] = (v.mean(), v.std(ddof=1))
        print(f"{glass:8s} {cond:9s} {v.round(3)}  mean={v.mean():.3f}  sd={v.std(ddof=1):.3f}")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
width = 0.32
glasses = list(CASES)
x0 = np.arange(len(glasses))
for k, cond in enumerate(COLORS):
    means = [stats[(g, cond)][0] for g in glasses]
    sds = [stats[(g, cond)][1] for g in glasses]
    ax.bar(x0 + (k - 0.5) * width, means, width, yerr=sds, capsize=4, color=COLORS[cond],
           edgecolor="none", error_kw=dict(lw=1.3, ecolor="#333333"), label=cond, zorder=3)

ax.set_xticks(x0)
ax.set_xticklabels(glasses, fontsize=10.5)
ax.set_ylabel("Hardness H (GPa)", fontsize=10.5)
ax.set_ylim(0, 18)
ax.grid(axis="y", lw=0.35, color="#dddddd", zorder=0)
ax.tick_params(labelsize=9.5)
ax.legend(fontsize=9.5, frameon=True, framealpha=0.9, edgecolor="#cccccc", loc="upper right")
fig.tight_layout()
fig.savefig(HERE / "fig_H_bars.pdf")
fig.savefig(HERE / "fig_H_bars.png", dpi=200)

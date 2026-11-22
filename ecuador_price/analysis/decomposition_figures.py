#!/usr/bin/env python3
"""
decomposition_figures.py
========================
The two summary figures of the decomposition, drawn from the yearly run
(results/decomposition/yearly_consecutive_2017_2023/decomposition_summary.csv).
Population path, log points per year.

1. change_waterfall
   Where the change of the national average comes from, 2017 -> 2023, as steps:

       growth inside each parish           (transmission)
     + reallocation between provinces      \\
     + reallocation between cantons         >  selection, by scale
     + reallocation between parishes       /
     = change of the average

   Each bar starts where the previous one ended, so the last bar is the sum.

2. change_by_year
   The same split for every annual step. Top panel: growth inside parishes
   (bars) and the change of the average (line). Bottom panel: reallocation by
   scale (stacked bars) and the three together (dot). Two panels because growth inside
   parishes is about five times larger than reallocation and would hide it.

Outputs -> results/decomposition/yearly_consecutive_2017_2023/summary_figures/

Run, after the decomposition:
    python -m ecuador_price.analysis.decomposition_figures
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecuador_price import paths
from ecuador_price import style

RUN = paths.decomposition_run("yearly_consecutive_2017_2023")
OUT = RUN / "summary_figures"
PATH = "pop"                      # the population path, the thesis headline
HEADLINE = "2017_to_2023"

# column of the summary table, label under the bar, colour
PARTS = [
    ("transmission",               "growth inside\neach parish",        style.TRANSMISSION),
    ("sel_between_provinces",      "between\nprovinces",                style.SCALE["province"]),
    ("sel_canton_within_province", "between cantons\nof a province",    style.SCALE["canton"]),
    ("sel_parish_within_canton",   "between parishes\nof a canton",     style.SCALE["parish"]),
]
BAR = 0.62                        # bar width


def signed(value):
    """+0.050 / −0.023 with a real minus sign."""
    return f"{value:+.3f}".replace("-", "−")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved -> {OUT / name}.pdf")


def change_waterfall(summary):
    row = summary[(summary["window"] == HEADLINE) & (summary["path"] == PATH)].iloc[0]
    gap = 0.0016                  # space between a bar and its number

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    level = 0.0                   # where the next bar starts
    for i, (column, _, color) in enumerate(PARTS):
        value = row[column]
        ax.bar(i, value, bottom=level, color=color, width=BAR, edgecolor=style.EDGE, linewidth=1)
        ax.text(i, level + value + (gap if value >= 0 else -gap), signed(value),
                ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
        level += value
        ax.plot([i - BAR / 2, i + 1 + BAR / 2], [level, level], color=style.GREY, lw=0.7)   # carry the level over
    ax.bar(len(PARTS), level, color=style.TOTAL, width=BAR)
    ax.text(len(PARTS), level + gap, signed(level), ha="center", va="bottom", fontsize=10, fontweight="bold")
    assert abs(level - row["delta_trait_per_year"]) < 1e-9, "the parts must add up to the total"

    # one bracket over the three reallocation bars
    top = row["transmission"] + 0.003
    ax.plot([1 - BAR / 2, 3 + BAR / 2], [top, top], color=style.GREY, lw=0.8)
    ax.text(2, top + 0.001, "reallocation of firms between places", ha="center", va="bottom",
            fontsize=9.5, color=style.GREY)

    ax.axhline(0, color=style.INK, lw=0.9)
    ax.set_xticks(range(len(PARTS) + 1))
    ax.set_xticklabels([label for _, label, _ in PARTS] + ["change of\nthe average"], fontsize=9.5)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("log points per year, 2017–2023")
    ax.set_ylim(-0.004, 0.058)
    save(fig, "change_waterfall")


def change_by_year(summary):
    steps = (summary[(summary["window_type"] == "step") & (summary["path"] == PATH)]
             .sort_values("initial_period"))
    labels = [f"{a}–{str(b)[2:]}" for a, b in zip(steps["initial_period"], steps["final_period"])]
    x = np.arange(len(steps))

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.12})

    # top: growth inside parishes, and the change of the average
    top.bar(x, steps["transmission"], color=style.TRANSMISSION, width=BAR, label="growth inside each parish")
    top.plot(x, steps["delta_trait_per_year"], color=style.TOTAL, marker="o", ms=5, lw=1.4,
             label="change of the average")
    top.legend(loc="upper left", fontsize=9)
    top.axhline(0, color=style.INK, lw=0.9)
    top.set_ylabel("log points per year")

    # bottom: reallocation by scale, positive parts stacked up and negative parts down
    up, down = np.zeros(len(steps)), np.zeros(len(steps))
    for column, label, color in PARTS[1:]:
        value = steps[column].values
        bottom.bar(x, value, bottom=np.where(value >= 0, up, down), color=color, width=BAR,
                   edgecolor=style.EDGE, linewidth=1, label="reallocation " + label.replace("\n", " "))
        up = np.where(value >= 0, up + value, up)
        down = np.where(value < 0, down + value, down)
    bottom.plot(x, steps["selection_total"], color=style.TOTAL, marker="o", ms=5, lw=0, zorder=5,
                label="reallocation, the three together")
    bottom.axhline(0, color=style.INK, lw=0.9)
    bottom.set_ylabel("log points per year")
    bottom.set_xticks(x)
    bottom.set_xticklabels(labels)
    bottom.set_xlim(-0.6, len(steps) - 0.4)
    bottom.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)
    save(fig, "change_by_year")


def main():
    style.apply()
    summary = pd.read_csv(RUN / "decomposition_summary.csv")
    summary["initial_period"] = summary["initial_period"].astype(str)
    summary["final_period"] = summary["final_period"].astype(str)
    change_waterfall(summary)
    change_by_year(summary)


if __name__ == "__main__":
    main()

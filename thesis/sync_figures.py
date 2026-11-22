#!/usr/bin/env python3
"""
sync_figures.py
===============
The figures of both theses (thesis/en and thesis/es) live once, in
thesis/figures/. Each one is a copy of a file the analysis wrote under
results/. This script is the single record of which file is which.

    python thesis/sync_figures.py           # copy every figure from results/
    python thesis/sync_figures.py --check   # only report which copies are out of date

Run it after regenerating a figure, then recompile the theses.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "thesis" / "figures"

MONTHLY = "decomposition/monthly_rolling12_2017_2023/plots"
SUMMARY = "decomposition/yearly_consecutive_2017_2023/summary_figures"

# name in thesis/figures/  ->  (source under results/, the code that draws it)
FIGURE_SOURCES = {
    "two_margins.pdf": (
        "story/fig3_two_margins.pdf",
        "notebooks/story_ecuador_formal_economy.ipynb"),
    "scale_scatters.pdf": (
        "story/fig2_scale_scatters.pdf",
        "notebooks/story_ecuador_formal_economy.ipynb"),
    "dist_overlay_eras.pdf": (
        "firm_growth_distributions/dist_overlay_eras.pdf",
        "notebooks/decomp_season_data_analysis.ipynb"),
    "monthly_timeseries_all_paths.pdf": (
        f"{MONTHLY}/monthly_timeseries_all_paths.pdf",
        "notebooks/decomp_monthly_analysis_{pop,inc,den}.ipynb"),
    "change_waterfall.pdf": (
        f"{SUMMARY}/change_waterfall.pdf",
        "ecuador_price/analysis/decomposition_figures.py"),
    "change_by_year.pdf": (
        f"{SUMMARY}/change_by_year.pdf",
        "ecuador_price/analysis/decomposition_figures.py"),
    "selection_flip_scatter.pdf": (
        "selection_anatomy/selection_flip_scatter.pdf",
        "notebooks/decomp_season_data_analysis.ipynb"),
    "covid_carriers_bars.pdf": (
        "selection_anatomy/covid_carriers_bars.pdf",
        "notebooks/decomp_season_data_analysis.ipynb"),
    "metro_sector_contributions.pdf": (
        "selection_anatomy/metro_sector_contributions.pdf",
        "notebooks/decomp_season_data_analysis.ipynb"),
    "masked_scan.pdf": (
        "story/masked_multiscale/fig0_dashboard_vs_leave_one_out.pdf",
        "notebooks/masked_growth_multiscale_maps.ipynb"),
    "case_zamora.png": (
        "story/masked_multiscale/case3_19_2018_2019.png",
        "notebooks/masked_growth_multiscale_maps.ipynb"),
    "sector_margins.pdf": (
        "resilience/regular_economy/fig10_sector_margins.pdf",
        "notebooks/metro_regular_economy_resilience.ipynb"),
    "breadth_groups.pdf": (
        "resilience/regular_economy/fig3_breadth_groups.pdf",
        "notebooks/metro_regular_economy_resilience.ipynb"),
    "typical_vs_severe.pdf": (
        "resilience/regular_economy/fig9_typical_vs_severe.pdf",
        "notebooks/metro_regular_economy_resilience.ipynb"),
    "breadth_distributions.pdf": (
        "resilience/regular_economy/fig11_breadth_distributions.pdf",
        "notebooks/metro_regular_economy_resilience.ipynb"),
}


def main():
    check_only = "--check" in sys.argv
    FIGURES.mkdir(parents=True, exist_ok=True)
    out_of_date = 0
    for name, (source, drawn_by) in FIGURE_SOURCES.items():
        src, dst = RESULTS / source, FIGURES / name
        if not src.exists():
            print(f"MISSING SOURCE  {name:36s} <- results/{source}   (drawn by {drawn_by})")
            out_of_date += 1
            continue
        same = dst.exists() and dst.read_bytes() == src.read_bytes()
        if same:
            print(f"up to date      {name}")
            continue
        out_of_date += 1
        if check_only:
            print(f"OUT OF DATE     {name:36s} <- results/{source}")
        else:
            shutil.copyfile(src, dst)
            print(f"copied          {name:36s} <- results/{source}")
    if check_only and out_of_date:
        sys.exit(f"{out_of_date} figure(s) differ from results/. Run without --check to copy them.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_all.py
==========
Run every Ecuador analysis script in this folder, in a sensible order, for the
requested decomposition windows (default: the three headline windows).

Run from the project root:
    python -m ecuador_price.plots.run_all
    python -m ecuador_price.plots.run_all all
    python -m ecuador_price.plots.run_all 2017_to_2023 2019_to_2020

Each script also runs standalone with the same window arguments, e.g.
    python -m ecuador_price.plots.createHeatmaps 2017_to_2023
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # the project root

SCRIPTS = [
    # summaries from the pickle
    "createHeatmaps.py",
    "selection_summary_heatmaps.py",
    "msa_normalized_selection_summary.py",
    # distributions
    "histogram_data.py",
    "specialty_histogram2.py",
    # scatters
    "simpleScatters.py",
    "diff_scatters.py",
    "specialty_scatter.py",
    "inequality_scatter.py",
    "sanity_check_scatter.py",
    # scaling / concentration / composition
    "selection_vs_population_plot.py",
    "selection_vs_population_plot_weighted.py",
    "scaling_analysis_summary.py",
    "selection_concentration_curves.py",
    "county_contribution_pie_charts.py",
    "variance_by_level.py",
    # inequality (Gini)
    "gini_inequality_scatter.py",
    "gini_scatter.py",
    "population_sorting_analysis.py",
    # accounting tables
    "population_turnover_analysis.py",
]


def main():
    args = sys.argv[1:]
    failures = []
    t0 = time.time()
    for script in SCRIPTS:
        module = "ecuador_price.plots." + script.removesuffix(".py")
        print(f"\n{'#'*70}\n# {script}\n{'#'*70}")
        t = time.time()
        result = subprocess.run([sys.executable, "-m", module] + args, cwd=ROOT)
        status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"# {script}: {status}  ({time.time() - t:.1f}s)")
        if result.returncode != 0:
            failures.append(script)

    print(f"\n{'='*70}")
    print(f"run_all finished in {time.time() - t0:.1f}s — "
          f"{len(SCRIPTS) - len(failures)}/{len(SCRIPTS)} scripts succeeded")
    if failures:
        print("Failed scripts:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()

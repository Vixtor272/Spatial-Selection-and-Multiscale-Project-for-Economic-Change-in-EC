"""
msa_normalized_selection_summary.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/msa_normalized_selection_summary.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Heatmap summarizing the mean and std dev of the Country-Normalized Selection
Intensity |Sel_level| / |AvgG_pop_country| × 100 across the canton, province
and country levels (Chicago: tr, cm, ct, st normalized by the MSA growth rate).
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

# --- Configuration ---
HIERARCHY_LEVELS = ec.HIERARCHY_LEVELS
BASE_ANALYSIS_LEVEL = ec.BASE_ANALYSIS_LEVEL  # parish-based decomposition
CUSTOM_ORANGE = ec.CUSTOM_ORANGE


def _plot_summary_heatmap(df_to_plot, title, output_path, cmap, vmin, vmax):
    """A private helper function to generate and save a summary heatmap with consistent styling."""
    n_rows, n_cols = df_to_plot.shape
    figsize = (2.0 * n_cols, n_rows)

    fig, ax = plt.subplots(figsize=figsize)

    # Format annotations with a percent sign
    annotations = np.array([["{:.1f}%".format(val) for val in row] for row in df_to_plot.values])

    sns.heatmap(
        df_to_plot,
        annot=annotations,
        fmt="",
        cmap=cmap,
        linewidths=.5,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        annot_kws={"size": 25}
    )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([str(c) for c in df_to_plot.columns], rotation=0, fontsize=12)
    ax.set_yticklabels([str(r) for r in df_to_plot.index], rotation=0, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def create_msa_normalized_summary_plot(decomposition_results, output_file):
    """
    Creates a heatmap summarizing the mean and std dev of the Country-Normalized
    Selection Intensity.
    """
    print("--- Generating Country-Normalized Selection Summary ---")

    # 1. Get the country's total growth rate from the country-level data
    try:
        df_st = decomposition_results[BASE_ANALYSIS_LEVEL]['country']
        msa_growth_rate = df_st['AvgG_pop_country'].iloc[0]
        if pd.isna(msa_growth_rate) or msa_growth_rate == 0:
            print("Error: country growth rate is zero or NaN. Cannot normalize.")
            return
        msa_growth_rate = abs(msa_growth_rate)
        print(f"Found country growth rate: {msa_growth_rate:.4f}")
    except (KeyError, IndexError):
        print("Error: Could not retrieve country growth rate from country-level data.")
        return

    # 2. Define levels and the corresponding selection columns
    levels_to_plot = ['canton', 'province', 'country']
    sel_col_map = {
        'canton': 'Sel_canton_from_parish_pop',
        'province': 'Sel_province_from_canton_pop',
        'country': 'Sel_country_from_province_pop',
    }

    summary_rows = []

    # 3. Loop through levels to calculate the metric and its stats
    for level in levels_to_plot:
        df_level = decomposition_results[BASE_ANALYSIS_LEVEL].get(level)
        if df_level is None or df_level.empty:
            continue
        df_level = df_level.copy()

        sel_col = sel_col_map.get(level)
        pop_col = f'PopInitial_{level}'

        if not all(c in df_level.columns for c in [sel_col, pop_col]):
            print(f"Warning: Missing required columns for level '{level}'. Skipping.")
            continue

        # Country-Normalized Selection Intensity for each unit, in percent
        df_level['msa_norm_sel'] = (df_level[sel_col].abs() / msa_growth_rate) * 100

        # --- Calculate both weighted and unweighted statistics ---

        unweighted_mean = df_level['msa_norm_sel'].mean()
        unweighted_std = df_level['msa_norm_sel'].std()

        weights = df_level[pop_col]
        if weights.isnull().all() or weights.sum() == 0:
            weighted_mean = unweighted_mean
            weighted_std = unweighted_std
        elif len(df_level) == 1:
            weighted_mean = df_level['msa_norm_sel'].iloc[0]
            weighted_std = 0.0
        else:
            weighted_mean = np.average(df_level['msa_norm_sel'], weights=weights)
            variance = np.average((df_level['msa_norm_sel'] - weighted_mean)**2, weights=weights)
            weighted_std = np.sqrt(variance)

        summary_rows.append({
            'Level': level.capitalize(),
            'Mean (Weighted)': weighted_mean,
            'Std (Weighted)': weighted_std,
            'Mean (Unweighted)': unweighted_mean,
            'Std (Unweighted)': unweighted_std
        })

    if not summary_rows:
        print("Could not generate summary data. Aborting.")
        return

    # 4. Create the final DataFrame for the heatmap
    df_plot = pd.DataFrame(summary_rows).set_index('Level').T

    # 5. Plotting, using the same style as the MSS variation summary
    custom_cmap = LinearSegmentedColormap.from_list('custom_white_orange', ['white', CUSTOM_ORANGE])

    _plot_summary_heatmap(
        df_to_plot=df_plot,
        title='Country-Normalized Selection Intensity (%)',
        output_path=output_file,
        cmap=custom_cmap,
        vmin=0,
        vmax=None
    )

    print(f"Successfully saved Country-Normalized Selection summary to {output_file}")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        output_dir = os.path.join(base_output_dir, "heatmaps_local_dominance/country_summary")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'msa_normalized_selection_summary.pdf')

        decomposition_data = ec.load_pickle(input_dir)
        if decomposition_data:
            create_msa_normalized_summary_plot(decomposition_data, output_file)


if __name__ == "__main__":
    main()

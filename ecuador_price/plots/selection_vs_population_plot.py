"""
selection_vs_population_plot.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/selection_vs_population_plot.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Chicago aggregated the tract- and community-level absolute population
selection to the 7 counties and regressed log(sum |Sel|) on log(county
population).  Here the canton- and province-level absolute selection is
aggregated to the 24 provinces:

    orange: canton-level  |Sel_canton_from_parish_pop| summed by province
    purple: province-level |Sel_province_from_canton_pop| (one point each)

against the log of the final firm stock of the province.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

# Custom Colors
CUSTOM_PURPLE = ec.CUSTOM_PURPLE
CUSTOM_ORANGE = ec.CUSTOM_ORANGE


def process_level_data(input_dir, level_suffix, county_col='ParentProvince'):
    """
    Loads and aggregates data for a specific geographic level to provinces.
    """
    print(f"Reading and processing parish_{level_suffix}_exported_terms.csv...")
    df = ec.load_terms(input_dir, level_suffix)
    if df is None:
        print(f"Error: Input file not found for level {level_suffix}")
        return None

    pop_col = f'PopFinal_{level_suffix}'
    if level_suffix == 'canton':
        selection_col = 'Sel_canton_from_parish_pop'
        group_col = county_col
    elif level_suffix == 'province':
        selection_col = 'Sel_province_from_canton_pop'
        group_col = 'UnitName'      # each province is its own group
    else:
        print(f"Unknown level suffix: {level_suffix}")
        return None

    if not all(c in df.columns for c in [pop_col, selection_col, group_col]):
        print(f"Error: Required columns not found for level {level_suffix}.")
        return None

    # Aggregate to province level
    county_pop = df.groupby(group_col)[pop_col].sum()
    county_abs_selection = df.groupby(group_col)[selection_col].apply(lambda x: x.abs().sum())

    df_county = pd.DataFrame({
        'total_pop': county_pop,
        'total_abs_selection': county_abs_selection
    }).reset_index()

    # Filter for log transformation
    df_filtered = df_county[(df_county['total_pop'] > 0) & (df_county['total_abs_selection'] > 0)].copy()
    if df_filtered.empty:
        return None

    df_filtered['log_pop'] = np.log(df_filtered['total_pop'])
    df_filtered['log_selection'] = np.log(df_filtered['total_abs_selection'])
    df_filtered['CountyName'] = df_filtered[group_col].map(ec.PROVINCE_NAME)

    return df_filtered


def create_plot(input_dir, base_output_dir):
    """
    Generates a scatter plot comparing canton and province level selection vs. population.
    """
    output_dir = os.path.join(base_output_dir, 'specialty_scatter_plots')
    output_file = os.path.join(output_dir, 'log_abs_selection_vs_log_population_comparison.pdf')
    os.makedirs(output_dir, exist_ok=True)

    df_tract = process_level_data(input_dir, 'canton')
    df_community = process_level_data(input_dir, 'province')

    if df_tract is None or df_community is None:
        print("Could not process one or both data levels. Aborting plot generation.")
        return

    # --- Calculate Slopes for Legend ---
    slope_tract, _ = np.polyfit(df_tract['log_pop'], df_tract['log_selection'], 1)
    slope_community, _ = np.polyfit(df_community['log_pop'], df_community['log_selection'], 1)

    tract_label = f'can (slope={slope_tract:.2f})'
    community_label = f'prov (slope={slope_community:.2f})'

    # --- Plotting ---
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Plot Canton Data (Chicago: Tract, orange)
    sns.regplot(
        x='log_pop', y='log_selection', data=df_tract, ax=ax,
        scatter_kws={'alpha': 0.5, 's': 100, 'color': CUSTOM_ORANGE},
        line_kws={'color': CUSTOM_ORANGE, 'linestyle': '-'},
        label=tract_label
    )

    # Plot Province Data (Chicago: Community, purple)
    sns.regplot(
        x='log_pop', y='log_selection', data=df_community, ax=ax,
        scatter_kws={'alpha': 0.5, 's': 100, 'color': CUSTOM_PURPLE},
        line_kws={'color': CUSTOM_PURPLE, 'linestyle': '-'},
        label=community_label
    )

    # --- Aesthetics ---
    ax.set_xlabel(r'$\ln$ (province firm stock)', fontsize=12)
    ax.set_ylabel(r'$\ln \sum |S|$', fontsize=12)

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.legend(fontsize=12)

    # --- Save ---
    print(f"Saving plot to {output_file}...")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print("Plotting complete.")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        create_plot(input_dir, base_output_dir)


if __name__ == '__main__':
    main()

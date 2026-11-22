"""
scaling_analysis_summary.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/scaling_analysis_summary.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Chicago aggregated tract-level absolute selection to counties and compared
three scaling relations against county population.  Here canton-level
absolute selection is aggregated to the 24 provinces:

    Case 1: Unweighted Sum            sum |Sel_canton_from_parish_pop|
    Case 2: Population-Weighted Sum   sum |Sel| * firms
    Case 3: Population-Weighted Avg   Case 2 / total firms

each regressed (log-log) on the province's final firm stock.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

CUSTOM_ORANGE = ec.CUSTOM_ORANGE


def process_tract_data(input_dir):
    """
    Loads canton data (Chicago: tract) and aggregates it to the province level
    using all three calculation methods.
    """
    print(f"Reading and processing parish_canton_exported_terms.csv from {input_dir}...")
    df = ec.load_terms(input_dir, 'canton')
    if df is None:
        print("Error: Input file not found")
        return None

    pop_col = 'PopFinal_canton'
    selection_col = 'Sel_canton_from_parish_pop'
    county_col = 'ParentProvince'

    if not all(c in df.columns for c in [pop_col, selection_col, county_col]):
        print("Error: Required columns not found.")
        return None

    # --- Perform all three aggregations ---
    county_pop = df.groupby(county_col)[pop_col].sum()

    # Case 1: Unweighted Sum
    unweighted_sum = df.groupby(county_col)[selection_col].apply(lambda x: x.abs().sum())

    # Case 2: Population-Weighted Sum
    df['weighted_abs_selection'] = df[selection_col].abs() * df[pop_col]
    weighted_sum = df.groupby(county_col)['weighted_abs_selection'].sum()

    # Case 3: Population-Weighted Average (Normalized)
    weighted_avg = weighted_sum / county_pop

    # --- Combine into a single DataFrame ---
    df_county = pd.DataFrame({
        'total_pop': county_pop,
        'unweighted_sum': unweighted_sum,
        'weighted_sum': weighted_sum,
        'weighted_avg': weighted_avg
    }).reset_index()

    # --- Log Transformations ---
    df_county = df_county[
        (df_county['total_pop'] > 0) &
        (df_county['unweighted_sum'] > 0) &
        (df_county['weighted_sum'] > 0) &
        (df_county['weighted_avg'] > 0)
    ].copy()

    df_county['log_pop'] = np.log(df_county['total_pop'])
    df_county['log_unweighted_sum'] = np.log(df_county['unweighted_sum'])
    df_county['log_weighted_sum'] = np.log(df_county['weighted_sum'])
    df_county['log_weighted_avg'] = np.log(df_county['weighted_avg'])

    return df_county


def create_summary_plot(df, base_output_dir):
    """
    Generates the 3-panel summary plot of the scaling analyses.
    """
    if df is None or df.empty:
        print("No data to plot.")
        return

    output_dir = os.path.join(base_output_dir, 'specialty_scatter_plots')
    output_file = os.path.join(output_dir, 'scaling_analysis_summary_canton.pdf')
    os.makedirs(output_dir, exist_ok=True)

    # --- Setup Figure ---
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.5))
    plot_configs = [
        {'y_col': 'log_unweighted_sum', 'title': 'Case 1: Unweighted Sum',
         'ylabel': r'$\ln \sum |S|$', 'ax': axes[0]},
        {'y_col': 'log_weighted_sum', 'title': 'Case 2: Population-Weighted Sum',
         'ylabel': r'$\ln \sum |S| \cdot n$', 'ax': axes[1]},
        {'y_col': 'log_weighted_avg', 'title': 'Case 3: Population-Weighted Average',
         'ylabel': r'$\ln\, (\sum |S| \cdot n \,/\, \sum n)$', 'ax': axes[2]}
    ]

    # --- Create Each Plot ---
    for config in plot_configs:
        y_col = config['y_col']
        ax = config['ax']

        slope, _ = np.polyfit(df['log_pop'], df[y_col], 1)
        label = f'Cantons (β={slope:.2f})'

        sns.regplot(
            x='log_pop', y=y_col, data=df, ax=ax,
            scatter_kws={'alpha': 0.7, 's': 80, 'color': CUSTOM_ORANGE},
            line_kws={'color': CUSTOM_ORANGE, 'linestyle': '-', 'linewidth': 1}
        )

        ax.legend([label], fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_title(config['title'], fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel(config['ylabel'], fontsize=10)
    axes[2].set_xlabel(r'$\ln$ (province firm stock)', fontsize=10)

    # (Chicago fixed axes[2] to [-7, -3] for its data range; left data-driven here.)
    plt.tight_layout(rect=[0, 0, 1, 1])

    # --- Save ---
    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plotting complete.")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        county_data = process_tract_data(input_dir)
        create_summary_plot(county_data, base_output_dir)


if __name__ == '__main__':
    main()

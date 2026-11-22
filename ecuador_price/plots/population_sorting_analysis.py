"""
population_sorting_analysis.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/population_sorting_analysis.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
The Chicago script mixed two data sources:
  (a) US Census API pulls (median gross rent B25064, binned income B19001 for
      a Theil decomposition) — there is no Ecuadorian analog in this project,
      so the rent and Theil sections are NOT ported;
  (b) the decomposition CSVs themselves, used for the "selection vs change in
      Gini" scatters — fully ported here at the canton level (Chicago: tract).

Plots (canton level, Gini across parishes within each canton):
    can_pop_ns_vs_gini_change.pdf                Population Selection NS (country-
                                                 normalized) vs Gini change
    can_inc_ns_vs_gini_change_sanity_check.pdf   Income Selection NS vs Gini change

Point size = initial firm stock; color bifurcated at the median initial income
(orange above / purple below), as in the source.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
import sys

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

warnings.filterwarnings('ignore', category=FutureWarning)


# --- GINI CALCULATION FUNCTIONS (shared with gini_scatter.py) ---

def calculate_gini_coefficient(incomes, weights=None):
    """
    Calculate Gini coefficient for a distribution of incomes.
    """
    if weights is None:
        weights = np.ones_like(incomes)

    valid_mask = ~(np.isnan(incomes) | np.isnan(weights))
    incomes = incomes[valid_mask]
    weights = weights[valid_mask]

    if len(incomes) < 2:
        return np.nan

    sorted_indices = np.argsort(incomes)
    sorted_incomes = incomes[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cum_weights = np.cumsum(sorted_weights)
    cum_income_weights = np.cumsum(sorted_incomes * sorted_weights)

    total_weight = cum_weights[-1]
    total_income_weight = cum_income_weights[-1]

    if total_weight == 0 or total_income_weight == 0:
        return np.nan

    cum_pop_share = cum_weights / total_weight
    cum_income_share = cum_income_weights / total_income_weight

    x = np.concatenate([[0], cum_pop_share])
    y = np.concatenate([[0], cum_income_share])

    area_under_lorenz = np.trapezoid(y, x)
    gini = 1 - 2 * area_under_lorenz
    return gini


def calculate_gini_changes(income_data, group_by_col):
    """
    Calculate Gini coefficient change for each group.

    The Chicago source used the raw log incomes here; for consistency with the
    other Gini scripts the per-firm mean sales exp(LogAvgInc)/firms are used,
    with initial weights for the initial Gini and final weights for the final
    (this script's own convention).
    """
    print(f"Calculating Gini coefficients by {group_by_col}...")
    results = []

    df = income_data.copy()
    df['initial_income'] = np.exp(df['LogAvgIncInitial_parish']) / df['PopInitial_parish']
    df['final_income'] = np.exp(df['LogAvgIncFinal_parish']) / df['PopFinal_parish']

    for group_name, group in df.groupby(group_by_col):
        if len(group) < 2:
            continue

        initial_gini = calculate_gini_coefficient(
            group['initial_income'].values,
            group['PopInitial_parish'].values
        )

        final_gini = calculate_gini_coefficient(
            group['final_income'].values,
            group['PopFinal_parish'].values
        )

        gini_change = final_gini - initial_gini if pd.notna(final_gini) and pd.notna(initial_gini) else np.nan

        results.append({
            group_by_col: group_name,
            'gini_change': gini_change
        })

    result_df = pd.DataFrame(results)
    print(f"  Calculated Gini changes for {len(result_df)} groups.")
    return result_df


# --- PLOTTING FUNCTION ---

def create_scatter_plot(data, x_col, y_col, size_col, color_col, title, xlabel, ylabel,
                        filename, output_dir):
    """
    Creates and saves a styled scatter plot with regression analysis.
    """
    clean_data = data[[x_col, y_col, size_col, color_col]].dropna()

    if len(clean_data) < 20:
        print(f"Skipping plot '{title}': Not enough valid data points ({len(clean_data)}).")
        return

    # --- Prepare point sizes ---
    pop = clean_data[size_col]
    s_min, s_max = 15, 200
    sizes = s_min + ((pop - pop.min()) / (pop.max() - pop.min()) * (s_max - s_min)) \
        if pop.min() != pop.max() else s_min

    # --- Prepare point colors ---
    # Bifurcate the color based on whether the initial income is above or below the median
    median_color_val = clean_data[color_col].median()
    colors = np.where(clean_data[color_col] > median_color_val, ec.CUSTOM_ORANGE, ec.CUSTOM_PURPLE)

    # --- Create plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.scatter(clean_data[x_col], clean_data[y_col], s=sizes, c=colors, alpha=0.6,
               edgecolors='white', linewidth=0.5)

    # --- Regression Analysis ---
    slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[x_col], clean_data[y_col])
    r_squared = r_value**2

    x_range = np.linspace(clean_data[x_col].min(), clean_data[x_col].max(), 100)
    ax.plot(x_range, intercept + slope * x_range, color='black', linestyle='--',
            linewidth=1.5, alpha=0.8)

    # --- Aesthetics ---
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    stats_text = f'$R^2 = {r_squared:.3f}$\np-value = {p_value:.3f}\nSlope = {slope:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    ax.axhline(0, color='grey', linestyle=':', linewidth=1)
    ax.axvline(0, color='grey', linestyle=':', linewidth=1)

    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to {full_path}")


# --- MAIN EXECUTION ---

def run_analysis(input_dir, base_output_dir):
    """
    Main function: load decomposition + Gini data and generate plots.
    (The Census-API rent fetch and Theil decomposition of the Chicago source
    have no Ecuador data source and are skipped.)
    """
    output_dir = os.path.join(base_output_dir, 'inequality_scatter_plots')
    os.makedirs(output_dir, exist_ok=True)

    # Only the canton level (Chicago ran the tract level only)
    for level, level_short in [('canton', 'can')]:
        print(f"\n{'='*60}\nProcessing {level.upper()} level data\n{'='*60}")

        # 1. Load local decomposition data
        print(f"Loading local data for {level}...")
        df_local = ec.load_terms(input_dir, level)
        if df_local is None:
            print("  Could not find local data file. Skipping.")
            continue
        df_local = ec.add_country_ns(df_local, input_dir)

        # --- GINI CALCULATION STEP ---
        gini_col = 'gini_change'
        print("Loading parish data to calculate Gini changes for cantons...")
        df_bg = ec.load_terms(input_dir, 'parish')
        if df_bg is None:
            print("  Could not find parish data file. Skipping Gini plot.")
            gini_col = None
        else:
            df_gini_change = calculate_gini_changes(df_bg, 'ParentCanton')
            df_local = pd.merge(df_local, df_gini_change, left_on='UnitName',
                                right_on='ParentCanton', how='left')

        # 2. Identify columns for plotting
        # For selection, the country-normalized NS of the level's own selection effect
        pop_ns_col = 'Sel_canton_from_parish_pop_ns_country'

        size_col = f'PopInitial_{level}'
        color_col = f'LogAvgIncInitial_{level}'

        # Plot 1: Population NS vs. Change in Gini
        if gini_col and gini_col in df_local.columns and pop_ns_col in df_local.columns:
            create_scatter_plot(
                data=df_local,
                x_col=pop_ns_col,
                y_col=gini_col,
                size_col=size_col,
                color_col=color_col,
                title=f'{level.title()} Level: Population Selection vs. Change in Gini',
                xlabel='Population Selection NS (Country-Normalized)',
                ylabel='Change in Gini Coefficient',
                filename=f'{level_short}_pop_ns_vs_gini_change.pdf',
                output_dir=output_dir
            )

            # --- Sanity Check Plot ---
            # Plot 2: Income NS vs. Change in Gini
            inc_ns_col = 'Sel_canton_from_parish_inc_ns_country'
            if inc_ns_col in df_local.columns:
                create_scatter_plot(
                    data=df_local,
                    x_col=inc_ns_col,
                    y_col=gini_col,
                    size_col=size_col,
                    color_col=color_col,
                    title=f'{level.title()} Level: Income Selection vs. Change in Gini (Sanity Check)',
                    xlabel='Income Selection NS (Country-Normalized)',
                    ylabel='Change in Gini Coefficient',
                    filename=f'{level_short}_inc_ns_vs_gini_change_sanity_check.pdf',
                    output_dir=output_dir
                )
            else:
                print(f"  Skipping sanity check plot: Missing column '{inc_ns_col}'")

            # Plot 3: Density NS vs. Change in Gini (den path: trait = firm count)
            den_ns_col = 'Sel_canton_from_parish_den_ns_country'
            if den_ns_col in df_local.columns:
                create_scatter_plot(
                    data=df_local,
                    x_col=den_ns_col,
                    y_col=gini_col,
                    size_col=size_col,
                    color_col=color_col,
                    title=f'{level.title()} Level: Density Selection vs. Change in Gini',
                    xlabel='Density Selection NS (Country-Normalized)',
                    ylabel='Change in Gini Coefficient',
                    filename=f'{level_short}_den_ns_vs_gini_change.pdf',
                    output_dir=output_dir
                )
            else:
                print(f"  Skipping density plot: Missing column '{den_ns_col}'")
        else:
            print("  Skipping Gini plots due to missing data or calculation failure.")

    print("\nNote: the Chicago rent-change and Theil-index plots need the US Census API")
    print("and were not ported (no equivalent Ecuadorian data source in this project).")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == "__main__":
    main()

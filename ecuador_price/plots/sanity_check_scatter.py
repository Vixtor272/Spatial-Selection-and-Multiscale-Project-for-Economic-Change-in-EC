"""
sanity_check_scatter.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/sanity_check_scatter.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Plots the decomposition identity LHS (AvgG_pop / AvgG_inc of each unit)
against the RHS (Transmitted Growth from the parish base + Cumulative
Selection).  Points should sit on the y=x diagonal if the multi-level Price
decomposition is exact.

Chicago levels cm/tr -> Ecuador province/canton.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

BASE = ec.BASE_ANALYSIS_LEVEL  # 'parish'


def load_data(input_dir):
    """Load the decomposition results from CSV files"""
    try:
        cm_data = ec.load_terms(input_dir, 'province')
        cm_data['Agg'] = 'province'
        print(f"Successfully loaded province data with shape: {cm_data.shape}")

        tr_data = ec.load_terms(input_dir, 'canton')
        tr_data['Agg'] = 'canton'
        print(f"Successfully loaded canton data with shape: {tr_data.shape}")

        df = pd.concat([cm_data, tr_data], ignore_index=True)
        print(f"Combined data shape: {df.shape}")
        print(f"Available aggregation levels: {df['Agg'].unique()}")

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def prepare_sanity_check_data(df, level_name):
    """Prepare LHS and RHS variables for sanity check plots"""
    print(f"\n=== Preparing sanity check data for {level_name.upper()} level ===")

    level_clean = level_name

    # Find LHS variables (exactly what the sanity checks validate)
    lhs_pop_col = f'AvgG_pop_{level_clean}'
    lhs_inc_col = f'AvgG_inc_{level_clean}'

    if lhs_pop_col in df.columns:
        df['lhs_population'] = df[lhs_pop_col]
        print(f"Using {lhs_pop_col} for LHS population decomposition")
    else:
        print(f"Warning: Could not find {lhs_pop_col}")
        df['lhs_population'] = np.nan

    if lhs_inc_col in df.columns:
        df['lhs_income'] = df[lhs_inc_col]
        print(f"Using {lhs_inc_col} for LHS income decomposition")
    else:
        print(f"Warning: Could not find {lhs_inc_col}")
        df['lhs_income'] = np.nan

    lhs_den_col = f'AvgG_den_{level_clean}'
    if lhs_den_col in df.columns:
        df['lhs_density'] = df[lhs_den_col]
        print(f"Using {lhs_den_col} for LHS density decomposition")
    else:
        print(f"Warning: Could not find {lhs_den_col}")
        df['lhs_density'] = np.nan

    # Find RHS components
    # 1. Transmitted Growth from base level (parish)
    transmitted_pop_col = f'Transmitted_AvgG_{BASE}_to_{level_clean}_pop'
    transmitted_inc_col = f'Transmitted_AvgG_{BASE}_to_{level_clean}_inc'

    if transmitted_pop_col in df.columns:
        df['transmitted_pop_growth'] = df[transmitted_pop_col]
        print(f"Found transmitted growth (pop): {transmitted_pop_col}")
    else:
        print(f"Warning: Could not find {transmitted_pop_col}")
        df['transmitted_pop_growth'] = np.nan

    if transmitted_inc_col in df.columns:
        df['transmitted_inc_growth'] = df[transmitted_inc_col]
        print(f"Found transmitted growth (inc): {transmitted_inc_col}")
    else:
        print(f"Warning: Could not find {transmitted_inc_col}")
        df['transmitted_inc_growth'] = np.nan

    transmitted_den_col = f'Transmitted_AvgG_{BASE}_to_{level_clean}_den'
    if transmitted_den_col in df.columns:
        df['transmitted_den_growth'] = df[transmitted_den_col]
        print(f"Found transmitted growth (den): {transmitted_den_col}")
    else:
        print(f"Warning: Could not find {transmitted_den_col}")
        df['transmitted_den_growth'] = np.nan

    # 2. Cumulative Selection (direct + transmitted selection)
    cumulative_sel_pop_col = f'Cumulative_Sel_pop_{level_clean}'
    cumulative_sel_inc_col = f'Cumulative_Sel_inc_{level_clean}'

    if cumulative_sel_pop_col in df.columns:
        df['cumulative_sel_pop'] = df[cumulative_sel_pop_col]
        print(f"Found cumulative selection (pop): {cumulative_sel_pop_col}")
    else:
        print(f"Warning: Could not find {cumulative_sel_pop_col}")
        df['cumulative_sel_pop'] = np.nan

    if cumulative_sel_inc_col in df.columns:
        df['cumulative_sel_inc'] = df[cumulative_sel_inc_col]
        print(f"Found cumulative selection (inc): {cumulative_sel_inc_col}")
    else:
        print(f"Warning: Could not find {cumulative_sel_inc_col}")
        df['cumulative_sel_inc'] = np.nan

    cumulative_sel_den_col = f'Cumulative_Sel_den_{level_clean}'
    if cumulative_sel_den_col in df.columns:
        df['cumulative_sel_den'] = df[cumulative_sel_den_col]
        print(f"Found cumulative selection (den): {cumulative_sel_den_col}")
    else:
        print(f"Warning: Could not find {cumulative_sel_den_col}")
        df['cumulative_sel_den'] = np.nan

    # Calculate complete RHS totals (matching the multi-level sanity check structure)
    df['rhs_pop_total'] = df['transmitted_pop_growth'] + df['cumulative_sel_pop']
    df['rhs_inc_total'] = df['transmitted_inc_growth'] + df['cumulative_sel_inc']
    df['rhs_den_total'] = df['transmitted_den_growth'] + df['cumulative_sel_den']

    # Calculate differences for diagnostics
    df['sanity_check_pop'] = df['lhs_population'] - df['rhs_pop_total']
    df['sanity_check_inc'] = df['lhs_income'] - df['rhs_inc_total']
    df['sanity_check_den'] = df['lhs_density'] - df['rhs_den_total']

    print(f"=== Summary for {level_name.upper()} ===")
    print("Population RHS = Transmitted Growth + Cumulative Selection")
    print("Income RHS = Transmitted Growth + Cumulative Selection")

    valid_pop = ~(pd.isna(df['lhs_population']) | pd.isna(df['rhs_pop_total']))
    valid_inc = ~(pd.isna(df['lhs_income']) | pd.isna(df['rhs_inc_total']))

    if valid_pop.any():
        sanity_pop_rmse = np.sqrt(np.mean(df.loc[valid_pop, 'sanity_check_pop']**2))
        sanity_pop_max_abs = np.abs(df.loc[valid_pop, 'sanity_check_pop']).max()
        print(f"Population sanity check - RMSE: {sanity_pop_rmse:.6f}, Max |diff|: {sanity_pop_max_abs:.6f}")

    if valid_inc.any():
        sanity_inc_rmse = np.sqrt(np.mean(df.loc[valid_inc, 'sanity_check_inc']**2))
        sanity_inc_max_abs = np.abs(df.loc[valid_inc, 'sanity_check_inc']).max()
        print(f"Income sanity check - RMSE: {sanity_inc_rmse:.6f}, Max |diff|: {sanity_inc_max_abs:.6f}")

    return df


def add_regression_line_and_stats(ax, x, y, color='red'):
    """Add regression line and statistics to plot"""
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if valid_mask.sum() < 2:
        ax.text(0.05, 0.95, 'Insufficient data for regression',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        return

    x_clean = x[valid_mask]
    y_clean = y[valid_mask]

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

    x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_pred = slope * x_range + intercept
    ax.plot(x_range, y_pred, color=color, linewidth=2, alpha=0.8, label='Regression')

    stats_text = f'y = {slope:.4f}x + {intercept:.4f}\n'
    stats_text += f'R² = {r_value**2:.4f}\n'
    stats_text += f'p = {p_value:.4e}\n'
    stats_text += f'n = {len(x_clean)}'

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def create_sanity_check_plot(data, x_var, y_var, title, filename, figsize=(10, 8)):
    """Create a sanity check scatter plot with y=x diagonal reference"""

    mask = ~(pd.isna(data[x_var]) | pd.isna(data[y_var]))
    plot_data = data[mask]

    if len(plot_data) == 0:
        print(f"Warning: No valid data for plot {title}")
        return

    plt.figure(figsize=figsize)

    plt.scatter(plot_data[x_var], plot_data[y_var], alpha=0.6, s=30, color='blue')

    # Add y=x diagonal reference line (perfect sanity check)
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)

    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2, label='Perfect Match (y = x)')

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    add_regression_line_and_stats(plt.gca(), plot_data[x_var].values, plot_data[y_var].values)

    plt.xlabel('LHS (Left-Hand Side of Decomposition)')
    plt.ylabel('RHS (Transmitted Growth + Cumulative Selection)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")


def create_all_sanity_check_plots(data, base_output_dir):
    """Create sanity check plots for both province and canton levels"""

    levels = ['province', 'canton']

    for level in levels:
        level_suffix = '_prov' if level == 'province' else '_can'
        level_data = data[data['Agg'] == level].copy()

        if len(level_data) == 0:
            print(f"No data found for level {level}")
            continue

        print(f"\n=== Processing {level.upper()} level data ===")
        level_data = prepare_sanity_check_data(level_data, level)

        output_dir = os.path.join(base_output_dir, f'sanity_check_{level}')

        # 1. Population Decomposition Sanity Check
        create_sanity_check_plot(
            level_data,
            'lhs_population',
            'rhs_pop_total',
            f'{level.upper()} Level: Population Decomposition Sanity Check\nLHS vs RHS (Transmitted Growth + Cumulative Selection)',
            os.path.join(output_dir, f'population_sanity_check{level_suffix}.png')
        )

        # 2. Income Decomposition Sanity Check
        create_sanity_check_plot(
            level_data,
            'lhs_income',
            'rhs_inc_total',
            f'{level.upper()} Level: Income Decomposition Sanity Check\nLHS vs RHS (Transmitted Growth + Cumulative Selection)',
            os.path.join(output_dir, f'income_sanity_check{level_suffix}.png')
        )

        # 3. Density Decomposition Sanity Check (den path: trait = firm count)
        create_sanity_check_plot(
            level_data,
            'lhs_density',
            'rhs_den_total',
            f'{level.upper()} Level: Density Decomposition Sanity Check\nLHS vs RHS (Transmitted Growth + Cumulative Selection)',
            os.path.join(output_dir, f'density_sanity_check{level_suffix}.png')
        )

    return data


def print_summary_statistics(data):
    """Print summary statistics for sanity check quality"""
    print("\n=== Sanity Check Quality Summary ===")

    for level in ['province', 'canton']:
        level_data = data[data['Agg'] == level]
        if len(level_data) == 0:
            continue

        print(f"\n{level.upper()} Level:")

        if 'sanity_check_pop' in level_data.columns:
            pop_check = level_data['sanity_check_pop'].dropna()
            if len(pop_check) > 0:
                print("  Population Sanity Check (LHS - RHS):")
                print(f"    Mean: {pop_check.mean():.6f}")
                print(f"    Std:  {pop_check.std():.6f}")
                print(f"    RMSE: {np.sqrt(np.mean(pop_check**2)):.6f}")
                print(f"    Max |error|: {np.abs(pop_check).max():.6f}")
                print(f"    N: {len(pop_check)}")

        if 'sanity_check_inc' in level_data.columns:
            inc_check = level_data['sanity_check_inc'].dropna()
            if len(inc_check) > 0:
                print("  Income Sanity Check (LHS - RHS):")
                print(f"    Mean: {inc_check.mean():.6f}")
                print(f"    Std:  {inc_check.std():.6f}")
                print(f"    RMSE: {np.sqrt(np.mean(inc_check**2)):.6f}")
                print(f"    Max |error|: {np.abs(inc_check).max():.6f}")
                print(f"    N: {len(inc_check)}")


def run_analysis(input_dir, base_output_dir):
    """Main function"""
    print("=== Decomposition Sanity Check Analysis ===")
    print("Plotting LHS vs RHS (Transmitted Growth + Cumulative Selection)")

    data = load_data(input_dir)
    if data is None:
        return

    print(f"\nDataset shape: {data.shape}")
    print(f"Available aggregation levels: {data['Agg'].unique()}")

    data = create_all_sanity_check_plots(data, base_output_dir)

    print("\n=== Analysis Complete ===")
    print(f"Check the '{base_output_dir}/sanity_check_province' and '{base_output_dir}/sanity_check_canton' directories for output plots.")
    print("\nThese plots should show points very close to the y=x diagonal if the decomposition is working correctly.")
    print("The regression line slope should be very close to 1.0 and R² should be very close to 1.0.")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == "__main__":
    main()

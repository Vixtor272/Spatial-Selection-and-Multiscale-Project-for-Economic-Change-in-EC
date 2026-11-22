#!/usr/bin/env python3
"""
diff_scatters.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/diff_scatters.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Generates scatter plots comparing empirical growth rates and initial income
against differences between various growth rate measures.

Chicago levels Community/Tract -> Ecuador Province/Canton.  The trait is
log real total sales; "population" growth is firm-count growth.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
import sys

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

# (level_name, level_suffix) pairs: Chicago ('Community','cm')/('Tract','tr')
LEVELS = [('Province', 'province'), ('Canton', 'canton')]


def load_data(input_dir):
    """Load and prepare the data from CSV files."""
    print(f"Loading data from {input_dir}...")

    cm_data = ec.load_terms(input_dir, 'province')
    if cm_data is None:
        print(f"Error: parish_province_exported_terms.csv not found in {input_dir}")
        return None, None
    print(f"Loaded province data: {cm_data.shape[0]} rows, {cm_data.shape[1]} columns")

    tr_data = ec.load_terms(input_dir, 'canton')
    if tr_data is None:
        print(f"Warning: parish_canton_exported_terms.csv not found in {input_dir}, using only province data")
    else:
        print(f"Loaded canton data: {tr_data.shape[0]} rows, {tr_data.shape[1]} columns")

    return cm_data, tr_data


def prepare_variables(data, level_name, level_suffix):
    """Extract and calculate the required variables from the data."""
    print(f"Preparing variables for {level_name} level...")

    df = data.copy()

    # Extract empirical growth rate from pre-calculated column
    if f'AvgG_emp_{level_suffix}' in df.columns:
        df['empirical_growth_rate'] = df[f'AvgG_emp_{level_suffix}']
    else:
        emp_cols = [col for col in df.columns if 'AvgG_emp' in col]
        if emp_cols:
            df['empirical_growth_rate'] = df[emp_cols[0]]
        else:
            print(f"Warning: Could not find empirical growth rate column (AvgG_emp_*) for {level_name}")
            df['empirical_growth_rate'] = np.nan

    # Population averaged growth rate
    if f'AvgG_pop_{level_suffix}' in df.columns:
        df['pop_averaged_growth_rate'] = df[f'AvgG_pop_{level_suffix}']
    else:
        pop_avg_cols = [col for col in df.columns if 'AvgG_pop' in col and 'Rel' not in col]
        if pop_avg_cols:
            df['pop_averaged_growth_rate'] = df[pop_avg_cols[0]]
        else:
            print(f"Warning: Could not find population averaged growth rate for {level_name}")
            df['pop_averaged_growth_rate'] = np.nan

    # Income averaged growth rate
    if f'AvgG_inc_{level_suffix}' in df.columns:
        df['inc_averaged_growth_rate'] = df[f'AvgG_inc_{level_suffix}']
    else:
        inc_avg_cols = [col for col in df.columns if 'AvgG_inc' in col and 'Rel' not in col]
        if inc_avg_cols:
            df['inc_averaged_growth_rate'] = df[inc_avg_cols[0]]
        else:
            print(f"Warning: Could not find income averaged growth rate for {level_name}")
            df['inc_averaged_growth_rate'] = np.nan

    # Initial income (using log average income initial)
    if f'LogAvgIncInitial_{level_suffix}' in df.columns:
        df['initial_income'] = df[f'LogAvgIncInitial_{level_suffix}']
    else:
        initial_inc_cols = [col for col in df.columns if 'LogAvgInc' in col and 'Initial' in col]
        if initial_inc_cols:
            df['initial_income'] = df[initial_inc_cols[0]]
        else:
            print(f"Warning: Could not find initial income for {level_name}")
            df['initial_income'] = np.nan

    # Calculate difference variables
    df['pop_avg_minus_empirical'] = df['pop_averaged_growth_rate'] - df['empirical_growth_rate']
    df['inc_avg_minus_empirical'] = df['inc_averaged_growth_rate'] - df['empirical_growth_rate']
    df['pop_avg_minus_inc_avg'] = df['pop_averaged_growth_rate'] - df['inc_averaged_growth_rate']

    # Add level identifier
    df['level'] = level_name

    key_vars = ['empirical_growth_rate', 'pop_averaged_growth_rate', 'inc_averaged_growth_rate',
                'initial_income', 'level',
                'pop_avg_minus_empirical', 'inc_avg_minus_empirical', 'pop_avg_minus_inc_avg']

    if 'UnitName' in df.columns:
        key_vars.append('UnitName')

    return df[key_vars]


def add_regression_line_and_stats(ax, x, y, color='red'):
    """Add regression line and statistics to a plot."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

    line_x = np.array([x_clean.min(), x_clean.max()])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color=color, linestyle='--', alpha=0.8, linewidth=2)

    r_squared = r_value ** 2
    stats_text = f'Slope = {slope:.4f}\nIntercept = {intercept:.4f}\nR² = {r_squared:.3f}\np = {p_value:.3f}\nn = {len(x_clean)}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def create_scatter_plot(data, x_var, y_var, title, filename, base_output_dir, figsize=(10, 8)):
    """Create a scatter plot with linear fit and save it."""
    print(f"Creating plot: {title}")

    mask = (
        data[x_var].notna() &
        data[y_var].notna() &
        np.isfinite(data[x_var]) &
        np.isfinite(data[y_var])
    )

    plot_data = data[mask].copy()

    if len(plot_data) < 2:
        print(f"  Skipping {title} - insufficient data points ({len(plot_data)})")
        return

    # Determine aggregation level from data
    if 'level' in plot_data.columns:
        level = plot_data['level'].iloc[0]
        agg_level = 'prov' if 'Province' in level else 'can'
    else:
        agg_level = 'prov'  # default

    # Define mathematical notation mapping (identical to the Chicago labels)
    math_labels = {
        'empirical_growth_rate': rf'$\gamma_{{{agg_level}}}$',
        'pop_averaged_growth_rate': rf'$\gamma_{{{agg_level}}}^p$',
        'inc_averaged_growth_rate': rf'$\gamma_{{{agg_level}}}^i$',
        'population_growth_rate': r'$\gamma_p$',
        'initial_income': r'$\ln\bar{y}_0$',
        'pop_avg_minus_empirical': rf'$\gamma_{{{agg_level}}}^p - \gamma_{{{agg_level}}}$',
        'inc_avg_minus_empirical': rf'$\gamma_{{{agg_level}}}^i - \gamma_{{{agg_level}}}$',
        'pop_avg_minus_inc_avg': rf'$\gamma_{{{agg_level}}}^p - \gamma_{{{agg_level}}}^i$'
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot - simple dark blue points
    ax.scatter(plot_data[x_var], plot_data[y_var],
               alpha=0.6, s=50, color='darkblue', edgecolors='black', linewidth=0.5)

    add_regression_line_and_stats(ax, plot_data[x_var].values, plot_data[y_var].values)

    # Add horizontal line at y=0 for difference plots
    if 'minus' in y_var:
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)

    x_label = math_labels.get(x_var, x_var.replace('_', ' ').title())
    y_label = math_labels.get(y_var, y_var.replace('_', ' ').title())

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(base_output_dir, 'diff_scatter_plots')
    os.makedirs(output_path, exist_ok=True)
    full_filename = os.path.join(output_path, filename)
    plt.tight_layout()
    plt.savefig(full_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {full_filename}")


def create_all_plots(data, base_output_dir):
    """Create all requested difference scatter plots."""
    print("Creating all difference scatter plots...")

    os.makedirs(os.path.join(base_output_dir, 'diff_scatter_plots'), exist_ok=True)

    levels = data['level'].unique()
    print(f"Creating plots for levels: {levels}")

    plot_configs = [
        {
            'x_var': 'empirical_growth_rate',
            'y_var': 'pop_avg_minus_empirical',
            'title_template': 'Empirical Growth Rate vs (Pop Avg - Empirical) - {level}',
            'filename_template': 'empirical_vs_pop_minus_emp_{level_short}.png'
        },
        {
            'x_var': 'empirical_growth_rate',
            'y_var': 'inc_avg_minus_empirical',
            'title_template': 'Empirical Growth Rate vs (Inc Avg - Empirical) - {level}',
            'filename_template': 'empirical_vs_inc_minus_emp_{level_short}.png'
        },
        {
            'x_var': 'inc_averaged_growth_rate',
            'y_var': 'pop_avg_minus_inc_avg',
            'title_template': 'Income Averaged Growth Rate vs (Pop Avg - Inc Avg) - {level}',
            'filename_template': 'inc_avg_vs_pop_minus_inc_{level_short}.png'
        },
        {
            'x_var': 'initial_income',
            'y_var': 'pop_avg_minus_empirical',
            'title_template': 'Initial Income vs (Pop Avg - Empirical) - {level}',
            'filename_template': 'initial_vs_pop_minus_emp_{level_short}.png'
        },
        {
            'x_var': 'initial_income',
            'y_var': 'inc_avg_minus_empirical',
            'title_template': 'Initial Income vs (Inc Avg - Empirical) - {level}',
            'filename_template': 'initial_vs_inc_minus_emp_{level_short}.png'
        },
        {
            'x_var': 'initial_income',
            'y_var': 'pop_avg_minus_inc_avg',
            'title_template': 'Initial Income vs (Pop Avg - Inc Avg) - {level}',
            'filename_template': 'initial_vs_pop_minus_inc_{level_short}.png'
        }
    ]

    for level in levels:
        level_data = data[data['level'] == level].copy()
        level_short = {'Province': 'prov', 'Canton': 'can'}.get(level, level.lower()[:2])

        print(f"\nCreating plots for {level} level ({len(level_data)} observations)...")

        for config in plot_configs:
            title = config['title_template'].format(level=level)
            filename = config['filename_template'].format(level_short=level_short)

            create_scatter_plot(
                level_data,
                config['x_var'],
                config['y_var'],
                title,
                filename,
                base_output_dir
            )


def print_summary_statistics(data):
    """Print summary statistics for all variables."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    variables = ['empirical_growth_rate', 'pop_averaged_growth_rate', 'inc_averaged_growth_rate',
                 'initial_income', 'pop_avg_minus_empirical', 'inc_avg_minus_empirical', 'pop_avg_minus_inc_avg']

    for var in variables:
        if var in data.columns:
            print(f"\n{var.replace('_', ' ').title()}:")
            print(f"  Mean: {data[var].mean():.4f}")
            print(f"  Std:  {data[var].std():.4f}")
            print(f"  Min:  {data[var].min():.4f}")
            print(f"  Max:  {data[var].max():.4f}")
            print(f"  N:    {data[var].count()}")


def run_analysis(input_dir, base_output_dir):
    """Main execution function."""
    print("="*60)
    print(f"DIFFERENCE SCATTER PLOTS ANALYSIS for {input_dir}")
    print("="*60)

    cm_data, tr_data = load_data(input_dir)
    if cm_data is None:
        print("Error: Could not load data files")
        return

    datasets = []

    # Process province data (Chicago: community)
    cm_processed = prepare_variables(cm_data, 'Province', 'province')
    if not cm_processed.empty:
        datasets.append(cm_processed)
        print(f"Province data: {len(cm_processed)} observations")

    # Process canton data (Chicago: tract) if available
    if tr_data is not None:
        tr_processed = prepare_variables(tr_data, 'Canton', 'canton')
        if not tr_processed.empty:
            datasets.append(tr_processed)
            print(f"Canton data: {len(tr_processed)} observations")

    if not datasets:
        print("Error: No usable data found")
        return

    combined_data = pd.concat(datasets, ignore_index=True)
    print(f"Combined data: {len(combined_data)} total observations")

    print_summary_statistics(combined_data)

    create_all_plots(combined_data, base_output_dir)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    num_levels = len(combined_data['level'].unique())
    total_plots = 6 * num_levels

    print(f"Generated {total_plots} difference scatter plots in {base_output_dir}/diff_scatter_plots/ directory")
    print(f"({6} plot types × {num_levels} data levels)")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == "__main__":
    main()

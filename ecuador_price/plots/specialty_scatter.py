#!/usr/bin/env python3
"""
specialty_scatter.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/specialty_scatter.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Creates specialty scatter plots of the income-averaged (purple) and
population-averaged (orange) growth rates against the empirical growth rate,
with the two points of each unit connected by a vertical line and weighted
regression bands.

Chicago levels cm/tr -> Ecuador province/canton; per-county plots become
per-province plots (canton level only).  Chicago filtered units to
>5,000 initial households; the firm-count analog here is MIN_INITIAL_FIRMS.
Axis limits are data-driven (Ecuador growth rates span a wider range than
Chicago's hardcoded window).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

# Custom colors from plotPrice.py
custom_purple = ec.CUSTOM_PURPLE
custom_orange = ec.CUSTOM_ORANGE
purple_dot = ec.PURPLE_DOT
orange_dot = ec.ORANGE_DOT

# Chicago: initial population > 5,000 households.  Firm stocks are far
# smaller; keep units with a meaningful initial stock of firms.
MIN_INITIAL_FIRMS = 10


def load_data_by_level(input_dir, level):
    """Load and prepare the data for a specific level."""
    print(f"Loading {level}-level data...")

    data = ec.load_terms(input_dir, level)
    if data is None:
        print(f"Error: parish_{level}_exported_terms.csv not found")
        return None, level
    print(f"Loaded {level} data: {data.shape[0]} rows, {data.shape[1]} columns")
    return data, level


def prepare_variables(data, level_suffix):
    """Extract and calculate the required variables from the data."""
    print(f"Preparing variables for {level_suffix} level...")

    df = data.copy()

    # Empirical growth rate
    if f'AvgG_emp_{level_suffix}' in df.columns:
        df['empirical_growth_rate'] = df[f'AvgG_emp_{level_suffix}']
    else:
        emp_cols = [col for col in df.columns if 'AvgG_emp' in col]
        if emp_cols:
            df['empirical_growth_rate'] = df[emp_cols[0]]
            print(f"Using {emp_cols[0]} for empirical growth rate")
        else:
            print("Warning: Could not find empirical growth rate column")
            df['empirical_growth_rate'] = np.nan

    # Income-averaged growth rate
    if f'AvgG_inc_{level_suffix}' in df.columns:
        df['inc_averaged_growth_rate'] = df[f'AvgG_inc_{level_suffix}']
    else:
        inc_avg_cols = [col for col in df.columns if 'AvgG_inc' in col and 'Rel' not in col]
        if inc_avg_cols:
            df['inc_averaged_growth_rate'] = df[inc_avg_cols[0]]
            print(f"Using {inc_avg_cols[0]} for income averaged growth rate")
        else:
            print("Warning: Could not find income averaged growth rate")
            df['inc_averaged_growth_rate'] = np.nan

    # Population-averaged growth rate
    if f'AvgG_pop_{level_suffix}' in df.columns:
        df['pop_averaged_growth_rate'] = df[f'AvgG_pop_{level_suffix}']
    else:
        pop_avg_cols = [col for col in df.columns if 'AvgG_pop' in col and 'Rel' not in col]
        if pop_avg_cols:
            df['pop_averaged_growth_rate'] = df[pop_avg_cols[0]]
            print(f"Using {pop_avg_cols[0]} for population averaged growth rate")
        else:
            print("Warning: Could not find population averaged growth rate")
            df['pop_averaged_growth_rate'] = np.nan

    # Population growth rate (for coloring)
    if f'PopG_{level_suffix}' in df.columns:
        df['population_growth_rate'] = df[f'PopG_{level_suffix}']
    else:
        pop_growth_cols = [col for col in df.columns if 'PopG' in col or 'PopFracChange' in col]
        if pop_growth_cols:
            df['population_growth_rate'] = df[pop_growth_cols[0]]
            print(f"Using {pop_growth_cols[0]} for population growth rate")
        else:
            print("Warning: Could not find population growth rate")
            df['population_growth_rate'] = np.nan

    # Province identifier (Chicago: county identifier)
    county_col = None
    possible_county_cols = ['ParentProvince', 'Province', 'province', 'PROVINCE']
    for col in possible_county_cols:
        if col in df.columns:
            county_col = col
            break

    if level_suffix == 'province':
        df['county'] = df['UnitName'].astype(str)
        print("Using UnitName as province identifier")
    elif county_col:
        df['county'] = df[county_col].astype(str)
        print(f"Using {county_col} as province identifier")
    else:
        print("Warning: Could not find province identifier column")
        df['county'] = 'Unknown'

    # Difference variables (Chicago plots the levels themselves; the
    # "- empirical" is commented out there too)
    df['inc_diff'] = df['inc_averaged_growth_rate']  # - df['empirical_growth_rate']
    df['pop_diff'] = df['pop_averaged_growth_rate']  # - df['empirical_growth_rate']

    if 'UnitName' in df.columns:
        df['community_id'] = df['UnitName']
    else:
        df['community_id'] = df.index

    # Initial population for filtering
    initial_pop_col = f'PopInitial_{level_suffix}'
    if initial_pop_col in df.columns:
        df['initial_population'] = df[initial_pop_col]
        print(f"Using {initial_pop_col} for initial population")
    else:
        print(f"Warning: Could not find initial population column {initial_pop_col}")
        df['initial_population'] = np.nan

    essential_vars = ['empirical_growth_rate', 'inc_diff', 'pop_diff', 'population_growth_rate']
    df_clean = df.dropna(subset=essential_vars)

    # Filter by initial firm stock
    if 'initial_population' in df_clean.columns:
        initial_count = len(df_clean)
        df_clean = df_clean[df_clean['initial_population'] > MIN_INITIAL_FIRMS]
        filtered_count = len(df_clean)
        print(f"Filtered by initial firms > {MIN_INITIAL_FIRMS}: {filtered_count} units remaining "
              f"(removed {initial_count - filtered_count})")

    print(f"Data prepared: {len(df_clean)} complete observations out of {len(df)} total")

    return df_clean


def create_specialty_scatter(data, level_suffix, title_suffix="", figsize=(5, 4.5)):
    """Create the specialty scatter plot."""
    print(f"Creating specialty scatter plot{title_suffix}...")

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_facecolor('white')

    x_emp = data['empirical_growth_rate']
    y_inc_diff = data['inc_diff']
    y_pop_diff = data['pop_diff']
    community_ids = data['community_id']

    agg = ec.MATH_SUBSCRIPT.get(level_suffix, level_suffix)

    # First scatter: Empirical vs Income-averaged (custom purple)
    ax.scatter(x_emp, y_inc_diff,
               color=purple_dot,
               alpha=.25,
               s=50,
               edgecolors='#3C3C3C',
               label=rf'$\gamma_{{{agg}}}^i$',
               zorder=-100)

    # Second scatter: Empirical vs Population-averaged (custom orange)
    ax.scatter(x_emp, y_pop_diff,
               color=orange_dot,
               alpha=.25,
               s=50,
               edgecolors='#3C3C3C',
               label=rf'$\gamma_{{{agg}}}^p$',
               zorder=-100)

    # Draw connecting lines between the same units
    print("Drawing connecting lines between unit pairs...")
    for i, community_id in enumerate(community_ids):
        x_val = x_emp.iloc[i]
        y_inc_val = y_inc_diff.iloc[i]
        y_pop_val = y_pop_diff.iloc[i]

        ax.plot([x_val, x_val], [y_inc_val, y_pop_val],
                color='black',
                linewidth=0.5,
                alpha=0.25,
                zorder=-101)

    # Fit regression lines and confidence intervals
    print("Calculating best fit lines and confidence intervals...")

    valid_inc_mask = ~(np.isnan(x_emp) | np.isnan(y_inc_diff))
    valid_pop_mask = ~(np.isnan(x_emp) | np.isnan(y_pop_diff))

    x_emp_inc = x_emp[valid_inc_mask]
    y_inc_diff_clean = y_inc_diff[valid_inc_mask]
    x_emp_pop = x_emp[valid_pop_mask]
    y_pop_diff_clean = y_pop_diff[valid_pop_mask]

    x_range = np.linspace(x_emp.min(), x_emp.max(), 100)

    # Income regression (purple)
    if len(x_emp_inc) > 3:
        slope_inc, intercept_inc, r_value_inc, p_value_inc, std_err_inc = \
            stats.linregress(x_emp_inc, y_inc_diff_clean)

        y_inc_fit = slope_inc * x_range + intercept_inc

        n_inc = len(x_emp_inc)
        x_mean_inc = np.mean(x_emp_inc)
        sxx_inc = np.sum((x_emp_inc - x_mean_inc) ** 2)

        residuals_inc = y_inc_diff_clean - (slope_inc * x_emp_inc + intercept_inc)
        mse_inc = np.sum(residuals_inc ** 2) / (n_inc - 2)

        t_val_inc = stats.t.ppf(0.975, n_inc - 2)  # 95% CI
        se_inc = np.sqrt(mse_inc * (1/n_inc + (x_range - x_mean_inc)**2 / sxx_inc))
        ci_inc = t_val_inc * se_inc

        ax.plot(x_range, y_inc_fit, color=custom_purple, linewidth=2, alpha=1,
                label=f'$m$={slope_inc:.2f},$r²$={r_value_inc**2:.2f}', zorder=4)
        ax.fill_between(x_range, y_inc_fit - ci_inc, y_inc_fit + ci_inc,
                        color=custom_purple, alpha=0.5, zorder=3)

    # Population regression (orange)
    if len(x_emp_pop) > 3:
        slope_pop, intercept_pop, r_value_pop, p_value_pop, std_err_pop = \
            stats.linregress(x_emp_pop, y_pop_diff_clean)

        y_pop_fit = slope_pop * x_range + intercept_pop

        n_pop = len(x_emp_pop)
        x_mean_pop = np.mean(x_emp_pop)
        sxx_pop = np.sum((x_emp_pop - x_mean_pop) ** 2)

        residuals_pop = y_pop_diff_clean - (slope_pop * x_emp_pop + intercept_pop)
        mse_pop = np.sum(residuals_pop ** 2) / (n_pop - 2)

        t_val_pop = stats.t.ppf(0.975, n_pop - 2)  # 95% CI
        se_pop = np.sqrt(mse_pop * (1/n_pop + (x_range - x_mean_pop)**2 / sxx_pop))
        ci_pop = t_val_pop * se_pop

        ax.plot(x_range, y_pop_fit, color=custom_orange, linewidth=2, alpha=1,
                label=f'$m$={slope_pop:.2f},$r²$={r_value_pop**2:.2f}', zorder=2)
        ax.fill_between(x_range, y_pop_fit - ci_pop, y_pop_fit + ci_pop,
                        color=custom_orange, alpha=0.5, zorder=1)

    # Data-driven symmetric-ish limits (Chicago hardcoded its own window)
    all_x = x_emp.dropna()
    all_y = pd.concat([y_inc_diff, y_pop_diff]).dropna()
    if len(all_x) and len(all_y):
        x_pad = 0.05 * (all_x.max() - all_x.min() + 1e-9)
        y_pad = 0.05 * (all_y.max() - all_y.min() + 1e-9)
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
        ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)

    # Add reference lines
    ax.axhline(0, color='#3c3c3c', linestyle='-', linewidth=1, alpha=1, zorder=1)
    ax.axvline(0, color='#3c3c3c', linestyle='-', linewidth=1, alpha=1, zorder=1)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], linestyle='--', color='black', linewidth=1, alpha=.5, zorder=1)

    # Styling
    ax.set_xlabel(rf'$\gamma_{{{agg}}}$', fontsize=14, fontweight='normal')
    ax.set_ylabel(rf'$\gamma_{{{agg}}}^x$', fontsize=14, fontweight='normal')

    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')

    ax.tick_params(axis='both', which='major', labelsize=12, colors='#333333')

    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False,
              fontsize=7.5, facecolor='white', edgecolor='gray', framealpha=0.9)

    plt.tight_layout()

    return fig, ax


def save_plot(fig, filename):
    """Save the plot."""
    print(f"Saving plot to {filename}...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Plot saved successfully")


def print_summary_statistics(data, title="SUMMARY STATISTICS"):
    """Print summary statistics for the variables."""
    print("\n" + "="*60)
    print(title)
    print("="*60)

    variables = ['empirical_growth_rate', 'inc_averaged_growth_rate', 'pop_averaged_growth_rate',
                 'population_growth_rate', 'inc_diff', 'pop_diff']

    for var in variables:
        if var in data.columns:
            var_data = data[var].dropna()
            if len(var_data) > 0:
                print(f"\n{var.replace('_', ' ').title()}:")
                print(f"  Mean: {var_data.mean():.4f}")
                print(f"  Std:  {var_data.std():.4f}")
                print(f"  Min:  {var_data.min():.4f}")
                print(f"  Max:  {var_data.max():.4f}")
                print(f"  N:    {len(var_data)}")


def process_level(input_dir, base_output_dir, level, level_suffix):
    """Process a single geographic level."""
    print(f"\n{'='*60}")
    print(f"PROCESSING {level.upper()} LEVEL")
    print(f"{'='*60}")

    data, _ = load_data_by_level(input_dir, level)
    if data is None:
        print(f"Error: Could not load {level} data")
        return

    prepared_data = prepare_variables(data, level_suffix)
    if prepared_data.empty:
        print(f"Error: No usable data after preparation for {level}")
        return

    print_summary_statistics(prepared_data, f"COUNTRYWIDE {level.upper()} SUMMARY STATISTICS")

    # Full-country plot (Chicago: Full MSA)
    unit_name = "provinces" if level == "province" else "cantons"
    print(f"\nCreating plot for Full Country ({len(prepared_data)} {unit_name})...")
    fig_msa, ax_msa = create_specialty_scatter(prepared_data, level_suffix,
                                               title_suffix=f" - Full Country ({level.upper()})")
    save_plot(fig_msa, os.path.join(base_output_dir,
                                    f'specialty_scatter_plots/specialty_scatter_full_country_{level}.pdf'))

    plots_created = 1

    # Per-province plots (Chicago: per-county) — only meaningful for cantons
    if level == 'canton':
        unique_counties = prepared_data['county'].unique()
        print(f"\nFound {len(unique_counties)} unique provinces: {list(unique_counties)}")

        for county_code in unique_counties:
            if county_code == 'Unknown':
                continue

            county_data = prepared_data[prepared_data['county'] == county_code].copy()

            if len(county_data) < 2:
                print(f"Skipping {county_code}: only {len(county_data)} units")
                continue

            county_name = ec.province_label(county_code)

            print(f"\nCreating plot for {county_name} ({len(county_data)} {unit_name})...")

            print_summary_statistics(county_data, f"{county_name.upper()} {level.upper()} SUMMARY STATISTICS")

            fig_county, ax_county = create_specialty_scatter(
                county_data, level_suffix, title_suffix=f" - {county_name} ({level.upper()})")

            safe_county_name = county_name.replace(' ', '_').lower()
            filename = os.path.join(base_output_dir,
                                    f'specialty_scatter_plots/specialty_scatter_{safe_county_name}_{level}.pdf')
            save_plot(fig_county, filename)

            plots_created += 1

    print(f"\n{'='*60}")
    print(f"{level.upper()} ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(prepared_data)} total {unit_name}")
    print(f"Generated {plots_created} plots for {level.upper()} level")
    return plots_created


def run_analysis(input_dir, base_output_dir):
    """Main execution function."""
    print("="*60)
    print("SPECIALTY SCATTER PLOT ANALYSIS")
    print("="*60)

    os.makedirs(os.path.join(base_output_dir, 'specialty_scatter_plots'), exist_ok=True)

    # (level_code, level_suffix): Chicago ('cm','cm'), ('tr','tr')
    levels_to_process = [
        ('province', 'province'),
        ('canton', 'canton')
    ]

    total_plots = 0

    for level, level_suffix in levels_to_process:
        plots_generated = process_level(input_dir, base_output_dir, level, level_suffix)
        if plots_generated:
            total_plots += plots_generated

    print("\n" + "="*60)
    print("OVERALL ANALYSIS COMPLETE")
    print("="*60)
    print(f"Generated {total_plots} total plots across all levels")
    print("\nPlot features:")
    print("- Purple points: Empirical vs Income-averaged growth rates")
    print("- Orange points: Empirical vs Population-averaged growth rates")
    print("- Black lines connect points for the same geographic units")
    print("- Trend lines show best fit with 95% confidence intervals")
    print("- R² and slope (m) values shown in legend")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == "__main__":
    main()

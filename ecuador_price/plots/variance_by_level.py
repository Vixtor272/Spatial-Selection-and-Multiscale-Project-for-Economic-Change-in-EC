#!/usr/bin/env python3
"""
variance_by_level.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/variance_by_level.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Creates a plot showing the mean and spread of the empirical (sales) growth
rate and the population (firm-count) growth rate across the levels of
geographic aggregation: parish, canton, province
(Chicago: block group, tract, community, county).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

# Custom colors from plotPrice.py
custom_purple = ec.CUSTOM_PURPLE
custom_orange = ec.CUSTOM_ORANGE


def load_data_by_level(input_dir, level):
    """Load and prepare the data for a specific level."""
    print(f"Loading {level}-level data...")

    data = ec.load_terms(input_dir, level)
    if data is None:
        print(f"Error: parish_{level}_exported_terms.csv not found")
        return None
    print(f"Loaded {level} data: {data.shape[0]} rows, {data.shape[1]} columns")
    return data


def extract_growth_rate_stats(data, level_suffix):
    """Extract statistics for both empirical and population growth rates."""
    print(f"Extracting growth rate statistics for {level_suffix} level...")

    possible_emp_growth_cols = [
        f'AvgG_emp_{level_suffix}',
        'AvgG_emp',
        'empirical_growth_rate'
    ]

    possible_pop_growth_cols = [
        f'PopG_{level_suffix}',
        f'PopFracChange_{level_suffix}',
        'PopG',
        'PopFracChange',
        'population_growth_rate'
    ]

    stats = {}

    emp_col = None
    for col_pattern in possible_emp_growth_cols:
        matching_cols = [col for col in data.columns if col == col_pattern]
        if matching_cols:
            emp_col = matching_cols[0]
            break

    if emp_col:
        print(f"Using {emp_col} for empirical growth rate")
        emp_data = pd.to_numeric(data[emp_col], errors='coerce').dropna()
        if len(emp_data) > 1:
            std_val = emp_data.std()
            n_val = len(emp_data)
            stats['empirical'] = {
                'mean': emp_data.mean(),
                'std': std_val,
                'variance': emp_data.var(),
                'n': n_val,
                'ci_95_margin': 1.96 * (std_val / np.sqrt(n_val))
            }
            print(f"Empirical growth stats for {level_suffix}: mean={stats['empirical']['mean']:.6f}, "
                  f"std={stats['empirical']['std']:.6f}, n={stats['empirical']['n']}")
        else:
            stats['empirical'] = None
    else:
        print(f"Warning: Could not find empirical growth rate column for {level_suffix}")
        stats['empirical'] = None

    pop_col = None
    for col_pattern in possible_pop_growth_cols:
        matching_cols = [col for col in data.columns if col == col_pattern]
        if matching_cols:
            pop_col = matching_cols[0]
            break

    if pop_col:
        print(f"Using {pop_col} for population growth rate")
        pop_data = pd.to_numeric(data[pop_col], errors='coerce').dropna()
        if len(pop_data) > 1:
            std_val = pop_data.std()
            n_val = len(pop_data)
            stats['population'] = {
                'mean': pop_data.mean(),
                'std': std_val,
                'variance': pop_data.var(),
                'n': n_val,
                'ci_95_margin': 1.96 * (std_val / np.sqrt(n_val))
            }
            print(f"Population growth stats for {level_suffix}: mean={stats['population']['mean']:.6f}, "
                  f"std={stats['population']['std']:.6f}, n={stats['population']['n']}")
        else:
            stats['population'] = None
    else:
        print(f"Warning: Could not find population growth rate column for {level_suffix}")
        stats['population'] = None

    return stats


def create_line_plot_with_confidence(stats_data, figsize=(5.5, 2.4)):
    """Create a line plot showing mean growth rates with ±1 std bands
    (flipped axes: levels on y, growth rate on x — as in Chicago)."""
    print("Creating line plot with confidence intervals...")

    fig, ax = plt.subplots(figsize=figsize)

    levels = []
    emp_means = []
    emp_stds = []
    pop_means = []
    pop_stds = []
    emp_ci_margins = []
    pop_ci_margins = []

    for level, stats in stats_data.items():
        levels.append(level)

        if stats['empirical'] is not None:
            emp_means.append(stats['empirical']['mean'])
            emp_stds.append(stats['empirical']['std'])
            emp_ci_margins.append(stats['empirical']['ci_95_margin'])
        else:
            emp_means.append(np.nan)
            emp_stds.append(np.nan)
            emp_ci_margins.append(np.nan)

        if stats['population'] is not None:
            pop_means.append(stats['population']['mean'])
            pop_stds.append(stats['population']['std'])
            pop_ci_margins.append(stats['population']['ci_95_margin'])
        else:
            pop_means.append(np.nan)
            pop_stds.append(np.nan)
            pop_ci_margins.append(np.nan)

    if not levels:
        print("Warning: No valid data to plot")
        return fig, ax

    y_positions = range(len(levels))

    emp_means_clean = np.array(emp_means)
    emp_stds_clean = np.array(emp_stds)
    valid_emp = ~np.isnan(emp_means_clean)

    if np.any(valid_emp):
        ax.plot(emp_means_clean[valid_emp], np.array(y_positions)[valid_emp],
                color=custom_purple, marker='o', linewidth=2, markersize=6,
                label='Empirical Growth Rate')
        ax.fill_betweenx(np.array(y_positions)[valid_emp],
                         (emp_means_clean - emp_stds_clean)[valid_emp],
                         (emp_means_clean + emp_stds_clean)[valid_emp],
                         color=custom_purple, alpha=0.2)

    pop_means_clean = np.array(pop_means)
    pop_stds_clean = np.array(pop_stds)
    valid_pop = ~np.isnan(pop_means_clean)

    if np.any(valid_pop):
        ax.plot(pop_means_clean[valid_pop], np.array(y_positions)[valid_pop],
                color=custom_orange, marker='s', linewidth=2, markersize=6,
                label='Population Growth Rate')
        ax.fill_betweenx(np.array(y_positions)[valid_pop],
                         (pop_means_clean - pop_stds_clean)[valid_pop],
                         (pop_means_clean + pop_stds_clean)[valid_pop],
                         color=custom_orange, alpha=0.2)

    ax.set_title('Growth Rates by Aggregation Level\n(Mean with ±1 SD band)',
                 fontsize=14, fontweight='bold')

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(levels)
    ax.set_xlabel('Growth rate (log points / yr)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)

    # Sample sizes as text at a fixed x position (data-driven here; Chicago
    # hardcoded -0.065 for its own axis range)
    finite_vals = np.concatenate([
        (emp_means_clean - emp_stds_clean)[valid_emp],
        (pop_means_clean - pop_stds_clean)[valid_pop],
    ]) if (np.any(valid_emp) or np.any(valid_pop)) else np.array([0.0])
    custom_text_x = np.nanmin(finite_vals)
    for i, (level, stats) in enumerate(stats_data.items()):
        text_parts = []
        if stats['population'] is not None:
            text_parts.append(f"n={stats['population']['n']:,}")
        if text_parts:
            ax.text(custom_text_x, i + .15, ' / '.join(text_parts), ha='left', va='center',
                    fontsize=8, color='#666666')

    # Style the plot
    ax.set_ylim(top=len(levels) - 0.5)
    ax.spines['top'].set_color('#D3D3D3')
    ax.spines['right'].set_color('#D3D3D3')
    ax.spines['left'].set_color('#D3D3D3')
    ax.spines['bottom'].set_color('#D3D3D3')
    ax.grid(True, axis='y', alpha=0.3, color='#D3D3D3')

    ax.tick_params(colors='#333333', labelsize=12)

    ax.axvline(x=0, color='#3D3D3D', alpha=0.75, linewidth=1)
    for i in range(1, len(levels) + 1):
        ax.axhline(y=i, color='#D3D3D3', alpha=0.75, linewidth=1, zorder=-1)

    plt.tight_layout()

    return fig, ax


def save_plot(fig, base_output_dir, filename):
    """Saves a plot to a file, creating directories if necessary."""
    full_path = os.path.join(base_output_dir, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    fig.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {full_path}")


def print_summary_statistics(stats_data):
    """Print summary statistics for the growth rate data."""
    print("\n" + "="*60)
    print("GROWTH RATE SUMMARY STATISTICS")
    print("="*60)

    for level, stats in stats_data.items():
        print(f"\n{level.upper()} Level:")

        if stats['empirical'] is not None:
            emp = stats['empirical']
            print("  Empirical Growth Rate:")
            print(f"    Mean: {emp['mean']:.6f}")
            print(f"    Std Dev: {emp['std']:.6f}")
            print(f"    Variance: {emp['variance']:.6f}")
            print(f"    Sample Size: {emp['n']}")
        else:
            print("  Empirical Growth Rate: No valid data available")

        if stats['population'] is not None:
            pop = stats['population']
            print("  Population Growth Rate:")
            print(f"    Mean: {pop['mean']:.6f}")
            print(f"    Std Dev: {pop['std']:.6f}")
            print(f"    Variance: {pop['variance']:.6f}")
            print(f"    Sample Size: {pop['n']}")
        else:
            print("  Population Growth Rate: No valid data available")


def run_analysis(input_dir, base_output_dir):
    """Main execution function."""
    print("="*60)
    print("GROWTH RATES BY AGGREGATION LEVEL ANALYSIS")
    print("="*60)

    os.makedirs(os.path.join(base_output_dir, 'variance_plots'), exist_ok=True)

    # Levels in order from finest to coarsest
    # (Chicago: bg, tr, cm, ct — Ecuador drops one level)
    levels_to_process = [
        ('parish', 'parish', 'Parish'),
        ('canton', 'canton', 'Canton'),
        ('province', 'province', 'Province'),
    ]

    stats_data = {}

    for level_code, level_suffix, level_name in levels_to_process:
        print(f"\nProcessing {level_name} level...")

        data = load_data_by_level(input_dir, level_code)
        if data is None:
            stats_data[level_name] = {'empirical': None, 'population': None}
            continue

        stats = extract_growth_rate_stats(data, level_suffix)
        stats_data[level_name] = stats

    print_summary_statistics(stats_data)

    fig, ax = create_line_plot_with_confidence(stats_data)

    if fig is not None:
        save_plot(fig, base_output_dir, 'variance_plots/growth_rates_by_level.pdf')
        plt.close(fig)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Generated growth rates by aggregation level plot")
        print("Features:")
        print("- Y-axis: Geographic aggregation levels (finest to coarsest)")
        print("- X-axis: Growth rates (empirical and population)")
        print("- Line colors: Purple (empirical), Orange (population)")
        print("- Shaded region: ±1 standard deviation")
    else:
        print("\nError: Could not generate plot due to insufficient data")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
inequality_scatter.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/inequality_scatter.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Creates scatter plots of:
1. Relative Income-Averaged Growth Rate vs Cumulative Income NS_ec
2. Relative Population-Averaged Growth Rate vs Cumulative Population NS_ec

Chicago community -> Ecuador province; the state-normalized NS_st becomes the
country-normalized NS_ec (Transmitted_Sel_canton_to_province_*_ns_country +
Sel_province_from_canton_*_ns_country).  Points are colored by relative
initial income (de-medianed log real sales) with the purple-white-orange
colorbar and sized by the initial firm stock.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

# Custom colors
custom_purple = ec.CUSTOM_PURPLE
custom_orange = ec.CUSTOM_ORANGE
outline_color = ec.OUTLINE_COLOR


def load_data(input_dir):
    """Load province data (the Chicago community analog)."""
    df = ec.load_terms(input_dir, 'province')
    if df is None:
        print(f"Error loading province terms from {input_dir}")
        return None
    df = ec.add_country_ns(df, input_dir)
    print(f"Successfully loaded province terms with {len(df)} rows")
    return df


def extract_sum_cm(df, trans_col, sel_col, scale=None):
    """Extract and sum transmitted and selection columns for province data"""
    if df is None:
        return pd.Series([])

    trans_vals = df[trans_col].fillna(0) if trans_col in df.columns else 0
    sel_vals = df[sel_col].fillna(0) if sel_col in df.columns else 0

    result = trans_vals + sel_vals

    if scale is not None:
        result = result * scale

    return result


def create_custom_colormap():
    """Create a custom colormap from purple through white to orange"""
    colors = [custom_purple, 'white', custom_orange]
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return cmap


def create_scatter_plot(x_data, y_data, color_data, x_label, y_label, color_label,
                        title, filename, base_output_dir, size_data=None, label=None):
    """Create a scatter plot with specified data and labels, colored by color_data"""
    # Remove NaN, inf, and exactly zero values from x and y data
    valid_mask = (np.isfinite(x_data) & np.isfinite(y_data) &
                  ~np.isnan(x_data) & ~np.isnan(y_data) &
                  (x_data != 0) & (y_data != 0))

    if size_data is not None:
        valid_mask &= np.isfinite(size_data) & ~np.isnan(size_data)

    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]
    label_clean = label[valid_mask] if label is not None else None
    color_clean = color_data[valid_mask]

    if size_data is not None:
        size_clean = size_data[valid_mask]
        s_min, s_max = 15, 150
        if size_clean.min() == size_clean.max():
            sizes = pd.Series(s_min, index=size_clean.index)
        else:
            sizes = s_min + ((size_clean - size_clean.min()) /
                             (size_clean.max() - size_clean.min()) * (s_max - s_min))
    else:
        sizes = 25

    print(f"\n{'='*60}")
    print(f"SCATTER PLOT: {title}")
    print(f"{'='*60}")
    print(f"Valid data points: {len(x_clean)}")

    if len(x_clean) == 0:
        print("No valid data points to plot!")
        return

    print(f"X-axis ({x_label}):")
    print(f"  - Mean: {x_clean.mean():.6f}")
    print(f"  - Std: {x_clean.std():.6f}")
    print(f"  - Min: {x_clean.min():.6f}")
    print(f"  - Max: {x_clean.max():.6f}")

    print(f"Y-axis ({y_label}):")
    print(f"  - Mean: {y_clean.mean():.6f}")
    print(f"  - Std: {y_clean.std():.6f}")
    print(f"  - Min: {y_clean.min():.6f}")
    print(f"  - Max: {y_clean.max():.6f}")

    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
    print(f"Correlation coefficient: {correlation:.6f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    custom_cmap = create_custom_colormap()

    # Determine color limits to center white at zero
    color_abs_max = max(abs(color_clean.min()), abs(color_clean.max()))
    vmin = -color_abs_max
    vmax = color_abs_max

    scatter = ax.scatter(x_clean, y_clean, c=color_clean, cmap=custom_cmap,
                         alpha=0.8, s=sizes, edgecolors=outline_color, linewidth=0.8,
                         vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(color_label, fontsize=12, color='#333333')
    cbar.ax.tick_params(labelsize=10, colors='#333333')

    # Add reference lines
    ax.axhline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel(x_label, fontsize=14, color='#333333')
    # Smaller font: the rotated y-label must fit the 4-inch figure height
    ax.set_ylabel(y_label, fontsize=11, color='#333333')

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CCCCCC')

    ax.tick_params(axis='x', which='major', labelsize=12, colors='#333333')
    ax.tick_params(axis='y', which='major', labelsize=12, colors='#333333')

    # (Chicago clipped the income-NS x-axis at -75 for its own outliers; the
    # Ecuador ranges differ, so the axis is left data-driven.)

    plt.tight_layout()
    os.makedirs(os.path.join(base_output_dir, "inequality_scatter_plots"), exist_ok=True)
    full_filename = os.path.join(os.path.join(base_output_dir, "inequality_scatter_plots"), filename)
    plt.savefig(full_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {full_filename}")


def run_analysis(input_dir, base_output_dir):
    # Load province data (Chicago: community)
    cm_data = load_data(input_dir)

    if cm_data is None:
        print(f"Error: Could not load province data file from {input_dir}.")
        return

    # Extract relative growth rates
    if 'RelAvgG_emp_province' in cm_data.columns:
        rel_emp_growth = cm_data['RelAvgG_emp_province']  # noqa: F841
        rel_pop_growth = cm_data['RelAvgG_pop_province']
        rel_inc_growth = cm_data['RelAvgG_inc_province']
        rel_den_growth = cm_data.get('RelAvgG_den_province')
        print(f"Successfully extracted relative empirical growth rate: {len(rel_pop_growth)} values")
    else:
        print("Error: 'RelAvgG_emp_province' column not found in province data.")
        print("Available columns:", list(cm_data.columns))
        return

    # Relative initial income for coloring (Chicago used the same de-medianed
    # LogAvgIncInitial despite the variable name)
    if 'LogAvgIncInitial_province' in cm_data.columns:
        pop_growth = cm_data['LogAvgIncInitial_province']
        pop_growth = pop_growth - pop_growth.median()
        print(f"Successfully extracted relative initial income: {len(pop_growth)} values")
    else:
        print("Error: 'LogAvgIncInitial_province' column not found in province data.")
        print("Available columns:", list(cm_data.columns))
        return

    # Compute cumulative income NS_ec (country-normalized)
    print("\nComputing cumulative income NS_ec...")
    cm_inc_ns = extract_sum_cm(cm_data, 'Transmitted_Sel_canton_to_province_inc_ns_country',
                               'Sel_province_from_canton_inc_ns_country', scale=1)

    # Compute cumulative population NS_ec
    print("Computing cumulative population NS_ec...")
    cm_pop_ns = extract_sum_cm(cm_data, 'Transmitted_Sel_canton_to_province_pop_ns_country',
                               'Sel_province_from_canton_pop_ns_country', scale=1)

    # Compute cumulative density NS_ec (den path: trait = firm count)
    print("Computing cumulative density NS_ec...")
    cm_den_ns = extract_sum_cm(cm_data, 'Transmitted_Sel_canton_to_province_den_ns_country',
                               'Sel_province_from_canton_den_ns_country', scale=1)

    pop_initial_cm = cm_data['PopInitial_province'] if 'PopInitial_province' in cm_data.columns else None

    cm_label = cm_data['UnitName']

    print("\nCreating scatter plots...")

    # Plot 1: Relative Income-Averaged Growth Rate vs Income NS_ec
    create_scatter_plot(
        x_data=cm_inc_ns,
        y_data=rel_inc_growth,
        color_data=pop_growth,
        size_data=pop_initial_cm,
        x_label="Cumulative Income NS_ec (%)",
        y_label="Relative Income-Averaged Growth Rate",
        color_label="Relative Initial Income",
        title="Relative Income-Averaged Growth Rate vs Income NS_ec",
        filename="rel_growth_vs_income_ns.pdf",
        base_output_dir=base_output_dir,
        label=cm_label
    )

    # Plot 2: Relative Population-Averaged Growth Rate vs Population NS_ec
    create_scatter_plot(
        x_data=cm_pop_ns,
        y_data=rel_pop_growth,
        color_data=pop_growth,
        size_data=pop_initial_cm,
        x_label="Cumulative Population NS_ec (%)",
        y_label="Relative Population-Averaged Growth Rate",
        color_label="Relative Initial Income",
        title="Relative Population-Averaged Growth Rate vs Population NS_ec",
        filename="rel_growth_vs_population_ns.pdf",
        base_output_dir=base_output_dir,
        label=cm_label
    )

    # Plot 3: den-path analog (trait = firm count; growth units = firms/yr)
    if rel_den_growth is not None:
        create_scatter_plot(
            x_data=cm_den_ns,
            y_data=rel_den_growth,
            color_data=pop_growth,
            size_data=pop_initial_cm,
            x_label="Cumulative Density NS_ec (%)",
            y_label="Relative Density-Averaged Growth Rate",
            color_label="Relative Initial Income",
            title="Relative Density-Averaged Growth Rate vs Density NS_ec",
            filename="rel_growth_vs_density_ns.pdf",
            base_output_dir=base_output_dir,
            label=cm_label
        )

    print("\n" + "="*60)
    print(f"INEQUALITY SCATTER ANALYSIS COMPLETE FOR {input_dir}")
    print("="*60)
    print("Generated plots:")
    print(f"1. {os.path.join(base_output_dir, 'inequality_scatter_plots', 'rel_growth_vs_income_ns.pdf')}")
    print(f"2. {os.path.join(base_output_dir, 'inequality_scatter_plots', 'rel_growth_vs_population_ns.pdf')}")
    print(f"3. {os.path.join(base_output_dir, 'inequality_scatter_plots', 'rel_growth_vs_density_ns.pdf')}")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == "__main__":
    main()

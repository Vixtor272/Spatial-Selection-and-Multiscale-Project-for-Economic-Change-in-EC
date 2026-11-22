"""
selection_concentration_curves.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/selection_concentration_curves.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Concentration (Lorenz-style) curves of the population-path selection effect.

Chicago -> Ecuador mapping:
    Tract level      -> Canton level    (Sel_canton_from_parish_pop)
    Community level  -> Province level  (Sel_province_from_canton_pop)
    County grouping  -> Province grouping (canton level only; the province
                        level has a single unit per province, so the
                        multi-group figure is skipped there)
    Statewide        -> Countrywide (Ecuador)
    Cook County      -> the focal province (Pichincha by default)

The focal-province canton-vs-province level comparison is skipped because the
province level degenerates to one unit inside a single province.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

# The multi-province comparison shows only the largest provinces (by initial
# firm stock); 24 curves were unreadable.  The countrywide reference curve
# still uses every province.
TOP_N_PROVINCES = 10


# --- ANALYSIS FUNCTIONS ---

def calculate_curve_data(df, selection_col, population_col, analysis_type,
                         use_unit_count_xaxis=False, sort_by_magnitude=False):
    """
    Helper function to calculate the x and y coordinates for a concentration curve.

    analysis_type: 'Positive', 'Negative', or 'Absolute'
    use_unit_count_xaxis: If True, x-axis is cumulative share of units, not population.
    sort_by_magnitude: If True, sorts by selection magnitude instead of per-capita intensity.
    """
    if df.empty:
        return None, None

    # 1. Separate data
    if analysis_type == 'Positive':
        sub_df = df[df[selection_col] > 0].copy()
        sel_col_to_process = selection_col
    elif analysis_type == 'Negative':
        sub_df = df[df[selection_col] < 0].copy()
        sel_col_to_process = selection_col
    else:  # Absolute
        sub_df = df.copy()
        sub_df['abs_selection'] = sub_df[selection_col].abs()
        sel_col_to_process = 'abs_selection'

    if sub_df.empty:
        return None, None

    # 2. Get totals for normalization
    total_sel = sub_df[sel_col_to_process].sum()
    if total_sel == 0:
        return None, None

    # 3. Sort units based on the chosen metric
    sort_ascending = True if analysis_type == 'Negative' else False

    if sort_by_magnitude:
        sorted_df = sub_df.sort_values(by=sel_col_to_process, ascending=sort_ascending)
    else:
        epsilon = 1e-9
        sub_df['selection_per_capita'] = sub_df[sel_col_to_process] / (sub_df[population_col] + epsilon)
        sorted_df = sub_df.sort_values(by='selection_per_capita', ascending=sort_ascending)

    # 4. Calculate cumulative shares
    if use_unit_count_xaxis:
        total_units = len(sorted_df)
        sorted_df['cum_x_share'] = np.arange(1, total_units + 1) / total_units
    else:
        subset_population = sub_df[population_col].sum()
        if subset_population == 0:
            return None, None
        sorted_df['cum_x_share'] = sorted_df[population_col].cumsum() / subset_population

    sorted_df['cum_sel_share'] = sorted_df[sel_col_to_process].cumsum() / total_sel

    # 5. Prepare plot data
    x_vals = np.concatenate([[0], sorted_df['cum_x_share']])
    y_vals = np.concatenate([[0], sorted_df['cum_sel_share']])

    return x_vals, y_vals


def create_county_comparison_plot(df_statewide, config, selection_type, output_dir):
    """
    Generates a 1x3 plot comparing concentration curves for all provinces found in the data.
    """
    level_name = config['level_name']
    population_col = config['pop_col']
    selection_col = config['sel_inc_col'] if selection_type == 'Income' else config['sel_pop_col']
    county_col = config['county_col']

    print(f"  Generating multi-province comparison plot for {level_name} - {selection_type} Selection...")

    # Keep only the TOP_N_PROVINCES largest provinces by initial firm stock
    province_sizes = (df_statewide.groupby(df_statewide[county_col].str.strip())[population_col]
                      .sum().sort_values(ascending=False))
    unique_counties = list(province_sizes.head(TOP_N_PROVINCES).index)
    print(f"    Limiting to the top {len(unique_counties)} provinces by firm stock: "
          f"{[ec.province_label(c) for c in unique_counties]}")

    # Map DPA province codes to names for a cleaner legend
    fips_to_name = ec.PROVINCE_NAME

    colors = plt.cm.inferno(np.linspace(0, 1, len(unique_counties)))
    county_color_map = {county: color for county, color in zip(unique_counties, colors)}

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

    analysis_types = ['Positive', 'Negative', 'Absolute']

    for i, analysis_type in enumerate(analysis_types):
        ax = axes[i]

        # Plot a curve for each province
        for fips in unique_counties:
            df_county = df_statewide[df_statewide[county_col].str.strip() == fips]
            if not df_county.empty:
                color = county_color_map[fips]

                x, y = calculate_curve_data(df_county, selection_col, population_col, analysis_type)
                if x is not None:
                    ax.plot(x, y, color=color, linewidth=2, alpha=.75)

        # Plot the countrywide curve for comparison
        x_state, y_state = calculate_curve_data(df_statewide, selection_col, population_col, analysis_type)
        if x_state is not None:
            ax.plot(x_state, y_state, label='Countrywide', color='black', linestyle='--', linewidth=2, zorder=3)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_title(f'{analysis_type} Selection', fontsize=14, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.set_xlabel('Cumulative share of firm stock', fontsize=11)
        ax.grid(True)

    axes[0].set_ylabel('Cumulative share of selection effect', fontsize=11)

    # Single legend (Chicago attached it to the community-level figure; here
    # the canton level is the only multi-group figure)
    county_handles = [plt.Line2D([0], [0], color=county_color_map[fips], lw=2) for fips in unique_counties]
    statewide_handle = plt.Line2D([0], [0], color='black', lw=2, linestyle='--')
    handles = [statewide_handle] + county_handles
    county_labels = [fips_to_name.get(fips, f'Province {fips}') for fips in unique_counties]
    labels = ['Countrywide'] + county_labels
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.02, 0.9), fontsize=10)

    filename = f'multi_county_{level_name.lower()}_{selection_type.lower()}_concentration.pdf'
    full_path = os.path.join(output_dir, filename)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.93])
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Multi-province comparison plot saved to {full_path}")


def create_unified_concentration_plot(df, population_col, selection_col, scope_name, selection_type, output_dir):
    """
    Positive, negative, and absolute curves on the same axis.
    """
    print(f"  Generating unified plot for {scope_name} - {selection_type} Selection...")

    df_pos = df[df[selection_col] > 0].copy()
    df_neg = df[df[selection_col] < 0].copy()
    df_abs = df.copy()
    df_abs['abs_selection'] = df_abs[selection_col].abs()

    total_population = df[population_col].sum()
    total_selection_pos = df_pos[selection_col].sum()
    total_selection_neg = df_neg[selection_col].sum()
    total_selection_abs = df_abs['abs_selection'].sum()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    datasets = {
        'Positive Selection': (df_pos, selection_col, total_selection_pos, ec.CUSTOM_ORANGE),
        'Negative Selection': (df_neg, selection_col, total_selection_neg, ec.CUSTOM_PURPLE),
        'Absolute Selection': (df_abs, 'abs_selection', total_selection_abs, 'black')
    }

    for label, (sub_df, sel_col, total_sel, color) in datasets.items():
        if sub_df.empty or total_sel == 0:
            continue

        epsilon = 1e-9
        sub_df['selection_per_capita'] = sub_df[sel_col] / (sub_df[population_col] + epsilon)
        sort_ascending = True if label == 'Negative Selection' else False
        sorted_df = sub_df.sort_values(by='selection_per_capita', ascending=sort_ascending)

        subset_population = sub_df[population_col].sum()
        pop_norm = subset_population if label in ['Positive Selection', 'Negative Selection'] else total_population
        if pop_norm == 0:
            continue

        sorted_df['cum_pop_share'] = sorted_df[population_col].cumsum() / pop_norm
        sorted_df['cum_sel_share'] = sorted_df[sel_col].cumsum() / total_sel

        x_vals = np.concatenate([[0], sorted_df['cum_pop_share']])
        y_vals = np.concatenate([[0], sorted_df['cum_sel_share']])

        ax.plot(x_vals, y_vals, label=label, color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Line of Equality')
    ax.set_title(f'{scope_name}: {selection_type} Selection Concentration', fontsize=16, fontweight='bold')
    ax.set_xlabel('Cumulative share of firm stock', fontsize=12)
    ax.set_ylabel(f'Cumulative share of {selection_type} selection effect', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left')
    ax.grid(True)

    filename = f'unified_{scope_name.lower().replace(" ", "_")}_{selection_type.lower()}_concentration.pdf'
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Unified plot saved to {full_path}")


def create_level_comparison_plot(scope_name, df_tract, df_community, tract_config,
                                 community_config, selection_type, output_dir):
    """
    Compares Canton and Province absolute selection concentration on the same axes.
    """
    print(f"  Generating level comparison plot for {scope_name} - {selection_type} Selection...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5), sharey=True)

    analysis_type = 'Absolute'

    CUSTOM_PURPLE = ec.CUSTOM_PURPLE
    CUSTOM_ORANGE = ec.CUSTOM_ORANGE

    # --- Canton Data (Chicago: Tract) ---
    tract_pop_col = tract_config['pop_col']
    tract_sel_col = tract_config['sel_inc_col'] if selection_type == 'Income' else tract_config['sel_pop_col']
    x_tract, y_tract = calculate_curve_data(df_tract, tract_sel_col, tract_pop_col, analysis_type)

    if x_tract is not None:
        ax.plot(x_tract, y_tract, label='Canton', color=CUSTOM_ORANGE, linestyle='-',
                linewidth=2, alpha=.95, zorder=1)
        try:
            print(f"The share of Canton selection at {x_tract[np.where(x_tract > 0.198)[0][0]]:.2f} "
                  f"of the population for {analysis_type} is {y_tract[np.where(x_tract > 0.198)[0][0]]:.2f}")
        except IndexError:
            pass

    # --- Province Data (Chicago: Community) ---
    comm_pop_col = community_config['pop_col']
    comm_sel_col = community_config['sel_inc_col'] if selection_type == 'Income' else community_config['sel_pop_col']
    x_comm, y_comm = calculate_curve_data(df_community, comm_sel_col, comm_pop_col, analysis_type)

    if x_comm is not None:
        ax.plot(x_comm, y_comm, label='Province', color=CUSTOM_PURPLE, linestyle='-',
                linewidth=2, alpha=.95, zorder=1)
        try:
            print(f"The share of Province selection at {x_comm[np.where(x_comm > 0.198)[0][0]]:.2f} "
                  f"of the population for {analysis_type} is {y_comm[np.where(x_comm > 0.198)[0][0]]:.2f}")
        except IndexError:
            pass

    ax.plot([0, 1], [0, 1], '--', color='#3D3D3D', alpha=0.5, linewidth=1.5, zorder=0)
    ax.set_xlim(-.005, 1.005)
    ax.set_ylim(-.005, 1.005)
    ax.grid(True, zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_xlabel('Cumulative share of firm stock', fontsize=12)
    ax.set_ylabel('Cumulative share of |selection|', fontsize=12)

    ax.legend()

    filename = f'level_comparison_{scope_name.lower().replace(" ", "_")}_{selection_type.lower()}_concentration.pdf'
    full_path = os.path.join(output_dir, filename)
    print(filename)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Level comparison plot saved to {full_path}")


def create_level_comparison_plot_by_units(scope_name, df_tract, df_community, tract_config,
                                          community_config, selection_type, output_dir):
    """
    Same comparison with the X-axis as the cumulative share of units.
    """
    print(f"  Generating level comparison plot by unit count for {scope_name} - {selection_type} Selection...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5), sharey=True)

    analysis_type = 'Absolute'

    CUSTOM_PURPLE = ec.CUSTOM_PURPLE
    CUSTOM_ORANGE = ec.CUSTOM_ORANGE

    tract_pop_col = tract_config['pop_col']
    tract_sel_col = tract_config['sel_inc_col'] if selection_type == 'Income' else tract_config['sel_pop_col']
    x_tract, y_tract = calculate_curve_data(df_tract, tract_sel_col, tract_pop_col, analysis_type,
                                            use_unit_count_xaxis=True, sort_by_magnitude=True)

    if x_tract is not None:
        ax.plot(x_tract, y_tract, label='Canton', color=CUSTOM_ORANGE, linestyle='-',
                linewidth=2, alpha=.95, zorder=1)
        try:
            print(f"The share of Canton selection at {x_tract[np.where(x_tract > 0.198)[0][0]]:.2f} "
                  f"of the NUMBER OF UNITS for {analysis_type} is {y_tract[np.where(x_tract > 0.198)[0][0]]:.2f}")
        except IndexError:
            pass

    comm_pop_col = community_config['pop_col']
    comm_sel_col = community_config['sel_inc_col'] if selection_type == 'Income' else community_config['sel_pop_col']
    x_comm, y_comm = calculate_curve_data(df_community, comm_sel_col, comm_pop_col, analysis_type,
                                          use_unit_count_xaxis=True, sort_by_magnitude=True)

    if x_comm is not None:
        ax.plot(x_comm, y_comm, label='Province', color=CUSTOM_PURPLE, linestyle='-',
                linewidth=2, alpha=.95, zorder=1)
        try:
            print(f"The share of Province selection at {x_comm[np.where(x_comm > 0.198)[0][0]]:.2f} "
                  f"of the NUMBER OF UNITS for {analysis_type} is {y_comm[np.where(x_comm > 0.198)[0][0]]:.2f}")
        except IndexError:
            pass

    ax.plot([0, 1], [0, 1], '--', color='#3D3D3D', alpha=0.5, linewidth=1.5, zorder=0)
    ax.set_xlim(-.005, 1.005)
    ax.set_ylim(-.005, 1.005)
    ax.grid(True, zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel('Cumulative share of units', fontsize=12)
    ax.set_ylabel('Cumulative share of |selection|', fontsize=12)

    ax.legend()

    filename = f'level_comparison_by_unit_{scope_name.lower().replace(" ", "_")}_{selection_type.lower()}_concentration.pdf'
    full_path = os.path.join(output_dir, filename)
    print(filename)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Level comparison plot by unit count saved to {full_path}")


def run_analysis(input_dir, base_output_dir):
    """
    Main function to generate all concentration curve plots for a given dataset.
    """
    print(f"\n{'='*60}\nGenerating Selection Concentration Curves for {input_dir}\n{'='*60}")

    focal_code = ec.FOCAL_PROVINCE_CODE
    focal_name = ec.focal_province_name()

    # --- CONFIGURATION (Chicago tr/cm -> Ecuador canton/province) ---
    LEVELS_TO_ANALYZE = {
        'canton': {
            'data_path': ec.terms_path(input_dir, 'canton'),
            'pop_col': 'PopInitial_canton',
            'sel_inc_col': 'Sel_canton_from_parish_inc',
            'sel_pop_col': 'Sel_canton_from_parish_pop',
            'level_name': 'Canton',
            'county_col': 'ParentProvince'
        },
        'province': {
            'data_path': ec.terms_path(input_dir, 'province'),
            'pop_col': 'PopInitial_province',
            'sel_inc_col': 'Sel_province_from_canton_inc',
            'sel_pop_col': 'Sel_province_from_canton_pop',
            'level_name': 'Province',
            'county_col': 'UnitName'
        }
    }

    OUTPUT_DIR = os.path.join(base_output_dir, 'concentration_curves')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load all data first ---
    statewide_data = {}
    cook_county_data = {}
    for level_code, config in LEVELS_TO_ANALYZE.items():
        df = ec.load_terms(input_dir, level_code)
        if df is None:
            print(f"Could not find data file: {config['data_path']}. Skipping this level.")
            continue

        statewide_data[level_code] = df

        county_col = config.get('county_col')
        if county_col and county_col in df.columns:
            # Filter for the focal province (the Cook County analog)
            if level_code == 'canton':
                df_cook = df[df[county_col].str.strip() == focal_code].copy()
            else:
                df_cook = df[df['UnitName'].str.strip() == focal_code].copy()
            if not df_cook.empty:
                cook_county_data[level_code] = df_cook

    # --- Generate comparative and unified plots ---
    print(f"\n--- Generating Countrywide vs. {focal_name} Plots (per level) ---")
    for level_code, config in LEVELS_TO_ANALYZE.items():
        if level_code not in statewide_data:
            continue

        df_state = statewide_data[level_code]
        df_cook = cook_county_data.get(level_code, pd.DataFrame())
        level_name = config['level_name']

        print(f"\nProcessing {level_name} level...")

        # Multi-group comparison: only meaningful at the canton level, where
        # each province groups several cantons.
        if level_code == 'canton':
            create_county_comparison_plot(df_state, config, 'Population', OUTPUT_DIR)
        else:
            print("  Skipping multi-province plot at the province level "
                  "(each province is a single unit there).")

        create_unified_concentration_plot(df_state, config['pop_col'], config['sel_pop_col'],
                                          f"Countrywide {level_name}", 'Population', OUTPUT_DIR)
        if not df_cook.empty and len(df_cook) > 1:
            create_unified_concentration_plot(df_cook, config['pop_col'], config['sel_pop_col'],
                                              f"{focal_name} {level_name}", 'Population', OUTPUT_DIR)

    # --- Canton vs. Province comparison plots (Chicago: Tract vs Community) ---
    print("\n--- Generating Canton vs. Province Plots (per scope) ---")
    if 'canton' in statewide_data and 'province' in statewide_data:
        create_level_comparison_plot(
            'Countrywide',
            statewide_data['canton'],
            statewide_data['province'],
            LEVELS_TO_ANALYZE['canton'],
            LEVELS_TO_ANALYZE['province'],
            'Population',
            OUTPUT_DIR
        )
    # (Chicago also compared the levels within Cook County; inside one focal
    # province the province level is a single unit, so that variant is skipped.)

    # --- Canton vs. Province comparison plots BY UNIT COUNT ---
    print("\n--- Generating Canton vs. Province Plots by Unit Count (per scope) ---")
    if 'canton' in statewide_data and 'province' in statewide_data:
        create_level_comparison_plot_by_units(
            'Countrywide',
            statewide_data['canton'],
            statewide_data['province'],
            LEVELS_TO_ANALYZE['canton'],
            LEVELS_TO_ANALYZE['province'],
            'Population',
            OUTPUT_DIR
        )


# --- MAIN EXECUTION ---

def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == "__main__":
    main()

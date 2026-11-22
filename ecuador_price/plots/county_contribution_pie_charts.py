"""
county_contribution_pie_charts.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/county_contribution_pie_charts.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Chicago showed, for the Tract and Community levels, each county's share of the
positive / negative / absolute population-path selection effect.  Ecuador has
one fewer level, so the rows become:

    Canton level    Sel_canton_from_parish_pop   grouped by province
    Province level  Sel_province_from_canton_pop each province is its own slice

"Population" is the number of active firms; the county grouping becomes the
province grouping (see _ecuador_common).  Colors, layout and the three summary
figures (contribution, per-capita density, population share) are unchanged.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

# Custom Colors for Plots
CUSTOM_PURPLE = ec.CUSTOM_PURPLE
CUSTOM_ORANGE = ec.CUSTOM_ORANGE

# Custom Colormaps
PURPLE_CMAP = LinearSegmentedColormap.from_list("custom_purple", ['#ffffff', CUSTOM_PURPLE])
ORANGE_CMAP = LinearSegmentedColormap.from_list("custom_orange", ['#ffffff', CUSTOM_ORANGE])


def levels_to_analyze(input_dir):
    """Ecuador analog of LEVELS_TO_ANALYZE (Tract -> Canton, Community -> Province)."""
    return {
        'Canton': {
            'data_path': ec.terms_path(input_dir, 'canton'),
            'level': 'canton',
            'pop_col': 'PopInitial_canton',
            'sel_inc_col': 'Sel_canton_from_parish_inc',
            'sel_pop_col': 'Sel_canton_from_parish_pop',
            'county_col': 'ParentProvince',
        },
        'Province': {
            'data_path': ec.terms_path(input_dir, 'province'),
            'level': 'province',
            'pop_col': 'PopInitial_province',
            'sel_inc_col': 'Sel_province_from_canton_inc',
            'sel_pop_col': 'Sel_province_from_canton_pop',
            'county_col': 'UnitName',   # each province is its own group
        },
    }


# --- PLOTTING FUNCTION ---

def create_pie_chart(ax, data, title, normalization_total=None, custom_cmap=None):
    """
    Creates a styled pie chart on a given matplotlib axis.
    If normalization_total is provided, percentages are calculated against it.
    """
    total = normalization_total if normalization_total is not None else data.sum()
    if total == 0:  # Avoid division by zero
        return

    # Group small slices into 'Other'
    threshold = 0.01  # Group anything less than 0.01%
    small_slices = data[data / total * 100 < threshold]
    other_sum = small_slices.sum()

    main_slices = data[data / total * 100 >= threshold]
    if other_sum > 0:
        main_slices['Other'] = other_sum

    # Sort slices for consistent color mapping
    main_slices = main_slices.sort_values(ascending=False)

    labels = main_slices.index
    sizes_for_pct = main_slices.values

    # Define colors
    if custom_cmap:
        cmap = custom_cmap
    else:
        cmap = plt.get_cmap("inferno")
    # Apply colors from dark to light
    colors = cmap(np.flip(np.linspace(.9, 0, len(labels))))

    # Name the slices; small ones stay unlabeled to avoid clutter.
    slice_shares = sizes_for_pct / sizes_for_pct.sum()
    slice_labels = [name if share >= 0.02 else ''
                    for name, share in zip(labels, slice_shares)]

    wedges, texts, autotexts = ax.pie(
        sizes_for_pct,
        labels=slice_labels,
        autopct=lambda pct: f'{pct:.0f}%' if pct >= 3 else '',
        startangle=140,
        colors=colors,
        pctdistance=0.85,
        wedgeprops=dict(width=0.8, edgecolor='w')
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.setp(autotexts, size=16, weight="bold", color="white")
    plt.setp(texts, size=16)


def _load_with_names(config):
    """Load a level CSV and attach the ProvinceName grouping column."""
    if not os.path.exists(config['data_path']):
        return None
    df = ec.load_terms(os.path.dirname(config['data_path']), config['level'])
    df['CountyName'] = (df[config['county_col']].astype(str).str.strip().str.zfill(2)
                        .map(ec.PROVINCE_NAME).fillna('Unknown'))
    return df


# --- MAIN EXECUTION ---

def run_analysis(input_dir, base_output_dir):
    """
    Main function to generate the 2x3 grid of pie charts.
    """
    print(f"\n{'='*60}\nGenerating Province Contribution Pie Charts\n{'='*60}")

    LEVELS_TO_ANALYZE = levels_to_analyze(input_dir)
    OUTPUT_DIR = os.path.join(base_output_dir, 'county_contribution_pies')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    for row_idx, (level_name, config) in enumerate(LEVELS_TO_ANALYZE.items()):
        print(f"\nProcessing {level_name} level...")

        df = _load_with_names(config)
        if df is None:
            print(f"  Data file not found: {config['data_path']}. Skipping.")
            for col_idx in range(3):
                axes[row_idx, col_idx].axis('off')
            continue

        analysis_types = ['Positive', 'Negative', 'Absolute']
        for col_idx, analysis_type in enumerate(analysis_types):
            ax = axes[row_idx, col_idx]
            sel_col = config['sel_pop_col']  # Using population selection as requested

            if sel_col not in df.columns:
                print(f"  Selection column '{sel_col}' not found. Skipping.")
                ax.axis('off')
                continue

            # Filter data for pos/neg/abs
            if analysis_type == 'Positive':
                subset_df = df[df[sel_col] > 0]
                contribution = subset_df.groupby('CountyName')[sel_col].sum()
            elif analysis_type == 'Negative':
                subset_df = df[df[sel_col] < 0]
                contribution = subset_df.groupby('CountyName')[sel_col].apply(lambda x: x.abs().sum())
            else:  # Absolute
                contribution = df.groupby('CountyName')[sel_col].apply(lambda x: x.abs().sum())

            contribution = contribution[contribution > 0]

            if contribution.empty:
                print(f"  No {analysis_type} selection data for {level_name}. Skipping pie chart.")
                ax.axis('off')
            else:
                title = f'{level_name} Level\n{analysis_type} Population Selection'
                create_pie_chart(ax, contribution, title, custom_cmap=PURPLE_CMAP)

    # Final adjustments and save
    fig.suptitle('Share of Population-Path Selection Effect by Province', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = 'county_selection_contribution_summary.pdf'
    full_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPie chart summary saved to {full_path}")

    # --- FIGURE 2: SELECTION DENSITY (PER CAPITA) ---
    print(f"\n{'='*60}\nGenerating Province Selection Density Pie Charts\n{'='*60}")

    fig2, axes2 = plt.subplots(2, 3, figsize=(24, 16))

    for row_idx, (level_name, config) in enumerate(LEVELS_TO_ANALYZE.items()):
        print(f"\nProcessing {level_name} level for density charts...")

        df = _load_with_names(config)
        if df is None:
            for col_idx in range(3):
                axes2[row_idx, col_idx].axis('off')
            continue

        # Calculate total initial population (firm stock) per province
        county_pops = df.groupby('CountyName')[config['pop_col']].sum()

        analysis_types = ['Positive', 'Negative', 'Absolute']
        for col_idx, analysis_type in enumerate(analysis_types):
            ax = axes2[row_idx, col_idx]
            sel_col = config['sel_pop_col']

            if sel_col not in df.columns:
                ax.axis('off')
                continue

            if analysis_type == 'Positive':
                subset_df = df[df[sel_col] > 0]
                county_sel = subset_df.groupby('CountyName')[sel_col].sum()
            elif analysis_type == 'Negative':
                subset_df = df[df[sel_col] < 0]
                county_sel = subset_df.groupby('CountyName')[sel_col].apply(lambda x: x.abs().sum())
            else:  # Absolute
                county_sel = df.groupby('CountyName')[sel_col].apply(lambda x: x.abs().sum())

            # Calculate per-capita (per-firm) selection density
            selection_density = county_sel / county_pops
            selection_density = selection_density[selection_density > 0].dropna()

            if selection_density.empty:
                print(f"  No selection density data for {level_name} - {analysis_type}. Skipping.")
                ax.axis('off')
            else:
                title = f'{level_name} Level\n{analysis_type} Population Selection Density'
                create_pie_chart(ax, selection_density, title, custom_cmap=ORANGE_CMAP)

    fig2.suptitle('Share of Per-Firm Selection Effect (Density) by Province', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename2 = 'county_selection_density_summary.pdf'
    full_path2 = os.path.join(OUTPUT_DIR, filename2)
    plt.savefig(full_path2, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSelection density pie chart summary saved to {full_path2}")

    # --- FIGURE 3: POPULATION SHARE ---
    print(f"\n{'='*60}\nGenerating Province Population Share Pie Charts\n{'='*60}")

    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 8))

    # 1. First, determine the definitive total population from the Canton level data
    #    (the Chicago script used the Tract level).
    canton_config = LEVELS_TO_ANALYZE.get('Canton')
    definitive_county_pops = pd.Series(dtype=float)
    df_can = _load_with_names(canton_config) if canton_config else None
    if df_can is not None:
        definitive_county_pops = df_can.groupby('CountyName')[canton_config['pop_col']].sum()
        definitive_total_pop = definitive_county_pops.sum()
        print(f"  Established definitive total firm stock from Canton level: {definitive_total_pop:,.0f}")
    else:
        definitive_total_pop = 0
        print("  Warning: Could not load Canton level data to establish definitive population total.")

    # Calculate province level pops for comparison table
    province_config = LEVELS_TO_ANALYZE.get('Province')
    community_county_pops = pd.Series(dtype=float)
    df_prov = _load_with_names(province_config) if province_config else None
    if df_prov is not None:
        community_county_pops = df_prov.groupby('CountyName')[province_config['pop_col']].sum()

    # Create and print the comparison DataFrame
    comparison_df = pd.DataFrame({
        'Firms_Canton_Level': definitive_county_pops,
        'Firms_Province_Level': community_county_pops
    }).fillna(0)
    comparison_df['Difference'] = comparison_df['Firms_Canton_Level'] - comparison_df['Firms_Province_Level']

    print("\n--- Province Firm Stock Comparison ---")
    print(comparison_df.to_string(formatters={'Firms_Canton_Level': '{:,.0f}'.format,
                                              'Firms_Province_Level': '{:,.0f}'.format,
                                              'Difference': '{:,.0f}'.format}))
    print("------------------------------------\n")

    # 2. Iterate and plot, using the definitive total for normalization.
    for ax_idx, (level_name, config) in enumerate(LEVELS_TO_ANALYZE.items()):
        print(f"\nProcessing {level_name} level for population share...")
        ax = axes3[ax_idx]

        df = _load_with_names(config)
        if df is None:
            print(f"  Data file not found: {config['data_path']}. Skipping.")
            ax.axis('off')
            continue

        # Total initial firm stock per province for the current level
        county_pops = df.groupby('CountyName')[config['pop_col']].sum()
        county_pops = county_pops[county_pops > 0]

        if county_pops.empty:
            print(f"  No population data for {level_name}. Skipping pie chart.")
            ax.axis('off')
        else:
            # For the province level, check for discrepancies and add an "Unaccounted" slice
            if level_name == 'Province' and definitive_total_pop > 0:
                current_total_pop = county_pops.sum()
                unaccounted_pop = definitive_total_pop - current_total_pop

                print(f"    DEBUG: Definitive Total (from Cantons): {definitive_total_pop:,.2f}")
                print(f"    DEBUG: Current Total (from Provinces): {current_total_pop:,.2f}")
                print(f"    DEBUG: Calculated Unaccounted Population: {unaccounted_pop:,.2f}")

                if unaccounted_pop > 0.01:  # Small threshold to avoid floating point issues
                    print(f"  Adding 'Unaccounted' slice with population: {unaccounted_pop:,.0f}")
                    county_pops['Unaccounted'] = unaccounted_pop

            title = f'{level_name} Level\nFirm Stock Share by Province'
            create_pie_chart(ax, county_pops, title, normalization_total=definitive_total_pop)

    fig3.suptitle('Share of Firm Stock by Province', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename3 = 'county_population_share_summary.pdf'
    full_path3 = os.path.join(OUTPUT_DIR, filename3)
    plt.savefig(full_path3, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPopulation share pie chart summary saved to {full_path3}")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == "__main__":
    main()

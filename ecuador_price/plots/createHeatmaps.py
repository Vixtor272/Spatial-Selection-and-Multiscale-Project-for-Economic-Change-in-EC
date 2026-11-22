"""
createHeatmaps.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/createHeatmaps.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Only the sections that are active in the Chicago script are ported (its long
per-county report loop and the "ultimate plot" for named Chicago communities
are commented out there):

  1. Country-level combined summary heatmap (Chicago: state_summary_combined)
     — one row each for MSS (Population), NS (Population), MSS (Income),
     NS (Income) across the decomposition components.
  2. The same combined summary for the focal province (Chicago: Cook County),
     Pichincha by default, using self-normalized NS.
  3. MSS variation summary heatmaps (weighted mean/std of Total Selection
     Dominance) for the whole country and for the focal province.

Level mapping: bg->parish, tr->canton, cm->province, st->country;
Cook County -> focal province.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import re
import sys

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

# --- Configuration ---
HIERARCHY_LEVELS = ec.HIERARCHY_LEVELS          # ['parish','canton','province','country']
BASE_ANALYSIS_LEVEL = ec.BASE_ANALYSIS_LEVEL    # 'parish'
CUSTOM_PURPLE = ec.CUSTOM_PURPLE
CUSTOM_ORANGE = ec.CUSTOM_ORANGE
CUSTOM_GREY = ec.CUSTOM_GREY

# Defines which column in the *child* level DataFrame identifies its parent.
CHILD_TO_PARENT_ID_COL_MAP = {
    'canton': 'ParentProvince',
    'province': 'ParentCountry',
}


def clean_column_names(col_name):
    """Creates human-readable labels for heatmap columns from the raw column names."""
    # First, strip any metric suffix like _mss or _ns
    col_name_base = re.sub(r'(_mss|_GIR|_ns)$', '', col_name)

    sel_match = re.match(r"Sel_(\w+?)_from_(\w+)", col_name_base)
    trans_sel_match = re.match(r"Transmitted_Sel_(\w+?)_to_(\w+)", col_name_base)
    trans_avgg_match = re.match(r"Transmitted_AvgG_(\w+?)_to_(\w+)", col_name_base)
    trans_agg_match = re.match(r"Transmitted_AggG_(?:pop_|inc_|den_)?(\w+?)_to_(\w+)", col_name_base)

    if trans_sel_match:
        origin = trans_sel_match.group(1).upper()
        return f"Transmitted Sel. ({origin})"
    elif trans_avgg_match:
        base = trans_avgg_match.group(1).upper()
        return f"Transmitted Growth ({base})"
    elif trans_agg_match:
        origin = trans_agg_match.group(1).upper()
        return f"Transmitted Growth ({origin})"
    elif sel_match:
        level = sel_match.group(1).upper()
        return f"Selection ({level})"

    return col_name_base.replace('_', ' ').title()


def prepare_mss_heatmap_data(df_for_level, path_suffix):
    """
    Extracts pre-calculated MSS columns (which are absolute percentages)
    and prepares the data for heatmap plotting.
    """
    if df_for_level.empty:
        return None

    mss_cols = sorted([col for col in df_for_level.columns if col.endswith(f'{path_suffix}_mss')])

    if not mss_cols:
        return None

    df_indexed = df_for_level.set_index('UnitName')

    df_mss = df_indexed[mss_cols].copy()

    clean_names = [clean_column_names(col) for col in df_mss.columns]
    df_mss.columns = clean_names

    # Explicitly rename the generic 'Selection (LVL)' to 'Selection'
    sel_col_to_rename = [col for col in df_mss.columns if col.startswith('Selection (')]
    if sel_col_to_rename:
        df_mss = df_mss.rename(columns={sel_col_to_rename[0]: 'Selection'})

    # --- Reorder columns to place Transmitted Growth last ---
    cols = df_mss.columns.tolist()
    trans_popwth_col = None
    for col in cols:
        if "Transmitted Growth" in col:
            trans_popwth_col = col
            break
    if trans_popwth_col:
        cols.remove(trans_popwth_col)
        cols.append(trans_popwth_col)
        df_mss = df_mss[cols]

    return df_mss


def prepare_ns_heatmap_data(df_for_level, target_level, path_suffix, hierarchy_levels_list, self_normalize=False):
    """
    Prepares a DataFrame with NS values for plotting.
    If self_normalize is True, it uses NS values relative to the unit's own total growth.
    Otherwise, it defaults to using values relative to the immediate parent.
    """
    if df_for_level.empty:
        return None

    if self_normalize:
        ns_suffix_for_filter = "_ns"
        print(f"  Note (prepare_ns_heatmap_data): Using self-normalized NS values for '{target_level}'.")
    else:
        try:
            current_level_idx = hierarchy_levels_list.index(target_level)
            if current_level_idx == len(hierarchy_levels_list) - 1:
                parent_level_for_ns = target_level
            else:
                parent_level_for_ns = hierarchy_levels_list[current_level_idx + 1]
        except ValueError:
            print(f"  Error (prepare_ns_heatmap_data): Target level '{target_level}' not in hierarchy.")
            return None

        ns_suffix_for_filter = f"_ns_{parent_level_for_ns}" if parent_level_for_ns != target_level else "_ns"

    ns_cols_regex = f"^(Sel|Transmitted).+({path_suffix}){ns_suffix_for_filter}$"

    ns_cols_to_use = [col for col in df_for_level.columns if re.match(ns_cols_regex, col)]

    if not ns_cols_to_use:
        print(f"  Warning (prepare_ns_heatmap_data): No NS columns matching regex "
              f"'{ns_cols_regex}' found for path '{path_suffix}' in level '{target_level}'.")
        return None

    ns_cols_to_use = sorted(list(set(ns_cols_to_use)))

    if not ns_cols_to_use or 'UnitName' not in df_for_level.columns:
        return None

    df_ns = df_for_level[['UnitName'] + ns_cols_to_use].set_index('UnitName')

    clean_names = [clean_column_names(re.sub(f"{ns_suffix_for_filter}$", "", col))
                   for col in df_ns.columns]
    df_ns.columns = clean_names

    sel_col_to_rename = [col for col in df_ns.columns if col.startswith('Selection (')]
    if sel_col_to_rename:
        df_ns = df_ns.rename(columns={sel_col_to_rename[0]: 'Selection'})

    # --- Reorder columns to place Transmitted Growth last ---
    cols = df_ns.columns.tolist()
    trans_popwth_col = None
    for col in cols:
        if "Transmitted Growth" in col:
            trans_popwth_col = col
            break
    if trans_popwth_col:
        cols.remove(trans_popwth_col)
        cols.append(trans_popwth_col)
        df_ns = df_ns[cols]

    return df_ns


def get_custom_colormap_and_bounds(df):
    """
    Creates a custom colormap and determines bounds based on selection terms.
    The colormap uses purple-white-orange for values within the bounds of selection
    terms and grey for values outside those bounds (e.g., transmitted growth).
    """
    trans_popwth_cols = [col for col in df.columns if 'Transmitted Growth' in col]
    selection_cols = [col for col in df.columns if col not in trans_popwth_cols]

    if selection_cols:
        v_bound = df[selection_cols].abs().max().max()
    else:
        v_bound = df.abs().max().max()

    if pd.isna(v_bound) or v_bound == 0:
        v_bound = 1.0

    vmin, vmax = -v_bound, v_bound

    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_purple_orange_grey",
        [CUSTOM_PURPLE, "white", CUSTOM_ORANGE]
    )
    custom_cmap.set_over(CUSTOM_GREY)
    custom_cmap.set_under(CUSTOM_GREY)

    return custom_cmap, vmin, vmax


def _plot_summary_heatmap(df_to_plot, title, output_path, cmap, vmin, vmax):
    """A private helper function to generate and save a summary heatmap with consistent styling."""
    n_rows, n_cols = df_to_plot.shape
    figsize = (2.0 * n_cols, 1.2 * n_rows)

    fig, ax = plt.subplots(figsize=figsize)

    annotations = np.array([["{:.1f}%".format(val) for val in row] for row in df_to_plot.values])

    sns.heatmap(
        df_to_plot,
        annot=annotations,
        fmt="",
        cmap=cmap,
        linewidths=.5,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        annot_kws={"size": 25}
    )

    ax.set_title(title, fontsize=18)

    ax.set_xlabel("")
    ax.set_ylabel("")
    # Label rows and columns from the DataFrame itself (MultiIndex rows are
    # joined into one line, e.g. ('Population', 'Mean') -> 'Population Mean').
    xlabels = ['\n'.join(map(str, c)) if isinstance(c, tuple) else str(c)
               for c in df_to_plot.columns]
    ylabels = [' '.join(map(str, r)) if isinstance(r, tuple) else str(r)
               for r in df_to_plot.index]
    ax.set_xticklabels(xlabels, rotation=0, fontsize=10)
    ax.set_yticklabels(ylabels, rotation=0, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def create_combined_summary_plot(df, level, output_dir, file_name, plot_title):
    """
    Creates a single, consolidated heatmap for a given region, showing MSS and NS
    for both population and income paths.
    """
    print(f"  Generating combined summary plot for {plot_title} (saving to {output_dir})...")
    os.makedirs(output_dir, exist_ok=True)

    # Self-normalize NS for the focal-province summary (Chicago: Cook County)
    self_normalize_ns = ec.focal_province_name() in plot_title

    # 1. Prepare the data series (pop = Price headline, inc = income path,
    #    den = density path: trait = firm count)
    ldr_pop = prepare_mss_heatmap_data(df, '_pop')
    ldr_inc = prepare_mss_heatmap_data(df, '_inc')
    ldr_den = prepare_mss_heatmap_data(df, '_den')
    pnc_pop = prepare_ns_heatmap_data(df, level, '_pop', HIERARCHY_LEVELS, self_normalize=self_normalize_ns)
    pnc_inc = prepare_ns_heatmap_data(df, level, '_inc', HIERARCHY_LEVELS, self_normalize=self_normalize_ns)
    pnc_den = prepare_ns_heatmap_data(df, level, '_den', HIERARCHY_LEVELS, self_normalize=self_normalize_ns)

    data_series = {
        'MSS (Population)': ldr_pop.iloc[0] if ldr_pop is not None and not ldr_pop.empty else pd.Series(dtype=float),
        'NS (Population)': pnc_pop.iloc[0] if pnc_pop is not None and not pnc_pop.empty else pd.Series(dtype=float),
        'MSS (Income)': ldr_inc.iloc[0] if ldr_inc is not None and not ldr_inc.empty else pd.Series(dtype=float),
        'NS (Income)': pnc_inc.iloc[0] if pnc_inc is not None and not pnc_inc.empty else pd.Series(dtype=float),
        'MSS (Density)': ldr_den.iloc[0] if ldr_den is not None and not ldr_den.empty else pd.Series(dtype=float),
        'NS (Density)': pnc_den.iloc[0] if pnc_den is not None and not pnc_den.empty else pd.Series(dtype=float),
    }

    # 2. Combine into a single DataFrame and clean up
    combined_df = pd.DataFrame(data_series).T.fillna(0)

    if combined_df.empty or combined_df.shape[1] == 0:
        print(f"    Cannot create combined plot for {plot_title}; not enough data.")
        return

    # --- Reorder columns sequentially by spatial scale ---
    canonical_order = [
        'Transmitted Sel. (CANTON)',
        'Transmitted Sel. (PROVINCE)',
        'Selection',
        'Transmitted Growth (PARISH)'
    ]
    final_order = [col for col in canonical_order if col in combined_df.columns]
    combined_df = combined_df[final_order]

    # Column labels that name the children each covariance is computed over
    # ('Selection' is the unit's own term, over its immediate children).
    sel_children = {'country': 'provinces', 'province': 'cantons'}.get(level, 'children')
    display_names = {
        'Transmitted Sel. (CANTON)': 'Sel. among parishes\n(in cantons)',
        'Transmitted Sel. (PROVINCE)': 'Sel. among cantons\n(in provinces)',
        'Selection': f'Sel. among\n{sel_children}',
        'Transmitted Growth (PARISH)': 'Transmission\n(within-parish)',
    }
    combined_df = combined_df.rename(columns=display_names)

    # 3. Get colormap and bounds based on selection terms from the raw data
    cmap, vmin, vmax = get_custom_colormap_and_bounds(combined_df)
    output_filename = os.path.join(output_dir, file_name)

    # 4. Plotting (delegated to helper)
    _plot_summary_heatmap(
        df_to_plot=combined_df,
        title=plot_title,
        output_path=output_filename,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    print(f"    Saved {output_filename}")


def create_mss_variation_summary_plot(decomposition_results, output_dir, file_name, plot_title,
                                      levels_to_plot, cook_county_only=False):
    """
    Creates a heatmap summarizing the mean and standard deviation of the "Total
    Selection Dominance" across different spatial levels.  Total Selection
    Dominance for a unit is the sum of the MSS of its selection-based growth
    components.
    """
    focal_code = ec.FOCAL_PROVINCE_CODE
    summary_rows = []

    for level in levels_to_plot:
        df_level = decomposition_results.get(level)
        if df_level is None or df_level.empty:
            continue
        df_level = df_level.copy()

        # Filter for the focal province if requested (Chicago: Cook County)
        if cook_county_only:
            if level == 'canton':
                if 'ParentProvince' in df_level.columns:
                    mask = df_level['ParentProvince'].astype(str).str.zfill(2) == focal_code
                    df_level = df_level[mask].copy()
            elif level == 'province':
                if 'UnitName' in df_level.columns:
                    df_level = df_level[df_level['UnitName'].astype(str).str.zfill(2) == focal_code].copy()
            else:  # 'country' level is skipped for the focal-province plot
                continue

        if df_level.empty:
            continue

        level_summary = {'Level': level.capitalize()}

        for path_suffix, path_name in [('_pop', 'Population'), ('_inc', 'Income'),
                                       ('_den', 'Density')]:
            mss_sel_cols = [
                c for c in df_level.columns
                if (c.startswith('Sel_') or c.startswith('Transmitted_Sel_'))
                and c.endswith(f'{path_suffix}_mss')
            ]

            if not mss_sel_cols:
                level_summary[f'{path_name}_Mean'] = np.nan
                level_summary[f'{path_name}_Std'] = np.nan
                continue

            # For each unit, sum the MSS of its selection terms
            df_level['TotalSelMSS'] = df_level[mss_sel_cols].sum(axis=1)

            pop_col = f'PopInitial_{level}'
            weights = df_level[pop_col] if pop_col in df_level.columns else None

            if len(df_level) == 1:
                mean = df_level['TotalSelMSS'].iloc[0]
                std = 0.0
            elif weights is not None and not weights.isnull().all() and weights.sum() > 0:
                mean = np.average(df_level['TotalSelMSS'], weights=weights)
                variance = np.average((df_level['TotalSelMSS'] - mean)**2, weights=weights)
                std = np.sqrt(variance)
            else:
                mean = df_level['TotalSelMSS'].mean()
                std = df_level['TotalSelMSS'].std()

            level_summary[f'{path_name}_Mean'] = mean
            level_summary[f'{path_name}_Std'] = std

        summary_rows.append(level_summary)

    if not summary_rows:
        print(f"Cannot generate MSS variation plot for {plot_title}, no data found.")
        return

    df_plot = pd.DataFrame(summary_rows).set_index('Level').T

    df_plot.index = pd.MultiIndex.from_tuples(
        [(name.split('_')[0], name.split('_')[1]) for name in df_plot.index],
        names=['Path', 'Metric']
    )

    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_white_orange',
        ['white', CUSTOM_ORANGE]
    )
    output_path = os.path.join(output_dir, file_name)

    _plot_summary_heatmap(
        df_to_plot=df_plot,
        title=plot_title,
        output_path=output_path,
        cmap=custom_cmap,
        vmin=0,
        vmax=None
    )

    print(f"  Successfully saved MSS variation summary to {output_path}")


def run_analysis(input_dir, base_output_dir):
    """
    Main function to orchestrate loading data and generating heatmaps.
    """
    OUTPUT_DIR = os.path.join(base_output_dir, "heatmaps_local_dominance")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n--- Running analysis for input: {input_dir} ---")
    print(f"--- Saving plots to: {base_output_dir} ---")

    print("--- Starting Plot Generation ---")

    decomposition_results = ec.load_pickle(input_dir)
    if decomposition_results is None:
        return

    focal_code = ec.FOCAL_PROVINCE_CODE
    focal_name = ec.focal_province_name()

    # --- Country-level Summary Plots (Chicago: State-level) ---
    print("\n--- Generating Country-level Summary Reports ---")
    try:
        df_st_full = decomposition_results[BASE_ANALYSIS_LEVEL]['country']
        if not df_st_full.empty:
            state_summary_dir = os.path.join(OUTPUT_DIR, 'country_summary')
            create_combined_summary_plot(
                df=df_st_full,
                level='country',
                output_dir=state_summary_dir,
                file_name='country_summary_combined.pdf',
                plot_title='Country-Level Decomposition Summary'
            )
        else:
            print("  Country-level data is empty. Skipping country summary plot.")

        # Focal province plot (Chicago: Cook County)
        df_ct_full = decomposition_results[BASE_ANALYSIS_LEVEL]['province']
        df_cook_county = df_ct_full[df_ct_full['UnitName'].astype(str).str.zfill(2) == focal_code].copy()
        if not df_cook_county.empty:
            cook_county_summary_dir = os.path.join(OUTPUT_DIR, 'country_summary',
                                                   focal_name.lower().replace(' ', '_'))
            create_combined_summary_plot(
                df=df_cook_county,
                level='province',
                output_dir=cook_county_summary_dir,
                file_name=f'{focal_name.lower().replace(" ", "_")}_summary_combined.pdf',
                plot_title=f'{focal_name} Decomposition Summary'
            )
        else:
            print(f"  Could not find the focal province ('{focal_code}') in the data. Skipping specific plot.")

    except (KeyError, TypeError) as e:
        print(f"  Could not generate country-level summary reports: {e}")

    # --- MSS Variation Summary Plots ---
    print("\n--- Generating MSS Variation Summary Plots ---")
    try:
        # Country-level summary
        create_mss_variation_summary_plot(
            decomposition_results=decomposition_results[BASE_ANALYSIS_LEVEL],
            output_dir=os.path.join(OUTPUT_DIR, 'country_summary'),
            file_name='country_ldr_variation_summary.pdf',
            plot_title='Country-Level MSS Selection Dominance Variation',
            levels_to_plot=['canton', 'province', 'country'],
            cook_county_only=False
        )

        # Focal province summary
        create_mss_variation_summary_plot(
            decomposition_results=decomposition_results[BASE_ANALYSIS_LEVEL],
            output_dir=os.path.join(OUTPUT_DIR, 'country_summary',
                                    focal_name.lower().replace(' ', '_')),
            file_name=f'{focal_name.lower().replace(" ", "_")}_ldr_variation_summary.pdf',
            plot_title=f'{focal_name} MSS Selection Dominance Variation',
            levels_to_plot=['canton', 'province'],
            cook_county_only=True
        )
    except (KeyError, TypeError) as e:
        print(f"  Could not generate MSS variation summary reports: {e}")

    print("\n--- Plot Generation Complete ---")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == "__main__":
    main()

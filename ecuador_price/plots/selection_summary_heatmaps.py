"""
selection_summary_heatmaps.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/selection_summary_heatmaps.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Two summary heatmaps (population path and income path), each 3 rows × 4 columns
(Chicago had 5 columns; Ecuador has one fewer level).

Columns (canonical order, matching country_summary_combined):
    CANTON | PROVINCE | COUNTRY | PARISH

Rows:
    Weighted Mean MSS  — population-weighted mean of |Sel_level| / |AvgG_country| × 100
                         across all units at each level.  PARISH empty.

    Weighted Std MSS   — population-weighted std dev of the same.  PARISH empty.

    NS                 — each column's Price-decomposition term / AvgG_country × 100.
                         These four values sum to 100%.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

BASE_ANALYSIS_LEVEL = ec.BASE_ANALYSIS_LEVEL  # 'parish'

# ── Style (identical to createHeatmaps.py / country_summary_combined) ─────────

CUSTOM_PURPLE = ec.CUSTOM_PURPLE
CUSTOM_ORANGE = ec.CUSTOM_ORANGE
CUSTOM_GREY = ec.CUSTOM_GREY

# Canonical column order: selection terms first, base-level growth last
CANONICAL_COLS = ['CANTON', 'PROVINCE', 'COUNTRY', 'PARISH']
TRANSMITTED_GROWTH_COL = 'PARISH'   # rendered last; excluded from colormap bounds

# ── Column definitions ────────────────────────────────────────────────────────

# For MSS rows: own-level selection col and level key, per display column and path.
MSS_SEL_COL = {
    'CANTON': {'_pop': ('canton', 'Sel_canton_from_parish_pop'),
               '_inc': ('canton', 'Sel_canton_from_parish_inc'),
               '_den': ('canton', 'Sel_canton_from_parish_den')},
    'PROVINCE': {'_pop': ('province', 'Sel_province_from_canton_pop'),
                 '_inc': ('province', 'Sel_province_from_canton_inc'),
                 '_den': ('province', 'Sel_province_from_canton_den')},
    'COUNTRY': {'_pop': ('country', 'Sel_country_from_province_pop'),
                '_inc': ('country', 'Sel_country_from_province_inc'),
                '_den': ('country', 'Sel_country_from_province_den')},
}

# For NS row: country-level Price decomposition terms (sum / AvgG_country = 100%).
NS_COLS = {
    '_pop': {
        'CANTON': 'Transmitted_Sel_canton_to_country_pop',
        'PROVINCE': 'Transmitted_Sel_province_to_country_pop',
        'COUNTRY': 'Sel_country_from_province_pop',
        'PARISH': 'Transmitted_AvgG_parish_to_country_pop',
    },
    '_inc': {
        'CANTON': 'Transmitted_Sel_canton_to_country_inc',
        'PROVINCE': 'Transmitted_Sel_province_to_country_inc',
        'COUNTRY': 'Sel_country_from_province_inc',
        'PARISH': 'Transmitted_AvgG_parish_to_country_inc',
    },
    '_den': {
        'CANTON': 'Transmitted_Sel_canton_to_country_den',
        'PROVINCE': 'Transmitted_Sel_province_to_country_den',
        'COUNTRY': 'Sel_country_from_province_den',
        'PARISH': 'Transmitted_AvgG_parish_to_country_den',
    },
}

MSA_GROWTH_COL = {'_pop': 'AvgG_pop_country', '_inc': 'AvgG_inc_country',
                  '_den': 'AvgG_den_country'}

# ── Row calculations ──────────────────────────────────────────────────────────


def _weighted_mean_std(series, weights):
    if weights is None or weights.isnull().all() or weights.sum() == 0:
        return series.mean(), series.std()
    if len(series) == 1:
        return series.iloc[0], 0.0
    mean = np.average(series, weights=weights)
    var = np.average((series - mean) ** 2, weights=weights)
    return mean, np.sqrt(var)


def _mss_mean_std(df_level, level_key, sel_col, msa_growth):
    if sel_col not in df_level.columns or pd.isna(msa_growth) or msa_growth == 0:
        return np.nan, np.nan
    mss = (df_level[sel_col].abs() / abs(msa_growth)) * 100
    pop_col = f'PopInitial_{level_key}'
    weights = df_level[pop_col] if pop_col in df_level.columns else None
    return _weighted_mean_std(mss, weights)


# ── Build 3×4 DataFrame ───────────────────────────────────────────────────────

def build_plot_df(bg_results, path_suffix):
    df_st = bg_results.get('country', pd.DataFrame())
    msa_col = MSA_GROWTH_COL[path_suffix]
    msa_growth = df_st[msa_col].iloc[0] \
        if not df_st.empty and msa_col in df_st.columns else np.nan

    mean_row = {}
    std_row = {}
    ns_row = {}

    for col_label in CANONICAL_COLS:
        # NS row
        ns_col = NS_COLS[path_suffix].get(col_label)
        if ns_col and not df_st.empty and ns_col in df_st.columns \
                and not pd.isna(msa_growth) and msa_growth != 0:
            ns_row[col_label] = (df_st[ns_col].iloc[0] / msa_growth) * 100
        else:
            ns_row[col_label] = 0.0

        # MSS rows — PARISH (base) has no own-level selection term
        if col_label == TRANSMITTED_GROWTH_COL:
            mean_row[col_label] = 0.0
            std_row[col_label] = 0.0
            continue

        level_key, sel_col = MSS_SEL_COL[col_label][path_suffix]
        df_level = bg_results.get(level_key, pd.DataFrame())
        if df_level.empty:
            mean_row[col_label] = 0.0
            std_row[col_label] = 0.0
            continue

        m, s = _mss_mean_std(df_level, level_key, sel_col, msa_growth)
        mean_row[col_label] = m if not pd.isna(m) else 0.0
        std_row[col_label] = s if not pd.isna(s) else 0.0

    df = pd.DataFrame(
        [mean_row, std_row, ns_row],
        index=['Weighted Mean MSS', 'Weighted Std MSS', 'NS'],
        columns=CANONICAL_COLS,
    )
    return df


# ── Colormap (matches get_custom_colormap_and_bounds in createHeatmaps.py) ────

def _colormap_and_bounds(df):
    # Bounds from selection columns only — exclude the transmitted growth column
    sel_cols = [c for c in df.columns if c != TRANSMITTED_GROWTH_COL]
    v_bound = df[sel_cols].abs().max().max() if sel_cols else df.abs().max().max()
    if pd.isna(v_bound) or v_bound == 0:
        v_bound = 1.0
    cmap = LinearSegmentedColormap.from_list(
        'custom_purple_orange_grey', [CUSTOM_PURPLE, 'white', CUSTOM_ORANGE]
    )
    cmap.set_over(CUSTOM_GREY)
    cmap.set_under(CUSTOM_GREY)
    return cmap, -v_bound, v_bound


# ── Plot (matches _plot_summary_heatmap in createHeatmaps.py exactly) ─────────

COL_LABELS = ['can', 'prov', 'ec', 'trans']

ROW_LABELS = {
    '_pop': [r'$\langle\rho^p_j\rangle$', r'$\sigma(\rho^p_j)$', r'$\omega^p_j$'],
    '_inc': [r'$\langle\rho^i_j\rangle$', r'$\sigma(\rho^i_j)$', r'$\omega^i_j$'],
    '_den': [r'$\langle\rho^n_j\rangle$', r'$\sigma(\rho^n_j)$', r'$\omega^n_j$'],
}


def plot_summary_heatmap(df, title, output_path, path_suffix='_pop'):
    n_rows, n_cols = df.shape
    figsize = (2.0 * n_cols, 1.2 * n_rows)

    cmap, vmin, vmax = _colormap_and_bounds(df)

    annotations = np.array(
        [['{:.1f}%'.format(val) for val in row] for row in df.values]
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df,
        annot=annotations,
        fmt='',
        cmap=cmap,
        linewidths=0.5,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        annot_kws={'size': 25},
    )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(COL_LABELS, fontsize=16)
    ax.set_yticklabels(ROW_LABELS[path_suffix], fontsize=16, rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_analysis(input_dir, base_output_dir):
    results = ec.load_pickle(input_dir)
    if results is None:
        return

    OUTPUT_DIR = os.path.join(base_output_dir, 'heatmaps_local_dominance', 'country_summary')

    bg_results = results.get(BASE_ANALYSIS_LEVEL, {})

    for path_suffix, label in [('_pop', 'Population'), ('_inc', 'Income'),
                               ('_den', 'Density')]:
        df_plot = build_plot_df(bg_results, path_suffix)

        print(f"\n{label} path:")
        print(df_plot.to_string())
        print(f"  NS row sum: {df_plot.loc['NS'].sum():.4f}%")

        fname = f'selection_summary_{label.lower()}.pdf'
        plot_summary_heatmap(
            df_plot,
            title=f'Selection Summary — {label} Path',
            output_path=os.path.join(OUTPUT_DIR, fname),
            path_suffix=path_suffix,
        )

    print("\nDone.")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == '__main__':
    main()

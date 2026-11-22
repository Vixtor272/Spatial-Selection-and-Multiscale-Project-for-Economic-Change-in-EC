#!/usr/bin/env python3
"""
gini_scatter.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/gini_scatter.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Scatter plots of income NS components vs the change in the within-unit Gini
coefficient, with population-weighted (WLS) fits, bifurcated by median initial
income (purple = below/eq. median, orange = above), plus a weighted-average
Gini change report and a summary heatmap.

Chicago -> Ecuador:
    Gini across BGs within community    -> parishes within province  (gini_change_bg)
    Gini across tracts within community -> cantons within province   (gini_change_tr)
    Gini across BGs within tract        -> parishes within canton    (gini_change_bg_by_tract)
    County summary row (never computed in Chicago) -> country row,
        parishes nationally (gini_change_bg_by_country), computed here.

Incomes are per-firm mean sales exp(LogAvgInc)/firms weighted by firm counts
(Chicago's exp(LogAvgInc) was already a per-household mean).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import warnings
import statsmodels.api as sm
import seaborn as sns

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)
warnings.filterwarnings('ignore')

# Custom colors matching other plots
custom_purple = ec.CUSTOM_PURPLE
custom_orange = ec.CUSTOM_ORANGE
outline_color = ec.OUTLINE_COLOR
purple_dot = ec.PURPLE_DOT
orange_dot = ec.ORANGE_DOT


def load_data(input_dir):
    """Load province, parish, canton and country data (Chicago: cm, bg, tr, ct)"""
    try:
        cm_data = ec.load_terms(input_dir, 'province')
        cm_data = ec.add_country_ns(cm_data, input_dir)
        print(f"Successfully loaded province data: {len(cm_data)} rows")

        bg_data = ec.load_terms(input_dir, 'parish')
        print(f"Successfully loaded parish data: {len(bg_data)} rows")

        tr_data = ec.load_terms(input_dir, 'canton')
        tr_data = ec.add_country_ns(tr_data, input_dir)
        print(f"Successfully loaded canton data: {len(tr_data)} rows")

        ct_data = ec.load_terms(input_dir, 'country')
        print(f"Successfully loaded country data: {len(ct_data)} rows")

        return cm_data, bg_data, tr_data, ct_data
    except Exception as e:
        print(f"Error loading data from {input_dir}: {e}")
        return None, None, None, None


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


def prepare_income_data(data, level_suffix, group_col):
    """
    Prepare a level's data for Gini calculation: per-firm mean sales
    (exp(log total sales)/firms) and firm-count weights, grouped by group_col.
    """
    print(f"Preparing {level_suffix} income data (grouped by {group_col})...")

    log_initial_income_col = f'LogAvgIncInitial_{level_suffix}'
    log_final_income_col = f'LogAvgIncFinal_{level_suffix}'

    if log_initial_income_col not in data.columns or log_final_income_col not in data.columns:
        print(f"Warning: Could not find log income columns. Available columns: {data.columns.tolist()}")
        return None

    prepared = data.copy()

    if group_col not in prepared.columns:
        print(f"Error: {group_col} column not found in {level_suffix} data")
        return None

    pop_initial_col = f'PopInitial_{level_suffix}'
    pop_final_col = f'PopFinal_{level_suffix}'

    prepared['initial_income'] = np.exp(prepared[log_initial_income_col]) / prepared[pop_initial_col]
    prepared['final_income'] = np.exp(prepared[log_final_income_col]) / prepared[pop_final_col]

    if pop_initial_col in prepared.columns:
        prepared['weight'] = prepared[pop_initial_col]
    else:
        prepared['weight'] = 1

    result = prepared[[group_col, 'initial_income', 'final_income', 'weight']].copy()

    result = result.dropna(subset=[group_col])
    result = result[(result['initial_income'] > 0) & (result['final_income'] > 0)]

    print(f"Prepared {len(result)} {level_suffix} records for Gini calculation")
    return result


def calculate_gini_changes(income_data, group_by_col):
    """
    Calculate Gini coefficient change for each group defined by group_by_col.
    """
    print(f"Calculating Gini coefficients by {group_by_col}...")

    results = []

    if income_data is None or group_by_col not in income_data.columns:
        print(f"Error: Cannot calculate Gini changes. Input data is None or missing "
              f"group_by column '{group_by_col}'.")
        return pd.DataFrame()

    for group_name, group in income_data.groupby(group_by_col):
        if len(group) < 2:
            continue

        initial_gini = calculate_gini_coefficient(
            group['initial_income'].values,
            group['weight'].values
        )

        final_gini = calculate_gini_coefficient(
            group['final_income'].values,
            group['weight'].values
        )

        gini_change = final_gini - initial_gini if pd.notna(final_gini) and pd.notna(initial_gini) else np.nan

        results.append({
            group_by_col: group_name,
            'initial_gini': initial_gini,
            'final_gini': final_gini,
            'gini_change': gini_change,
            'num_members': len(group)
        })

    result_df = pd.DataFrame(results)
    print(f"Calculated Gini changes for {len(result_df)} groups.")

    if not result_df.empty:
        valid_changes = result_df['gini_change'].dropna()
        if len(valid_changes) > 0:
            print("Gini change statistics:")
            print(f"  Mean: {valid_changes.mean():.4f}")
            print(f"  Std: {valid_changes.std():.4f}")
            print(f"  Min: {valid_changes.min():.4f}")
            print(f"  Max: {valid_changes.max():.4f}")

    return result_df


def extract_cumulative_income_ns(cm_data, path='inc', label='income'):
    """Extract cumulative NS_ec from province data for one path (inc or den)"""
    print(f"Extracting cumulative {label} NS_ec...")

    transmitted_col = f'Transmitted_Sel_canton_to_province_{path}_ns_country'
    selection_col = f'Sel_province_from_canton_{path}_ns_country'

    if transmitted_col not in cm_data.columns or selection_col not in cm_data.columns:
        print("Warning: Expected NS columns not found.")
        return None

    transmitted_vals = cm_data[transmitted_col].fillna(0)
    selection_vals = cm_data[selection_col].fillna(0)

    cumulative_ns = transmitted_vals + selection_vals

    print(f"Cumulative {label} NS_ec statistics:")
    print(f"  Mean: {cumulative_ns.mean():.4f}   Std: {cumulative_ns.std():.4f}")
    print(f"  Min: {cumulative_ns.min():.4f}   Max: {cumulative_ns.max():.4f}")

    return cumulative_ns


def extract_transmitted_income_ns(cm_data, path='inc', label='income'):
    """Extract transmitted NS_ec from province data for one path (inc or den)"""
    transmitted_col = f'Transmitted_Sel_canton_to_province_{path}_ns_country'
    if transmitted_col not in cm_data.columns:
        print(f"Warning: Expected NS column not found: {transmitted_col}.")
        return None
    return cm_data[transmitted_col].fillna(0)


def extract_selection_income_ns(cm_data, path='inc', label='income'):
    """Extract selection NS_ec from province data for one path (inc or den)"""
    selection_col = f'Sel_province_from_canton_{path}_ns_country'
    if selection_col not in cm_data.columns:
        print(f"Warning: Expected NS column not found: {selection_col}.")
        return None
    return cm_data[selection_col].fillna(0)


def create_scatter_plot(data, x_col, y_col, color_col, title, xlabel, ylabel, output_filename,
                        size_col=None, color_label='Color Variable', filter_col=None,
                        filter_threshold=None, bifurcate_on_color=False, ylim=None, xlim=None,
                        base_output_dir=None):
    """
    Generates a scatter plot with a weighted linear regression fit and
    confidence interval, bifurcated by the boolean color column.
    """
    # 1. Filter data
    if filter_col and filter_threshold is not None:
        plot_data = data[data[filter_col] >= filter_threshold].copy()
    else:
        plot_data = data.copy()

    if plot_data.empty:
        print(f"Warning: No data to plot for {title} after filtering. Skipping plot.")
        return

    essential_cols = [x_col, y_col, color_col]
    if size_col:
        essential_cols.append(size_col)

    plot_data = plot_data.dropna(subset=essential_cols)

    if plot_data.empty or len(plot_data) < 3:
        print(f"Warning: Not enough data to plot for {title} after dropping NaNs. Skipping plot.")
        return

    # 2. Prepare point sizes
    if size_col and size_col in plot_data.columns:
        pop = plot_data[size_col]
        s_min, s_max = 15, 200
        if pop.min() == pop.max():
            sizes = pd.Series(s_min, index=plot_data.index)
        else:
            sizes = s_min + ((pop - pop.min()) / (pop.max() - pop.min()) * (s_max - s_min))
        plot_data['point_size'] = sizes
    else:
        plot_data['point_size'] = 50

    # 3. Setup plot
    fig, ax = plt.subplots(figsize=(3.25, 2.9))
    fig.patch.set_facecolor('white')

    for spine in ax.spines.values():
        spine.set_edgecolor(outline_color)
        spine.set_linewidth(1.5)

    # 4. Bifurcate data for coloring and fitting
    low_income_data = plot_data[plot_data[color_col] <= 0]
    high_income_data = plot_data[plot_data[color_col] > 0]

    # 5. Scatter plot (bifurcated colors, scaled sizes)
    ax.scatter(low_income_data[x_col], low_income_data[y_col], color=purple_dot, alpha=0.25,
               s=low_income_data['point_size'], edgecolors=outline_color, linewidth=0.8,
               label='Below/Eq. Median Income')
    ax.scatter(high_income_data[x_col], high_income_data[y_col], color=orange_dot, alpha=0.25,
               s=high_income_data['point_size'], edgecolors=outline_color, linewidth=0.8,
               label='Above Median Income')

    # --- Fit Lines and Stats ---
    y = plot_data[y_col]
    X = sm.add_constant(plot_data[[x_col]])
    weights = plot_data[size_col] if size_col else None

    wls_model = sm.WLS(y, X, weights=weights)
    wls_fit = wls_model.fit()

    slope = wls_fit.params[x_col]
    r_squared = wls_fit.rsquared

    stats_text = f'Full Fit:\n  m={slope:.3f}, R²={r_squared:.3f}'

    x_pred_full = pd.DataFrame({x_col: np.linspace(plot_data[x_col].min(), plot_data[x_col].max(), 100)})
    x_pred_full_sm = sm.add_constant(x_pred_full, has_constant='add')
    y_pred_full = wls_fit.predict(x_pred_full_sm)
    ax.plot(x_pred_full[x_col], y_pred_full, color='#3D3D3D', linestyle='-', linewidth=2, label='Full Fit')

    pred_full = wls_fit.get_prediction(x_pred_full_sm)
    ci_full_df = pred_full.summary_frame(alpha=0.05)
    ax.fill_between(x_pred_full[x_col], ci_full_df['mean_ci_lower'], ci_full_df['mean_ci_upper'],
                    color='#B0B0B0', alpha=0.3)

    if bifurcate_on_color:
        # Fit for the above-median subset
        if len(high_income_data) > 2:
            y_high = high_income_data[y_col]
            X_high = sm.add_constant(high_income_data[[x_col]])
            weights_high = high_income_data[size_col] if size_col else None

            wls_model_high = sm.WLS(y_high, X_high, weights=weights_high)
            wls_fit_high = wls_model_high.fit()

            x_pred_high = pd.DataFrame({x_col: np.linspace(high_income_data[x_col].min(),
                                                           high_income_data[x_col].max(), 100)})
            x_pred_high_sm = sm.add_constant(x_pred_high, has_constant='add')
            y_pred_high = wls_fit_high.predict(x_pred_high_sm)
            ax.plot(x_pred_high[x_col], y_pred_high, color=custom_orange, linestyle='--', linewidth=2)

            pred_high = wls_fit_high.get_prediction(x_pred_high_sm)
            ci_high_df = pred_high.summary_frame(alpha=0.05)
            ax.fill_between(x_pred_high[x_col], ci_high_df['mean_ci_lower'], ci_high_df['mean_ci_upper'],
                            color=custom_orange, alpha=0.2)

            stats_text += f'\nAbove Median Fit:\n  m={wls_fit_high.params[x_col]:.3f}, R²={wls_fit_high.rsquared:.3f}'

        # Fit for the below-median subset
        if len(low_income_data) > 2:
            y_low = low_income_data[y_col]
            X_low = sm.add_constant(low_income_data[[x_col]])
            weights_low = low_income_data[size_col] if size_col else None

            wls_model_low = sm.WLS(y_low, X_low, weights=weights_low)
            wls_fit_low = wls_model_low.fit()

            x_pred_low = pd.DataFrame({x_col: np.linspace(low_income_data[x_col].min(),
                                                          low_income_data[x_col].max(), 100)})
            x_pred_low_sm = sm.add_constant(x_pred_low, has_constant='add')
            y_pred_low = wls_fit_low.predict(x_pred_low_sm)
            ax.plot(x_pred_low[x_col], y_pred_low, color=custom_purple, linestyle='--', linewidth=2)

            pred_low = wls_fit_low.get_prediction(x_pred_low_sm)
            ci_low_df = pred_low.summary_frame(alpha=0.05)
            ax.fill_between(x_pred_low[x_col], ci_low_df['mean_ci_lower'], ci_low_df['mean_ci_upper'],
                            color=custom_purple, alpha=0.2)

            stats_text += f'\nBelow/Eq. Median Fit:\n  m={wls_fit_low.params[x_col]:.3f}, R²={wls_fit_low.rsquared:.3f}'

    print(stats_text)

    # 8. Reference lines and style
    ax.axhline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CCCCCC')
    ax.tick_params(axis='x', which='major', labelsize=12, colors='#333333')
    ax.tick_params(axis='y', which='major', labelsize=12, colors='#333333')

    ax.set_xlabel(xlabel, fontsize=10, color='#333333')
    ax.set_ylabel(ylabel, fontsize=10, color='#333333')
    ax.legend(fontsize=6, loc='best', framealpha=0.8)

    # 10. Save figure
    output_dir = os.path.join(base_output_dir, "gini_scatter_plots")
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, output_filename)

    plt.tight_layout()
    plt.savefig(full_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot saved to {full_path}")


def create_gini_summary_heatmap(all_tr_data, all_cm_data, all_ct_data, output_dir):
    """
    Calculates weighted mean and std dev of Gini change for each level and
    plots them as a summary heatmap.
    (Chicago rows Tracts/Communities/Counties -> Cantons/Provinces/Country.)
    """
    print("\n--- Generating Gini Change Summary Heatmap ---")
    summary_data = {}

    def get_stats(df, level_name):
        level_map = {
            'Cantons': ('gini_change_bg_by_tract', 'PopInitial_canton'),
            'Provinces': ('gini_change_bg', 'PopInitial_province'),
            'Country': ('gini_change_bg_by_country', 'PopInitial_country'),
        }
        if df is None or df.empty:
            return {'Mean': np.nan, 'Std Dev': np.nan}

        gini_col, pop_col = level_map[level_name]
        if gini_col not in df.columns or pop_col not in df.columns:
            return {'Mean': np.nan, 'Std Dev': np.nan}

        df_filtered = df[[gini_col, pop_col]].dropna()
        if df_filtered.empty or df_filtered[pop_col].sum() == 0:
            return {'Mean': np.nan, 'Std Dev': np.nan}

        mean = np.average(df_filtered[gini_col], weights=df_filtered[pop_col])
        variance = np.average((df_filtered[gini_col] - mean)**2, weights=df_filtered[pop_col])
        std_dev = np.sqrt(variance)

        return {'Mean': mean, 'Std Dev': std_dev}

    summary_data['Cantons'] = get_stats(all_tr_data, 'Cantons')
    summary_data['Provinces'] = get_stats(all_cm_data, 'Provinces')
    summary_data['Country'] = get_stats(all_ct_data, 'Country')

    df_plot = pd.DataFrame(summary_data).T

    if df_plot.isnull().all().all():
        print("Cannot generate Gini summary heatmap; not enough data.")
        return

    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_white_purple', ['white', custom_purple])
    output_path = os.path.join(output_dir, 'gini_change_summary_heatmap.pdf')

    _plot_gini_summary_heatmap(
        df_to_plot=df_plot.T,
        title='Gini Change Summary by Level',
        output_path=output_path,
        cmap=custom_cmap,
        vmin=None,
        vmax=None
    )


def _plot_gini_summary_heatmap(df_to_plot, title, output_path, cmap, vmin, vmax):
    """A private helper function to generate and save a summary heatmap for Gini stats."""
    n_rows, n_cols = df_to_plot.shape
    figsize = (2.8 * n_cols, 1.6 * n_rows)

    fig, ax = plt.subplots(figsize=figsize)

    annotations = np.array([["{:.4f}".format(val) for val in row] for row in df_to_plot.values])

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
        annot_kws={"size": 20}
    )

    ax.set_title(title, fontsize=18)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(df_to_plot.columns, rotation=0, fontsize=14)
    ax.set_yticklabels(df_to_plot.index, rotation=0, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Successfully saved Gini summary heatmap to {output_path}")


def report_weighted_avg_gini_change(df, level_name):
    """
    Calculates and prints the population-weighted average of Gini change for a given level.
    """
    if df is None or df.empty:
        print(f"Cannot report weighted average Gini change for {level_name}s: Data is not available.")
        return

    level_map = {
        'canton': ('gini_change_bg_by_tract', 'PopInitial_canton'),
        'province': ('gini_change_bg', 'PopInitial_province'),
        'country': ('gini_change_bg_by_country', 'PopInitial_country'),
    }

    if level_name not in level_map:
        print(f"Error: Unknown level '{level_name}' for Gini change reporting.")
        return

    gini_col, pop_col = level_map[level_name]

    if gini_col not in df.columns or pop_col not in df.columns:
        print(f"Cannot report weighted average Gini change for {level_name}s: "
              f"Missing required columns ('{gini_col}' or '{pop_col}').")
        return

    df_filtered = df[[gini_col, pop_col]].dropna()

    if df_filtered.empty:
        print(f"Cannot report weighted average Gini change for {level_name}s: No valid data after cleaning.")
        return

    total_population = df_filtered[pop_col].sum()

    if total_population == 0:
        print(f"Cannot report weighted average Gini change for {level_name}s: Total population is zero.")
        return

    weighted_avg = np.average(df_filtered[gini_col], weights=df_filtered[pop_col])

    print(f"Population-weighted average change in Gini across all {level_name}s: {weighted_avg:.6f}")


def run_analysis(input_dir, base_output_dir):
    # Load data
    cm_data, bg_data, tr_data, ct_data = load_data(input_dir)
    if cm_data is None:
        return

    # --- Gini Calculations (Province-level; Chicago: community-level) ---
    bg_income_province_grouped = prepare_income_data(bg_data, 'parish', 'ParentProvince')
    gini_change_bg_by_province = calculate_gini_changes(bg_income_province_grouped, 'ParentProvince')
    if not gini_change_bg_by_province.empty:
        gini_change_bg_by_province = gini_change_bg_by_province.rename(
            columns={'gini_change': 'gini_change_bg', 'ParentProvince': 'UnitName'})

    tr_income_province_grouped = prepare_income_data(tr_data, 'canton', 'ParentProvince')
    gini_change_tr_by_province = calculate_gini_changes(tr_income_province_grouped, 'ParentProvince')
    if not gini_change_tr_by_province.empty:
        gini_change_tr_by_province = gini_change_tr_by_province.rename(
            columns={'gini_change': 'gini_change_tr', 'ParentProvince': 'UnitName'})

    # --- Gini Calculations (Canton-level; Chicago: tract-level) ---
    bg_income_canton_grouped = prepare_income_data(bg_data, 'parish', 'ParentCanton')
    gini_change_bg_by_canton = calculate_gini_changes(bg_income_canton_grouped, 'ParentCanton')
    if not gini_change_bg_by_canton.empty:
        gini_change_bg_by_canton = gini_change_bg_by_canton.rename(
            columns={'gini_change': 'gini_change_bg_by_tract', 'ParentCanton': 'UnitName'})

    # --- Gini Calculations (Country-level; Chicago never computed its county row) ---
    bg_income_country_grouped = prepare_income_data(bg_data, 'parish', 'ParentCountry')
    gini_change_bg_by_country = calculate_gini_changes(bg_income_country_grouped, 'ParentCountry')
    if not gini_change_bg_by_country.empty and ct_data is not None:
        gini_change_bg_by_country = gini_change_bg_by_country.rename(
            columns={'gini_change': 'gini_change_bg_by_country', 'ParentCountry': 'UnitName'})
        all_ct_data = pd.merge(ct_data, gini_change_bg_by_country, on='UnitName', how='inner')
    else:
        all_ct_data = None

    # --- Data Merging and Preparation ---
    if gini_change_bg_by_province.empty or gini_change_tr_by_province.empty:
        print("Error: Could not calculate province-level Gini changes. Aborting.")
        return

    all_cm_data = pd.merge(cm_data, gini_change_bg_by_province, on='UnitName', how='inner')
    all_cm_data = pd.merge(all_cm_data, gini_change_tr_by_province, on='UnitName', how='inner')

    # Bifurcation column based on median initial income for province plots
    if 'LogAvgIncInitial_province' in all_cm_data.columns:
        median_income_cm = all_cm_data['LogAvgIncInitial_province'].median()
        all_cm_data['bifurcate_col_cm'] = all_cm_data['LogAvgIncInitial_province'] > median_income_cm
        print(f"Province median initial log income: {median_income_cm:.4f}")
    else:
        print("Warning: 'LogAvgIncInitial_province' not found for province bifurcation.")
        all_cm_data['bifurcate_col_cm'] = False

    if all_cm_data.empty:
        print("Error: No common provinces found after merging all data sources. Cannot create plots.")
        return

    # Merge canton data with the canton-level Gini results
    if not gini_change_bg_by_canton.empty and tr_data is not None:
        all_tr_data = pd.merge(tr_data, gini_change_bg_by_canton, on='UnitName', how='inner')
        if 'LogAvgIncInitial_canton' in all_tr_data.columns:
            median_income_tr = all_tr_data['LogAvgIncInitial_canton'].median()
            all_tr_data['bifurcate_col_tr'] = all_tr_data['LogAvgIncInitial_canton'] > median_income_tr
            print(f"Canton median initial log income: {median_income_tr:.4f}")
        else:
            print("Warning: 'LogAvgIncInitial_canton' not found for canton bifurcation.")
            all_tr_data['bifurcate_col_tr'] = False
    else:
        all_tr_data = pd.DataFrame()
        print("Warning: Could not create merged canton-level data for new plot.")

    # Extract NS components from the merged province data (scaled to fractions)
    all_cm_data['cumulative_income_ns'] = extract_cumulative_income_ns(all_cm_data) / 100
    all_cm_data['transmitted_income_ns'] = extract_transmitted_income_ns(all_cm_data) / 100
    all_cm_data['selection_income_ns'] = extract_selection_income_ns(all_cm_data) / 100

    # Den-path analogs (density: trait = firm count; NS is dimensionless)
    cum_den = extract_cumulative_income_ns(all_cm_data, path='den', label='density')
    trans_den = extract_transmitted_income_ns(all_cm_data, path='den', label='density')
    sel_den = extract_selection_income_ns(all_cm_data, path='den', label='density')
    have_den = cum_den is not None and trans_den is not None and sel_den is not None
    if have_den:
        all_cm_data['cumulative_den_ns'] = cum_den / 100
        all_cm_data['transmitted_den_ns'] = trans_den / 100
        all_cm_data['selection_den_ns'] = sel_den / 100

    output_dir = os.path.join(base_output_dir, "gini_scatter_plots")
    os.makedirs(output_dir, exist_ok=True)

    if 'PopInitial_province' not in all_cm_data.columns:
        print("Warning: 'PopInitial_province' not found in data. Regressions will not be weighted.")
        size_col_cm = None
    else:
        size_col_cm = 'PopInitial_province'

    plot_configs = [
        {
            'data': all_cm_data,
            'x': 'cumulative_income_ns',
            'y': 'gini_change_bg',
            'title': 'Cumulative NS vs. Parish Gini Change',
            'xlabel': 'Cumulative Income NS (Province)',
            'ylabel': 'Change in Parish Gini Coefficient',
            'color': 'bifurcate_col_cm',
            'output': 'cumulative_ns_vs_bg_gini.pdf',
            'filter_col': 'cumulative_income_ns',
            'filter_threshold': -500,
            'bifurcate_on_color': True,
            'size_col': size_col_cm,
        },
        {
            'data': all_cm_data,
            'x': 'cumulative_income_ns',
            'y': 'gini_change_tr',
            'title': 'Cumulative NS vs. Canton Gini Change',
            'xlabel': 'Cumulative Income NS (Province)',
            'ylabel': 'Change in Canton Gini Coefficient',
            'color': 'bifurcate_col_cm',
            'output': 'cumulative_ns_vs_tr_gini.pdf',
            'filter_col': 'cumulative_income_ns',
            'filter_threshold': -500,
            'bifurcate_on_color': True,
            'size_col': size_col_cm,
        },
        {
            'data': all_cm_data,
            'x': 'transmitted_income_ns',
            'y': 'gini_change_bg',
            'title': 'Transmitted NS vs. Parish Gini Change',
            'xlabel': 'Transmitted Income NS (Province)',
            'ylabel': 'Change in Parish Gini Coefficient',
            'color': 'bifurcate_col_cm',
            'output': 'transmitted_ns_vs_bg_gini.pdf',
            'filter_col': 'cumulative_income_ns',
            'filter_threshold': -500,
            'bifurcate_on_color': True,
            'size_col': size_col_cm,
        },
        {
            'data': all_cm_data,
            'x': 'selection_income_ns',
            'y': 'gini_change_tr',
            'title': 'Selection NS vs. Canton Gini Change',
            'xlabel': 'Selection Income NS (Province)',
            'ylabel': 'Change in Canton Gini Coefficient',
            'color': 'bifurcate_col_cm',
            'output': 'selection_ns_vs_tr_gini.pdf',
            'filter_col': 'cumulative_income_ns',
            'filter_threshold': -500,
            'bifurcate_on_color': True,
            'size_col': size_col_cm,
        }
    ]

    # Den-path analogs of the four province-level plots
    if have_den:
        plot_configs.extend([
            {
                'data': all_cm_data,
                'x': 'cumulative_den_ns',
                'y': 'gini_change_bg',
                'title': 'Cumulative Density NS vs. Parish Gini Change',
                'xlabel': 'Cumulative Density NS (Province)',
                'ylabel': 'Change in Parish Gini Coefficient',
                'color': 'bifurcate_col_cm',
                'output': 'cumulative_den_ns_vs_bg_gini.pdf',
                'filter_col': 'cumulative_den_ns',
                'filter_threshold': -500,
                'bifurcate_on_color': True,
                'size_col': size_col_cm,
            },
            {
                'data': all_cm_data,
                'x': 'cumulative_den_ns',
                'y': 'gini_change_tr',
                'title': 'Cumulative Density NS vs. Canton Gini Change',
                'xlabel': 'Cumulative Density NS (Province)',
                'ylabel': 'Change in Canton Gini Coefficient',
                'color': 'bifurcate_col_cm',
                'output': 'cumulative_den_ns_vs_tr_gini.pdf',
                'filter_col': 'cumulative_den_ns',
                'filter_threshold': -500,
                'bifurcate_on_color': True,
                'size_col': size_col_cm,
            },
            {
                'data': all_cm_data,
                'x': 'transmitted_den_ns',
                'y': 'gini_change_bg',
                'title': 'Transmitted Density NS vs. Parish Gini Change',
                'xlabel': 'Transmitted Density NS (Province)',
                'ylabel': 'Change in Parish Gini Coefficient',
                'color': 'bifurcate_col_cm',
                'output': 'transmitted_den_ns_vs_bg_gini.pdf',
                'filter_col': 'cumulative_den_ns',
                'filter_threshold': -500,
                'bifurcate_on_color': True,
                'size_col': size_col_cm,
            },
            {
                'data': all_cm_data,
                'x': 'selection_den_ns',
                'y': 'gini_change_tr',
                'title': 'Selection Density NS vs. Canton Gini Change',
                'xlabel': 'Selection Density NS (Province)',
                'ylabel': 'Change in Canton Gini Coefficient',
                'color': 'bifurcate_col_cm',
                'output': 'selection_den_ns_vs_tr_gini.pdf',
                'filter_col': 'cumulative_den_ns',
                'filter_threshold': -500,
                'bifurcate_on_color': True,
                'size_col': size_col_cm,
            },
        ])

    # Canton-level plots (Chicago: tract-level)
    if not all_tr_data.empty:
        if 'PopInitial_canton' not in all_tr_data.columns:
            print("Warning: 'PopInitial_canton' not found in canton data. New plots will not be weighted.")
            size_col_tr = None
        else:
            size_col_tr = 'PopInitial_canton'

        # Canton Selection vs. Change in Parish Gini
        selection_col_tr = 'Sel_canton_from_parish_inc'
        if selection_col_tr in all_tr_data.columns and 'gini_change_bg_by_tract' in all_tr_data.columns:
            plot_configs.append({
                'data': all_tr_data,
                'x': selection_col_tr,
                'y': 'gini_change_bg_by_tract',
                'title': 'Canton Selection vs. Change in Parish Gini',
                'xlabel': 'Selection Term (Canton, from Parishes)',
                'ylabel': 'Change in Gini Coefficient (from Parishes)',
                'color': 'bifurcate_col_tr',
                'output': 'tract_selection_vs_bg_gini.pdf',
                'filter_col': None,
                'filter_threshold': None,
                'bifurcate_on_color': True,
                'size_col': size_col_tr,
            })
        else:
            print(f"Warning: Skipping 'Canton Selection vs. Change in Parish Gini' plot. "
                  f"Missing columns: '{selection_col_tr}' or 'gini_change_bg_by_tract'.")

        # Canton Selection NS vs. Change in Parish Gini
        selection_ns_col_tr = 'Sel_canton_from_parish_inc_ns_country'
        if selection_ns_col_tr in all_tr_data.columns and 'gini_change_bg_by_tract' in all_tr_data.columns:
            all_tr_data['selection_ns_st_tr'] = all_tr_data[selection_ns_col_tr] / 100

            plot_configs.append({
                'data': all_tr_data,
                'x': 'selection_ns_st_tr',
                'y': 'gini_change_bg_by_tract',
                'title': 'Canton Selection NS vs. Change in Parish Gini',
                'xlabel': 'Selection NS (Canton, from Parishes, Country-Normalized)',
                'ylabel': 'Change in Gini Coefficient (from Parishes)',
                'color': 'bifurcate_col_tr',
                'output': 'tract_selection_ns_vs_bg_gini.pdf',
                'filter_col': None,
                'filter_threshold': None,
                'bifurcate_on_color': True,
                'size_col': size_col_tr,
            })
        else:
            print(f"Warning: Skipping 'Canton Selection NS vs. Change in Parish Gini' plot. "
                  f"Missing columns: '{selection_ns_col_tr}' or 'gini_change_bg_by_tract'.")

        # Den-path analogs of the two canton-level plots
        selection_den_col_tr = 'Sel_canton_from_parish_den'
        if selection_den_col_tr in all_tr_data.columns and 'gini_change_bg_by_tract' in all_tr_data.columns:
            plot_configs.append({
                'data': all_tr_data,
                'x': selection_den_col_tr,
                'y': 'gini_change_bg_by_tract',
                'title': 'Canton Density Selection vs. Change in Parish Gini',
                'xlabel': 'Density Selection Term (Canton, from Parishes, firms/yr)',
                'ylabel': 'Change in Gini Coefficient (from Parishes)',
                'color': 'bifurcate_col_tr',
                'output': 'tract_den_selection_vs_bg_gini.pdf',
                'filter_col': None,
                'filter_threshold': None,
                'bifurcate_on_color': True,
                'size_col': size_col_tr,
            })

        selection_den_ns_col_tr = 'Sel_canton_from_parish_den_ns_country'
        if selection_den_ns_col_tr in all_tr_data.columns and 'gini_change_bg_by_tract' in all_tr_data.columns:
            all_tr_data['selection_den_ns_st_tr'] = all_tr_data[selection_den_ns_col_tr] / 100
            plot_configs.append({
                'data': all_tr_data,
                'x': 'selection_den_ns_st_tr',
                'y': 'gini_change_bg_by_tract',
                'title': 'Canton Density Selection NS vs. Change in Parish Gini',
                'xlabel': 'Density Selection NS (Canton, from Parishes, Country-Normalized)',
                'ylabel': 'Change in Gini Coefficient (from Parishes)',
                'color': 'bifurcate_col_tr',
                'output': 'tract_den_selection_ns_vs_bg_gini.pdf',
                'filter_col': None,
                'filter_threshold': None,
                'bifurcate_on_color': True,
                'size_col': size_col_tr,
            })

    # --- Report population-weighted average Gini changes ---
    print("\n--- Population-Weighted Average Gini Change Summary ---")
    report_weighted_avg_gini_change(all_tr_data, 'canton')
    report_weighted_avg_gini_change(all_cm_data, 'province')
    report_weighted_avg_gini_change(all_ct_data, 'country')

    # Create the summary heatmap
    create_gini_summary_heatmap(all_tr_data, all_cm_data, all_ct_data, output_dir)

    # Generate plots
    for config in plot_configs:
        create_scatter_plot(
            data=config['data'],
            x_col=config['x'],
            y_col=config['y'],
            color_col=config['color'],
            title=config['title'],
            xlabel=config['xlabel'],
            ylabel=config['ylabel'],
            output_filename=config['output'],
            base_output_dir=base_output_dir,
            filter_col=config['filter_col'],
            filter_threshold=config['filter_threshold'],
            bifurcate_on_color=config.get("bifurcate_on_color", False),
            size_col=config.get("size_col"),
        )


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == "__main__":
    main()

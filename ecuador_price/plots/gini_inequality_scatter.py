#!/usr/bin/env python3
"""
gini_inequality_scatter.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/gini_inequality_scatter.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Creates scatter plots of cumulative income NS_ec vs change in Gini coefficient.

Chicago: Gini across block groups / tracts within each community.
Ecuador:  Gini across parishes / cantons within each province.

X-axis: Cumulative Income NS_ec (province level, country-normalized)
Y-axis: Change in Gini coefficient (final_gini - initial_gini) per province
Color:  Relative population (firm-count) growth with the purple-white-orange map

Income adaptation: Chicago's exp(LogAvgInc) is a per-household mean income;
the Ecuador trait is log *total* sales, so incomes here are per-firm mean
sales exp(LogAvgInc)/firms, weighted by the firm count.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import warnings
from scipy import stats
import sys

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

warnings.filterwarnings('ignore')

# Custom colors matching other plots
custom_purple = ec.CUSTOM_PURPLE
custom_orange = ec.CUSTOM_ORANGE
outline_color = ec.OUTLINE_COLOR


def load_data(input_dir):
    """Load province, parish, and canton data (Chicago: cm, bg, tr)"""
    try:
        cm_data = ec.load_terms(input_dir, 'province')
        cm_data = ec.add_country_ns(cm_data, input_dir)
        print(f"Successfully loaded province data: {len(cm_data)} rows")

        bg_data = ec.load_terms(input_dir, 'parish')
        print(f"Successfully loaded parish data: {len(bg_data)} rows")

        tr_data = ec.load_terms(input_dir, 'canton')
        print(f"Successfully loaded canton data: {len(tr_data)} rows")

        return cm_data, bg_data, tr_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def calculate_gini_coefficient(incomes, weights=None):
    """
    Calculate Gini coefficient for a distribution of incomes.

    Args:
        incomes: Array of income values
        weights: Array of weights (firm counts) for each income value

    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
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


def _prepare_income_data(data, level_suffix, group_col):
    """Shared preparation: per-firm mean sales and firm-count weights."""
    log_initial_income_col = f'LogAvgIncInitial_{level_suffix}'
    log_final_income_col = f'LogAvgIncFinal_{level_suffix}'

    if log_initial_income_col not in data.columns or log_final_income_col not in data.columns:
        print(f"Warning: Could not find log income columns. Available columns: {data.columns.tolist()}")
        return None

    print(f"Using log initial income column: {log_initial_income_col}")
    print(f"Using log final income column: {log_final_income_col}")

    prepared = data.copy()

    if group_col not in prepared.columns:
        print(f"Error: {group_col} column not found in data")
        return None

    pop_initial_col = f'PopInitial_{level_suffix}'
    pop_final_col = f'PopFinal_{level_suffix}'

    # Per-firm mean sales: exp(log total sales) / firm count
    prepared['initial_income'] = np.exp(prepared[log_initial_income_col]) / prepared[pop_initial_col]
    prepared['final_income'] = np.exp(prepared[log_final_income_col]) / prepared[pop_final_col]

    if pop_initial_col in prepared.columns:
        prepared['weight'] = prepared[pop_initial_col]
        print(f"Using weight column: {pop_initial_col}")
    else:
        print("Warning: No population column found, using equal weights")
        prepared['weight'] = 1

    result = prepared[[group_col, 'initial_income', 'final_income', 'weight']].copy()

    result = result.dropna(subset=[group_col])
    result = result[(result['initial_income'] > 0) & (result['final_income'] > 0)]

    print(f"Prepared {len(result)} records for Gini calculation")
    return result


def prepare_bg_income_data(bg_data):
    """Parish incomes grouped by province (Chicago: block groups by community)."""
    print("Preparing parish income data...")
    return _prepare_income_data(bg_data, 'parish', 'ParentProvince')


def prepare_tr_income_data(tr_data):
    """Canton incomes grouped by province (Chicago: tracts by community)."""
    print("Preparing canton income data...")
    return _prepare_income_data(tr_data, 'canton', 'ParentProvince')


def calculate_community_gini_changes(income_data):
    """
    Calculate Gini coefficient change for each province.
    """
    print("Calculating Gini coefficients by province...")

    results = []

    for community, group in income_data.groupby('ParentProvince'):
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
            'ParentProvince': community,
            'initial_gini': initial_gini,
            'final_gini': final_gini,
            'gini_change': gini_change,
            'num_units': len(group)
        })

    result_df = pd.DataFrame(results)
    print(f"Calculated Gini changes for {len(result_df)} provinces")

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
        print(f"Warning: Expected NS columns not found. Available columns: {cm_data.columns.tolist()}")
        return None

    transmitted_vals = cm_data[transmitted_col].fillna(0)
    selection_vals = cm_data[selection_col].fillna(0)

    cumulative_ns = transmitted_vals + selection_vals

    print(f"Cumulative {label} NS_ec statistics:")
    print(f"  Mean: {cumulative_ns.mean():.4f}")
    print(f"  Std: {cumulative_ns.std():.4f}")
    print(f"  Min: {cumulative_ns.min():.4f}")
    print(f"  Max: {cumulative_ns.max():.4f}")

    return cumulative_ns


def extract_transmitted_income_ns(cm_data, path='inc', label='income'):
    """Extract transmitted NS_ec from province data for one path (inc or den)"""
    print(f"Extracting transmitted {label} NS_ec...")

    transmitted_col = f'Transmitted_Sel_canton_to_province_{path}_ns_country'

    if transmitted_col not in cm_data.columns:
        print(f"Warning: Expected NS column not found: {transmitted_col}.")
        return None

    transmitted_vals = cm_data[transmitted_col].fillna(0)

    print(f"Transmitted {label} NS_ec statistics:")
    print(f"  Mean: {transmitted_vals.mean():.4f}")
    print(f"  Std: {transmitted_vals.std():.4f}")

    return transmitted_vals


def extract_selection_income_ns(cm_data, path='inc', label='income'):
    """Extract selection NS_ec from province data for one path (inc or den)"""
    print(f"Extracting selection {label} NS_ec...")

    selection_col = f'Sel_province_from_canton_{path}_ns_country'

    if selection_col not in cm_data.columns:
        print(f"Warning: Expected NS column not found: {selection_col}.")
        return None

    selection_vals = cm_data[selection_col].fillna(0)

    print(f"Selection {label} NS_ec statistics:")
    print(f"  Mean: {selection_vals.mean():.4f}")
    print(f"  Std: {selection_vals.std():.4f}")

    return selection_vals


def create_custom_colormap():
    """Create a custom colormap from purple through white to orange"""
    colors = [custom_purple, 'white', custom_orange]
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return cmap


def _gini_scatter(clean_data, x_col, xlabel, ylabel, filename, base_output_dir):
    """Shared scatter builder (Chicago repeated this block per plot)."""
    if len(clean_data) == 0:
        print("No valid data points to plot!")
        return

    correlation = np.corrcoef(clean_data[x_col], clean_data['gini_change'])[0, 1]
    print(f"Correlation coefficient: {correlation:.6f}")

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        clean_data[x_col], clean_data['gini_change']
    )
    r_squared = r_value**2

    print("Linear regression statistics:")
    print(f"  - Slope: {slope:.6f}")
    print(f"  - R²: {r_squared:.6f}")
    print(f"  - P-value: {p_value:.6f}")

    def calculate_prediction_interval(x, y, x_pred, confidence=0.95):
        """Calculate prediction interval for linear regression"""
        n = len(x)
        x_mean = np.mean(x)

        residuals = y - (slope * x + intercept)
        mse = np.sum(residuals**2) / (n - 2)

        se_pred = np.sqrt(mse * (1 + 1/n + (x_pred - x_mean)**2 / np.sum((x - x_mean)**2)))

        t_val = stats.t.ppf((1 + confidence) / 2, n - 2)

        y_pred = slope * x_pred + intercept
        margin = t_val * se_pred

        return y_pred, y_pred - margin, y_pred + margin

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    custom_cmap = create_custom_colormap()

    color_abs_max = max(abs(clean_data['color_variable'].min()), abs(clean_data['color_variable'].max()))
    vmin = -color_abs_max
    vmax = color_abs_max

    scatter = ax.scatter(clean_data[x_col], clean_data['gini_change'],
                         c=clean_data['color_variable'], cmap=custom_cmap,
                         alpha=0.8, s=50, edgecolors=outline_color, linewidth=0.8,
                         vmin=vmin, vmax=vmax)

    x_range = np.linspace(clean_data[x_col].min(),
                          clean_data[x_col].max(), 100)
    y_pred, y_lower, y_upper = calculate_prediction_interval(
        clean_data[x_col].values,
        clean_data['gini_change'].values,
        x_range
    )

    ax.fill_between(x_range, y_lower, y_upper, alpha=0.2, color='#3D3D3D', label='95% CI')

    ax.plot(x_range, y_pred, color='#3D3D3D', linewidth=2, alpha=0.8, linestyle='-')

    stats_text = f'Slope: {slope:.4f}\nR² = {r_squared:.4f}\np = {p_value:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Relative Population Growth', fontsize=12, color='#333333')
    cbar.ax.tick_params(labelsize=10, colors='#333333')

    ax.axhline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(0, color='#3D3D3D', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=14, color='#333333')
    ax.set_ylabel(ylabel, fontsize=14, color='#333333')

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CCCCCC')

    ax.tick_params(axis='x', which='major', labelsize=12, colors='#333333')
    ax.tick_params(axis='y', which='major', labelsize=12, colors='#333333')

    plt.tight_layout()
    out_dir = os.path.join(base_output_dir, "inequality_scatter_plots")
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {full_path}")


def _clean(merged_data, x_col):
    filtered_data = merged_data[merged_data[x_col] >= -100].copy()
    valid_mask = (np.isfinite(filtered_data[x_col]) &
                  np.isfinite(filtered_data['gini_change']) &
                  np.isfinite(filtered_data['color_variable']) &
                  ~np.isnan(filtered_data[x_col]) &
                  ~np.isnan(filtered_data['gini_change']) &
                  ~np.isnan(filtered_data['color_variable']))
    return filtered_data[valid_mask].copy()


def run_analysis(input_dir, base_output_dir):
    print("Starting Gini inequality scatter analysis...")

    cm_data, bg_data, tr_data = load_data(input_dir)
    if cm_data is None or bg_data is None or tr_data is None:
        print("Error: Could not load required data files.")
        return

    cumulative_income_ns = extract_cumulative_income_ns(cm_data)
    if cumulative_income_ns is None:
        print("Error: Could not extract cumulative income NS_ec.")
        return

    # Population growth for coloring
    pop_growth_col = None
    for col in ['RelPopG_province', 'PopG_province', 'LogAvgIncInitial_province']:
        if col in cm_data.columns:
            pop_growth_col = col
            break

    if pop_growth_col is None:
        print("Warning: Could not find population growth column, using zeros")
        color_variable = pd.Series(0, index=cm_data.index)
    else:
        color_variable = cm_data[pop_growth_col] - cm_data[pop_growth_col].mean()
        print(f"Using color variable: {pop_growth_col} (de-meaned)")

    if 'UnitName' in cm_data.columns:
        cm_plot_data = pd.DataFrame({
            'community': cm_data['UnitName'],
            'cumulative_income_ns': cumulative_income_ns,
            'color_variable': color_variable
        })
    else:
        print("Error: UnitName column not found in province data")
        return

    # === PARISH LEVEL ANALYSIS (Chicago: block group) ===
    print("\n" + "="*60)
    print("PROCESSING PARISH LEVEL DATA")
    print("="*60)

    bg_gini_results = None
    bg_income_data = prepare_bg_income_data(bg_data)
    if bg_income_data is not None:
        bg_gini_results = calculate_community_gini_changes(bg_income_data)
        if len(bg_gini_results) > 0:
            bg_merged_data = pd.merge(
                bg_gini_results,
                cm_plot_data,
                left_on='ParentProvince',
                right_on='community',
                how='inner'
            )

            print(f"Successfully merged parish data for {len(bg_merged_data)} provinces")

            clean = _clean(bg_merged_data, 'cumulative_income_ns')
            print(f"\n{'='*60}\nGINI INEQUALITY SCATTER PLOT\n{'='*60}")
            print(f"Valid data points: {len(clean)}")
            _gini_scatter(clean, 'cumulative_income_ns',
                          'Cumulative Income NS_ec', 'Change in Gini Coefficient',
                          'gini_change_vs_income_ns.pdf', base_output_dir)
        else:
            print("Error: No valid parish Gini calculations.")
    else:
        print("Error: Could not prepare parish income data.")

    # === CANTON LEVEL ANALYSIS (Chicago: tract) ===
    print("\n" + "="*60)
    print("PROCESSING CANTON LEVEL DATA")
    print("="*60)

    tr_gini_results = None
    tr_income_data = prepare_tr_income_data(tr_data)
    if tr_income_data is not None:
        tr_gini_results = calculate_community_gini_changes(tr_income_data)
        if len(tr_gini_results) > 0:
            tr_merged_data = pd.merge(
                tr_gini_results,
                cm_plot_data,
                left_on='ParentProvince',
                right_on='community',
                how='inner'
            )

            print(f"Successfully merged canton data for {len(tr_merged_data)} provinces")

            clean = _clean(tr_merged_data, 'cumulative_income_ns')
            print(f"\n{'='*60}\nCANTON-LEVEL GINI INEQUALITY SCATTER PLOT\n{'='*60}")
            print(f"Valid data points: {len(clean)}")
            _gini_scatter(clean, 'cumulative_income_ns',
                          'Cumulative Income NS_ec', 'Change in Gini Coefficient (Canton Level)',
                          'gini_change_vs_income_ns_tract.pdf', base_output_dir)
        else:
            print("Error: No valid canton Gini calculations.")
    else:
        print("Error: Could not prepare canton income data.")

    # === TRANSMITTED vs PARISH GINI, SELECTION vs CANTON GINI ===
    print("\n" + "="*60)
    print("PROCESSING NEW PLOTS: TRANSMITTED vs PARISH GINI, SELECTION vs CANTON GINI")
    print("="*60)

    cm_data_filtered = cm_data[cumulative_income_ns >= -100].copy()
    print(f"Filtered province data to {len(cm_data_filtered)} rows based on cumulative_income_ns >= -100")

    if pop_growth_col:
        color_variable_filtered = cm_data_filtered[pop_growth_col] - cm_data_filtered[pop_growth_col].mean()
    else:
        color_variable_filtered = pd.Series(0, index=cm_data_filtered.index)

    transmitted_ns = extract_transmitted_income_ns(cm_data_filtered)
    selection_ns = extract_selection_income_ns(cm_data_filtered)

    # PLOT 3: Transmitted vs Parish Gini
    if transmitted_ns is not None and bg_gini_results is not None:
        cm_plot_data_transmitted = pd.DataFrame({
            'community': cm_data_filtered['UnitName'],
            'transmitted_ns': transmitted_ns,
            'color_variable': color_variable_filtered
        })
        transmitted_merged_data = pd.merge(
            bg_gini_results,
            cm_plot_data_transmitted,
            left_on='ParentProvince',
            right_on='community',
            how='inner'
        )
        valid = (np.isfinite(transmitted_merged_data['transmitted_ns']) &
                 np.isfinite(transmitted_merged_data['gini_change']) &
                 np.isfinite(transmitted_merged_data['color_variable']))
        clean = transmitted_merged_data[valid].copy()
        print(f"\n{'='*60}\nTRANSMITTED VS PARISH GINI SCATTER PLOT\n{'='*60}")
        print(f"Valid data points: {len(clean)}")
        _gini_scatter(clean, 'transmitted_ns',
                      'Transmitted Income NS_ec', 'Change in Gini Coefficient (Parish)',
                      'transmitted_vs_bg_gini.pdf', base_output_dir)

    # PLOT 4: Selection vs Canton Gini
    if selection_ns is not None and tr_gini_results is not None:
        cm_plot_data_selection = pd.DataFrame({
            'community': cm_data_filtered['UnitName'],
            'selection_ns': selection_ns,
            'color_variable': color_variable_filtered
        })
        selection_merged_data = pd.merge(
            tr_gini_results,
            cm_plot_data_selection,
            left_on='ParentProvince',
            right_on='community',
            how='inner'
        )
        valid = (np.isfinite(selection_merged_data['selection_ns']) &
                 np.isfinite(selection_merged_data['gini_change']) &
                 np.isfinite(selection_merged_data['color_variable']))
        clean = selection_merged_data[valid].copy()
        print(f"\n{'='*60}\nSELECTION VS CANTON GINI SCATTER PLOT\n{'='*60}")
        print(f"Valid data points: {len(clean)}")
        _gini_scatter(clean, 'selection_ns',
                      'Selection Income NS_ec', 'Change in Gini Coefficient (Canton Level)',
                      'selection_vs_tr_gini.pdf', base_output_dir)

    # === DEN-PATH ANALOGS (density: trait = firm count; NS is dimensionless) ===
    print("\n" + "="*60)
    print("PROCESSING DEN-PATH ANALOG PLOTS")
    print("="*60)

    cumulative_den_ns = extract_cumulative_income_ns(cm_data, path='den', label='density')
    if cumulative_den_ns is not None:
        cm_den_data = pd.DataFrame({
            'community': cm_data['UnitName'],
            'cumulative_den_ns': cumulative_den_ns,
            'color_variable': color_variable
        })
        if bg_gini_results is not None and len(bg_gini_results) > 0:
            merged = pd.merge(bg_gini_results, cm_den_data,
                              left_on='ParentProvince', right_on='community', how='inner')
            clean = _clean(merged, 'cumulative_den_ns')
            print(f"\n{'='*60}\nGINI VS DENSITY NS SCATTER PLOT (PARISH)\n{'='*60}")
            print(f"Valid data points: {len(clean)}")
            _gini_scatter(clean, 'cumulative_den_ns',
                          'Cumulative Density NS_ec', 'Change in Gini Coefficient',
                          'gini_change_vs_density_ns.pdf', base_output_dir)
        if tr_gini_results is not None and len(tr_gini_results) > 0:
            merged = pd.merge(tr_gini_results, cm_den_data,
                              left_on='ParentProvince', right_on='community', how='inner')
            clean = _clean(merged, 'cumulative_den_ns')
            print(f"\n{'='*60}\nGINI VS DENSITY NS SCATTER PLOT (CANTON)\n{'='*60}")
            print(f"Valid data points: {len(clean)}")
            _gini_scatter(clean, 'cumulative_den_ns',
                          'Cumulative Density NS_ec', 'Change in Gini Coefficient (Canton Level)',
                          'gini_change_vs_density_ns_tract.pdf', base_output_dir)

    transmitted_den_ns = extract_transmitted_income_ns(cm_data_filtered, path='den', label='density')
    if transmitted_den_ns is not None and bg_gini_results is not None:
        cm_den_transmitted = pd.DataFrame({
            'community': cm_data_filtered['UnitName'],
            'transmitted_den_ns': transmitted_den_ns,
            'color_variable': color_variable_filtered
        })
        merged = pd.merge(bg_gini_results, cm_den_transmitted,
                          left_on='ParentProvince', right_on='community', how='inner')
        valid = (np.isfinite(merged['transmitted_den_ns']) &
                 np.isfinite(merged['gini_change']) &
                 np.isfinite(merged['color_variable']))
        clean = merged[valid].copy()
        print(f"\n{'='*60}\nTRANSMITTED DENSITY VS PARISH GINI SCATTER PLOT\n{'='*60}")
        print(f"Valid data points: {len(clean)}")
        _gini_scatter(clean, 'transmitted_den_ns',
                      'Transmitted Density NS_ec', 'Change in Gini Coefficient (Parish)',
                      'transmitted_den_vs_bg_gini.pdf', base_output_dir)

    selection_den_ns = extract_selection_income_ns(cm_data_filtered, path='den', label='density')
    if selection_den_ns is not None and tr_gini_results is not None:
        cm_den_selection = pd.DataFrame({
            'community': cm_data_filtered['UnitName'],
            'selection_den_ns': selection_den_ns,
            'color_variable': color_variable_filtered
        })
        merged = pd.merge(tr_gini_results, cm_den_selection,
                          left_on='ParentProvince', right_on='community', how='inner')
        valid = (np.isfinite(merged['selection_den_ns']) &
                 np.isfinite(merged['gini_change']) &
                 np.isfinite(merged['color_variable']))
        clean = merged[valid].copy()
        print(f"\n{'='*60}\nSELECTION DENSITY VS CANTON GINI SCATTER PLOT\n{'='*60}")
        print(f"Valid data points: {len(clean)}")
        _gini_scatter(clean, 'selection_den_ns',
                      'Selection Density NS_ec', 'Change in Gini Coefficient (Canton Level)',
                      'selection_den_vs_tr_gini.pdf', base_output_dir)

    print("\n" + "="*60)
    print("GINI INEQUALITY SCATTER ANALYSIS COMPLETE")
    print("="*60)
    print("Generated plots:")
    print("  - Parish:  inequality_scatter_plots/gini_change_vs_income_ns.pdf")
    print("  - Canton:  inequality_scatter_plots/gini_change_vs_income_ns_tract.pdf")
    print("  - Transmitted vs Parish Gini: inequality_scatter_plots/transmitted_vs_bg_gini.pdf")
    print("  - Selection vs Canton Gini: inequality_scatter_plots/selection_vs_tr_gini.pdf")
    print("  - Den-path analogs: gini_change_vs_density_ns[_tract].pdf, "
          "transmitted_den_vs_bg_gini.pdf, selection_den_vs_tr_gini.pdf")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == "__main__":
    main()

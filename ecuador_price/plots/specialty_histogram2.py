#!/usr/bin/env python3
"""
specialty_histogram2.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/specialty_histogram2.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Plots six histogram families (inc NS, inc MSS, pop NS, pop MSS, den NS, den MSS):
- Province level (Chicago: community) in purple
- Canton level (Chicago: tract) in orange
- Overlaid on the same axes, with the same fits (skew-t via MLE for NS,
  Gamma for MSS), in linear and log-scale variants.

NS is country-normalized (the Chicago NS_st).  The Ecuador export does not
carry *_ns_country at the canton level, so it is derived from the national
growth rate via _ecuador_common.add_country_ns (identical formula).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import sys
from scipy.stats import t, gamma
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ecuador_price.plots import _ecuador_common as ec

plt.rcParams.update(ec.RC_PARAMS)

# Custom colors
custom_purple = ec.CUSTOM_PURPLE
custom_orange = ec.CUSTOM_ORANGE


# --- Extraction helpers ---
def extract_sum_cm(df, trans_col, sel_col, scale=None):
    vals = pd.Series([np.nan]*len(df))
    if trans_col in df.columns and sel_col in df.columns:
        trans = pd.to_numeric(df[trans_col], errors='coerce')
        sel = pd.to_numeric(df[sel_col], errors='coerce')
        vals = trans + sel
        if scale:
            vals = vals / scale
    return vals


def extract_tr(df, sel_col, scale=None):
    if sel_col in df.columns:
        vals = pd.to_numeric(df[sel_col], errors='coerce')
        if scale:
            vals = vals / scale
        return vals
    else:
        return pd.Series([np.nan]*len(df))


# --- Fit helpers ---

# Custom skew-t implementation
def skewt_pdf(x, df, loc, scale, alpha):
    """
    Azzalini's skew-t PDF implementation
    """
    if scale <= 0 or df <= 0:
        return np.full_like(x, np.nan)

    z = (x - loc) / scale
    t_pdf = t.pdf(z, df)
    t_cdf = t.cdf(alpha * z * np.sqrt((df + 1) / (df + z**2)), df + 1)

    return 2 * t_pdf * t_cdf / scale


def _skewt_neg_log_likelihood(params, data):
    """Negative log-likelihood for the skew-t distribution."""
    df, loc, scale, alpha = params
    if scale <= 0 or df <= 0:
        return 1e12
    pdf_vals = skewt_pdf(data, df, loc, scale, alpha)
    pdf_vals = np.maximum(pdf_vals, 1e-300)
    return -np.sum(np.log(pdf_vals))


def fit_skewt(data):
    """
    Fit skew-t distribution via MLE on raw observations.
    df >= 3 ensures finite mean and variance.
    Returns: (df, loc, scale, alpha) or None if fit fails.
    """
    # Zeros represent upstream data unsuitability; remove them along with NaN/inf
    clean_data = data[np.isfinite(data) & ~np.isnan(data) & (data != 0)]
    n_removed = len(data) - len(clean_data)
    if n_removed > 0:
        print(f"    [fit_skewt] Removed {n_removed} values (NaN/inf/zero) from {len(data)} observations")

    if len(clean_data) < 10:
        return None

    x0 = [5.0, np.median(clean_data), np.std(clean_data), 0.0]
    bounds = [(3.0, 50.0),
              (np.min(clean_data), np.max(clean_data)),
              (1e-6, np.std(clean_data) * 5),
              (-10.0, 10.0)]

    best_result = None
    best_nll = np.inf

    for df_init in [3.0, 5.0, 10.0]:
        for alpha_init in [-1.0, 0.0, 1.0]:
            x0_trial = [df_init, np.median(clean_data), np.std(clean_data), alpha_init]
            try:
                result = minimize(_skewt_neg_log_likelihood, x0_trial, args=(clean_data,),
                                  method='L-BFGS-B', bounds=bounds, options={'maxiter': 5000})
                if result.success and result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except Exception:
                continue

    if best_result is not None and best_result.success:
        return tuple(best_result.x)

    # Fallback: try Nelder-Mead (unbounded) with clamping
    try:
        result = minimize(_skewt_neg_log_likelihood, x0, args=(clean_data,),
                          method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-8})
        if result.success:
            df, loc, scale, alpha = result.x
            df = max(3.0, min(df, 50.0))
            scale = max(1e-6, scale)
            return (df, loc, scale, alpha)
    except Exception as e:
        print(f"Skew-t MLE fit failed: {e}")

    return None


# Custom gamma implementation from scratch
def gamma_pdf(x, shape, loc, scale):
    """
    Gamma PDF implementation from scratch
    """
    if shape <= 0 or scale <= 0:
        return np.full_like(x, np.nan)

    x_shifted = x - loc

    result = np.zeros_like(x)
    valid_mask = x_shifted > 0

    if np.any(valid_mask):
        x_valid = x_shifted[valid_mask]

        log_pdf = (shape - 1) * np.log(x_valid) - x_valid / scale - shape * np.log(scale) - np.log(math.gamma(shape))
        result[valid_mask] = np.exp(log_pdf)

    return result


def fit_gamma(data):
    """
    Fit gamma distribution via MLE on raw observations using scipy.stats.gamma.fit.
    Returns: (shape, loc, scale) or None if fit fails.
    """
    clean_data = data[np.isfinite(data) & ~np.isnan(data) & (data > 0)]
    n_removed = len(data) - len(clean_data)
    if n_removed > 0:
        print(f"    [fit_gamma] Removed {n_removed} values (NaN/inf/non-positive) from {len(data)} observations")

    if len(clean_data) < 10:
        return None

    try:
        shape, loc, scale = gamma.fit(clean_data, floc=0)
        return (shape, loc, scale)
    except Exception as e:
        print(f"Gamma MLE fit failed: {e}")
        return None


# --- Plotting ---
def _axis_label(name):
    """X-axis label with units for a plot name like 'Population NS_ec'."""
    if 'NS_ec' in name:
        return name.replace('NS_ec', r'selection NS$_{EC}$ (% of national growth)')
    if 'MSS' in name:
        return name.replace('MSS', r'selection MSS (% of unit gross budget)')
    return name


def plot_hist(cm, tr, name, dist_type, output_dir):
    # Remove NaN, inf, and exactly zero values.
    cm_clean = cm[np.isfinite(cm) & ~np.isnan(cm) & (cm != 0)]
    tr_clean = tr[np.isfinite(tr) & ~np.isnan(tr) & (tr != 0)]
    cm_removed = len(cm) - len(cm_clean)
    tr_removed = len(tr) - len(tr_clean)
    all_data = np.concatenate([cm_clean, tr_clean])

    print(f"\n{'='*60}")
    print(f"PLOT: {name} ({dist_type.upper()} distribution)")
    print(f"{'='*60}")
    if cm_removed > 0:
        print(f"  Province: removed {cm_removed}/{len(cm)} values (NaN/inf/zero)")
    if tr_removed > 0:
        print(f"  Canton: removed {tr_removed}/{len(tr)} values (NaN/inf/zero)")

    print("\nDATA STATISTICS:")
    print("Province data:")
    print(f"  - Count: {len(cm_clean)}")
    if len(cm_clean) > 0:
        cm_mean = cm_clean.mean()
        cm_std = cm_clean.std()
        print(f"  - Mean: {cm_mean:.6f}")
        print(f"  - Std: {cm_std:.6f}")
        print(f"  - Min: {cm_clean.min():.6f}")
        print(f"  - Max: {cm_clean.max():.6f}")
        print(f"  - Median: {np.median(cm_clean):.6f}")

    print("\nCanton data:")
    print(f"  - Count: {len(tr_clean)}")
    if len(tr_clean) > 0:
        tr_mean = tr_clean.mean()
        tr_std = tr_clean.std()
        print(f"  - Mean: {tr_mean:.6f}")
        print(f"  - Std: {tr_std:.6f}")
        print(f"  - Min: {tr_clean.min():.6f}")
        print(f"  - Max: {tr_clean.max():.6f}")
        print(f"  - Median: {np.median(tr_clean):.6f}")

    if len(all_data) == 0:
        print("No valid data to plot!")
        return

    # --- Create figure and axis ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    dark_purple = custom_purple
    dark_orange = custom_orange

    # --- Plot histograms ---
    y_max = 0

    # Province histogram
    if len(cm_clean) > 0:
        n_bins_cm = 2 * (min(30, max(10, int(np.sqrt(len(cm_clean))))) if len(cm_clean) > 0 else 10)
        n_cm, bins_cm, patches_cm = ax.hist(cm_clean, bins=n_bins_cm, alpha=0.7,
                                            color=dark_purple, density=True,
                                            edgecolor='white', linewidth=0.8,
                                            label='Provinces')
        y_max = max(y_max, n_cm.max())
        print(f"\nProvince histogram: {n_bins_cm} bins, max density = {n_cm.max():.6f}")

    # Canton histogram
    if len(tr_clean) > 0:
        n_bins_tr = 3 * (min(75, max(20, int(np.sqrt(len(tr_clean))))) if len(tr_clean) > 0 else 20)
        n_tr, bins_tr, patches_tr = ax.hist(tr_clean, bins=n_bins_tr, alpha=0.7,
                                            color=dark_orange, density=True,
                                            edgecolor='white', linewidth=0.8,
                                            label='Cantons')
        y_max = max(y_max, n_tr.max())
        print(f"Canton histogram: {n_bins_tr} bins, max density = {n_tr.max():.6f}")

    # --- Fit and plot distributions ---
    print("\nFITTING RESULTS:")

    # Province fit
    cm_stats_text = None
    if len(cm_clean) >= 2:
        print(f"\nProvince {dist_type.upper()} fit:")
        if dist_type == 'skewt':
            cm_fit = fit_skewt(cm_clean)
            if cm_fit:
                df, loc, scale, a = cm_fit
                print(f"  - SUCCESS: df={df:.6f}, loc={loc:.6f}, scale={scale:.6f}, skew={a:.6f}")
                x = np.linspace(cm_clean.min(), cm_clean.max(), 200)
                y_fit = skewt_pdf(x, df, loc, scale, a)
                ax.plot(x, y_fit, color=dark_purple, linewidth=2.5)
                cm_stats_text = (f'df = {df:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\n'
                                 f'skew = {a:.3f}\nmean = {cm_clean.mean():.3f}\nmed = {np.median(cm_clean):.3f}')
            else:
                print("  - FAILED: Could not fit skew-t distribution")
        elif dist_type == 'gamma':
            cm_fit = fit_gamma(cm_clean)
            if cm_fit:
                shape, loc, scale = cm_fit
                print(f"  - SUCCESS: shape={shape:.6f}, loc={loc:.6f}, scale={scale:.6f}")
                x = np.linspace(cm_clean.min(), cm_clean.max(), 200)
                y_fit = gamma_pdf(x, shape, loc, scale)
                ax.plot(x, y_fit, color=dark_purple, linewidth=2.5)
                cm_stats_text = (f'shape = {shape:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\n'
                                 f'mean = {cm_clean.mean():.3f}\nmed = {np.median(cm_clean):.3f}')
            else:
                print("  - FAILED: Could not fit gamma distribution")

    # Canton fit
    tr_stats_text = None
    if len(tr_clean) >= 2:
        print(f"\nCanton {dist_type.upper()} fit:")
        if dist_type == 'skewt':
            tr_fit = fit_skewt(tr_clean)
            if tr_fit:
                df, loc, scale, a = tr_fit
                print(f"  - SUCCESS: df={df:.6f}, loc={loc:.6f}, scale={scale:.6f}, skew={a:.6f}")
                x = np.linspace(tr_clean.min(), tr_clean.max(), 200)
                y_fit = skewt_pdf(x, df, loc, scale, a)
                ax.plot(x, y_fit, color=dark_orange, linewidth=2.5)
                tr_stats_text = (f'df = {df:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\n'
                                 f'skew = {a:.3f}\nmean = {tr_clean.mean():.3f}\nmed = {np.median(tr_clean):.3f}')
            else:
                print("  - FAILED: Could not fit skew-t distribution")
        elif dist_type == 'gamma':
            tr_fit = fit_gamma(tr_clean)
            if tr_fit:
                shape, loc, scale = tr_fit
                print(f"  - SUCCESS: shape={shape:.6f}, loc={loc:.6f}, scale={scale:.6f}")
                x = np.linspace(tr_clean.min(), tr_clean.max(), 200)
                y_fit = gamma_pdf(x, shape, loc, scale)
                ax.plot(x, y_fit, color=dark_orange, linewidth=2.5)
                tr_stats_text = (f'shape = {shape:.3f}\nloc = {loc:.3f}\nscale = {scale:.3f}\n'
                                 f'mean = {tr_clean.mean():.3f}\nmed = {np.median(tr_clean):.3f}')
            else:
                print("  - FAILED: Could not fit gamma distribution")

    # --- Mean (solid), median (dashed), and zero reference lines ---
    if len(cm_clean) > 0:
        ax.axvline(cm_clean.mean(), color=dark_purple, linestyle='-', linewidth=1.5, alpha=0.75)
        ax.axvline(np.median(cm_clean), color=dark_purple, linestyle='--', linewidth=1.0, alpha=0.6)
    if len(tr_clean) > 0:
        ax.axvline(tr_clean.mean(), color=dark_orange, linestyle='-', linewidth=1.5, alpha=0.75)
        ax.axvline(np.median(tr_clean), color=dark_orange, linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.75)

    # --- Statistics boxes ---
    if cm_stats_text:
        ax.text(0.7, 0.95, cm_stats_text, transform=ax.transAxes, fontsize=5,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=dark_purple))
    if tr_stats_text:
        ax.text(0.70, 0.80, tr_stats_text, transform=ax.transAxes, fontsize=5,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=dark_orange))

    # --- Axis, ticks, and title formatting ---
    mean_value = all_data.mean()
    std_value = all_data.std()
    if dist_type == 'gamma':
        x_min_limit = 0
    else:
        x_min_limit = mean_value - 2 * std_value
    x_max_limit = mean_value + 2 * std_value
    ax.set_xlim(x_min_limit, x_max_limit)

    ax.set_ylim(bottom=0)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CCCCCC')
    ax.tick_params(axis='x', which='major', labelsize=18, colors='#333333', bottom=True, top=False)
    ax.tick_params(axis='y', which='major', labelsize=18, colors='#333333', left=True, right=False, labelleft=True)
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    ax.set_xlabel(_axis_label(name), fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)

    ax.grid(False)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"specialty_histogram_{name.lower().replace(' ', '_').replace('/', '_')}.pdf")
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fname}")


def plot_hist_log_scale(cm, tr, name, dist_type, output_dir):
    # Log-scale y-axis variant. Same zero-removal rationale as plot_hist.
    cm_clean = cm[np.isfinite(cm) & ~np.isnan(cm) & (cm != 0)]
    tr_clean = tr[np.isfinite(tr) & ~np.isnan(tr) & (tr != 0)]
    all_data = np.concatenate([cm_clean, tr_clean])

    if len(all_data) == 0:
        print(f"No valid data to plot for log-scale {name}!")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    dark_purple = custom_purple
    dark_orange = custom_orange

    y_max = 0

    if len(tr_clean) > 0:
        n_bins_tr = 3 * (min(75, max(20, int(np.sqrt(len(tr_clean))))) if len(tr_clean) > 0 else 20)
        n_tr, _, _ = ax.hist(tr_clean, bins=n_bins_tr, alpha=0.7, color=dark_orange,
                             density=True, edgecolor='white', linewidth=0.8,
                             label='Cantons')
        y_max = max(y_max, n_tr.max())

    if len(cm_clean) > 0:
        n_bins_cm = 2 * (min(30, max(10, int(np.sqrt(len(cm_clean))))) if len(cm_clean) > 0 else 10)
        n_cm, _, _ = ax.hist(cm_clean, bins=n_bins_cm, alpha=0.7, color=dark_purple,
                             density=True, edgecolor='white', linewidth=0.8,
                             label='Provinces')
        y_max = max(y_max, n_cm.max())

    # --- Set Y-axis to log scale ---
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-4)

    # Fit and plot distributions
    if len(cm_clean) >= 2:
        if dist_type == 'skewt':
            cm_fit = fit_skewt(cm_clean)
            if cm_fit:
                x = np.linspace(cm_clean.min(), cm_clean.max(), 200)
                y_fit = skewt_pdf(x, *cm_fit)
                ax.plot(x, y_fit, color=dark_purple, linewidth=2.5)
        elif dist_type == 'gamma':
            cm_fit = fit_gamma(cm_clean)
            if cm_fit:
                x = np.linspace(cm_clean.min(), cm_clean.max(), 200)
                y_fit = gamma_pdf(x, *cm_fit)
                ax.plot(x, y_fit, color=dark_purple, linewidth=2.5)

    if len(tr_clean) >= 2:
        if dist_type == 'skewt':
            tr_fit = fit_skewt(tr_clean)
            if tr_fit:
                x = np.linspace(tr_clean.min(), tr_clean.max(), 200)
                y_fit = skewt_pdf(x, *tr_fit)
                ax.plot(x, y_fit, color=dark_orange, linewidth=2.5)
        elif dist_type == 'gamma':
            tr_fit = fit_gamma(tr_clean)
            if tr_fit:
                x = np.linspace(tr_clean.min(), tr_clean.max(), 200)
                y_fit = gamma_pdf(x, *tr_fit)
                ax.plot(x, y_fit, color=dark_orange, linewidth=2.5)

    # --- Mean (solid), median (dashed), and zero reference lines ---
    if len(cm_clean) > 0:
        ax.axvline(cm_clean.mean(), color=dark_purple, linestyle='-', linewidth=1.5, alpha=0.75)
        ax.axvline(np.median(cm_clean), color=dark_purple, linestyle='--', linewidth=1.0, alpha=0.6)
    if len(tr_clean) > 0:
        ax.axvline(tr_clean.mean(), color=dark_orange, linestyle='-', linewidth=1.5, alpha=0.75)
        ax.axvline(np.median(tr_clean), color=dark_orange, linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.75)

    # --- Aesthetics ---
    mean_value = all_data.mean()
    std_value = all_data.std()
    ax.set_xlim(mean_value - 2 * std_value, mean_value + 2 * std_value)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CCCCCC')
    ax.tick_params(axis='x', which='major', labelsize=18, colors='#333333', bottom=True, top=False)
    ax.tick_params(axis='y', which='major', labelsize=18, colors='#333333', left=True, right=False, labelleft=True)
    ax.set_xlabel(_axis_label(name), fontsize=14)
    ax.set_ylabel('Density (log scale)', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(False)

    plt.tight_layout()
    fname = os.path.join(output_dir,
                         f"specialty_histogram_{name.lower().replace(' ', '_').replace('/', '_')}_log_scale.pdf")
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved log-scale plot: {fname}")


# --- Main logic ---
def run_analysis(input_dir, base_output_dir):
    output_dir = os.path.join(base_output_dir, 'specialty_histograms2')
    os.makedirs(output_dir, exist_ok=True)

    cm_data = ec.load_terms(input_dir, 'province')
    tr_data = ec.load_terms(input_dir, 'canton')

    if cm_data is None or tr_data is None:
        print("Error: Could not load required data files.")
        return

    # Country-normalized NS (adds the missing *_ns_country at the canton level)
    cm_data = ec.add_country_ns(cm_data, input_dir)
    tr_data = ec.add_country_ns(tr_data, input_dir)

    # inc NS_ec
    cm_inc_ns = extract_tr(cm_data, 'Sel_province_from_canton_inc_ns_country', scale=1)
    tr_inc_ns = extract_tr(tr_data, 'Sel_canton_from_parish_inc_ns_country', scale=1)
    plot_hist(cm_inc_ns, tr_inc_ns, 'Income NS_ec', 'skewt', output_dir)
    plot_hist_log_scale(cm_inc_ns, tr_inc_ns, 'Income NS_ec', 'skewt', output_dir)

    # inc MSS
    cm_inc_mss = extract_tr(cm_data, 'Sel_province_from_canton_inc_mss', scale=1)
    tr_inc_mss = extract_tr(tr_data, 'Sel_canton_from_parish_inc_mss', scale=1)
    plot_hist(cm_inc_mss, tr_inc_mss, 'Income MSS', 'gamma', output_dir)
    plot_hist_log_scale(cm_inc_mss, tr_inc_mss, 'Income MSS', 'gamma', output_dir)

    # pop NS_ec
    cm_pop_ns = extract_tr(cm_data, 'Sel_province_from_canton_pop_ns_country', scale=1)
    tr_pop_ns = extract_tr(tr_data, 'Sel_canton_from_parish_pop_ns_country', scale=1)
    plot_hist(cm_pop_ns, tr_pop_ns, 'Population NS_ec', 'skewt', output_dir)
    plot_hist_log_scale(cm_pop_ns, tr_pop_ns, 'Population NS_ec', 'skewt', output_dir)

    # pop MSS
    cm_pop_mss = extract_tr(cm_data, 'Sel_province_from_canton_pop_mss', scale=1)
    tr_pop_mss = extract_tr(tr_data, 'Sel_canton_from_parish_pop_mss', scale=1)
    plot_hist(cm_pop_mss, tr_pop_mss, 'Population MSS', 'gamma', output_dir)
    plot_hist_log_scale(cm_pop_mss, tr_pop_mss, 'Population MSS', 'gamma', output_dir)

    # den NS_ec (density path: trait = firm count)
    cm_den_ns = extract_tr(cm_data, 'Sel_province_from_canton_den_ns_country', scale=1)
    tr_den_ns = extract_tr(tr_data, 'Sel_canton_from_parish_den_ns_country', scale=1)
    plot_hist(cm_den_ns, tr_den_ns, 'Density NS_ec', 'skewt', output_dir)
    plot_hist_log_scale(cm_den_ns, tr_den_ns, 'Density NS_ec', 'skewt', output_dir)

    # den MSS
    cm_den_mss = extract_tr(cm_data, 'Sel_province_from_canton_den_mss', scale=1)
    tr_den_mss = extract_tr(tr_data, 'Sel_canton_from_parish_den_mss', scale=1)
    plot_hist(cm_den_mss, tr_den_mss, 'Density MSS', 'gamma', output_dir)
    plot_hist_log_scale(cm_den_mss, tr_den_mss, 'Density MSS', 'gamma', output_dir)


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir)


if __name__ == "__main__":
    main()

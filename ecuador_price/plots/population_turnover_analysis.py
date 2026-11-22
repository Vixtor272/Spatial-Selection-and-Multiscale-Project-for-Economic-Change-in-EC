"""
population_turnover_analysis.py  (Ecuador port)

Port of UrbanTraitSelection/analysis_scripts/population_turnover_analysis.py.
Original code by Jordan T. Kemp and co-authors, https://github.com/JordanK10/UrbanTraitSelection (commit 045a994);
see the README of this folder for the full credit and citation.
Chicago compared the population turnover implied by the census panel with
county births+deaths (CDC data).  There is no firm birth/death registry here,
so only the turnover accounting is ported:

  Part 1: per-level total absolute firm-stock change (parish, canton,
          province, country), total and per year.
  Part 2: per-province net turnover, from the parish-level rows
          (Chicago: county-by-county from the block-group rows).

Results are printed and written to
    <window>/population_turnover/turnover_by_level.csv
    <window>/population_turnover/turnover_by_province.csv
"""

import os
import sys
import csv

import pandas as pd

from ecuador_price.plots import _ecuador_common as ec

HIERARCHY_LEVELS = ec.HIERARCHY_LEVELS


def calculate_population_turnover_no_pandas(input_dir, analysis_period_years):
    """
    Reads the exported CSVs for each hierarchical level using Python's
    built-in csv module and calculates the total absolute firm-stock change.
    (Chicago kept this pandas-free to bypass an environment issue; the
    structure is preserved.)
    """
    turnover_results = {}

    for level in HIERARCHY_LEVELS:
        filepath = ec.terms_path(input_dir, level)

        if not os.path.exists(filepath):
            print(f"Warning: File not found for level '{level}': {filepath}. Skipping.")
            continue
        try:
            total_turnover = 0.0
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                pop_initial_col = f'PopInitial_{level}'
                pop_final_col = f'PopFinal_{level}'
                try:
                    idx_initial = header.index(pop_initial_col)
                    idx_final = header.index(pop_final_col)
                except ValueError:
                    continue
                for i, row in enumerate(reader):
                    try:
                        initial_pop = float(row[idx_initial])
                        final_pop = float(row[idx_final])
                        total_turnover += abs(final_pop - initial_pop)
                    except (ValueError, IndexError):
                        continue
            turnover_results[level] = total_turnover
        except Exception as e:
            print(f"An error occurred processing {filepath}: {e}")

    if not turnover_results:
        print("No results were calculated. Please check file paths and column names.")
        return turnover_results

    n = analysis_period_years or 1
    print(f"{'Level':<10} {'Total Absolute Change (' + str(n) + ' Years)':<38} {'Avg. Annual Absolute Change':<30}")
    print("-" * 78)
    for level, total_change in turnover_results.items():
        annual_change = total_change / n
        pad = max(1, 38 - len(f'{total_change:,.0f}'))
        print(f"{level:<10} {total_change:,.0f}{' ':<{pad}} {annual_change:,.0f}")
    print("-" * 78)
    print("\nNote: 'population' is the stock of active formal firms; a firm birth/death")
    print("register (the CDC births+deaths analog) is not available for this data.")
    return turnover_results


def calculate_net_turnover_by_province(input_dir):
    """
    Calculates the window's NET firm-stock turnover for each province by
    summing the absolute changes of all its constituent parishes.
    This is the |Pop_Final - Pop_Initial| calculation.
    (Chicago: county turnover from the block-group file.)
    """
    df = ec.load_terms(input_dir, 'parish')
    if df is None:
        print("Error: Parish data file not found. Cannot calculate net turnover.")
        return {}

    df = df.copy()
    df['turnover'] = (df['PopFinal_parish'] - df['PopInitial_parish']).abs()
    return df.groupby('ParentProvince')['turnover'].sum().to_dict()


def run_analysis(input_dir, base_output_dir, window):
    y0, y1 = ec.window_years(window)
    analysis_period_years = (y1 - y0) if (y0 is not None and y1 is not None) else 1

    out_dir = os.path.join(base_output_dir, 'population_turnover')
    os.makedirs(out_dir, exist_ok=True)

    # --- Part 1: Grand total per level ---
    print("--- Running Total Turnover Analysis (per level) ---")
    turnover_results = calculate_population_turnover_no_pandas(input_dir, analysis_period_years)

    if turnover_results:
        level_df = pd.DataFrame([
            {'level': level,
             'total_absolute_change': total,
             'avg_annual_absolute_change': total / analysis_period_years}
            for level, total in turnover_results.items()
        ])
        level_path = os.path.join(out_dir, 'turnover_by_level.csv')
        level_df.to_csv(level_path, index=False)
        print(f"Saved: {level_path}")

    print("\n" + "="*80 + "\n")

    # --- Part 2: Province-by-Province Analysis ---
    print("--- Running Province-by-Province Turnover Analysis ---")

    net_turnover_data = calculate_net_turnover_by_province(input_dir)

    if not net_turnover_data:
        print("Could not perform province-level analysis due to missing turnover data.")
        return

    comparison_data = []
    for code in sorted(net_turnover_data):
        comparison_data.append({
            "Province Code": code,
            "Province": ec.province_label(code),
            f"Net {analysis_period_years}-Year Turnover": net_turnover_data[code],
            "Avg Annual Turnover": net_turnover_data[code] / analysis_period_years,
        })

    # --- Display Province Comparison Table ---
    print("\n" + "="*22 + " Province-Level Turnover " + "="*22)
    header = f"{'Code':<6} {'Province':<32} {'Net Turnover':>16} {'Annual':>12}"
    print(header)
    print("-" * len(header))

    comparison_data.sort(key=lambda x: x[f"Net {analysis_period_years}-Year Turnover"], reverse=True)

    for row in comparison_data:
        print(f"{row['Province Code']:<6} {row['Province']:<32} "
              f"{row[f'Net {analysis_period_years}-Year Turnover']:>16,.0f} "
              f"{row['Avg Annual Turnover']:>12,.0f}")

    print("-" * len(header))
    print("Net Turnover: |Firms_final - Firms_initial|, summed over all parishes in the province.")

    prov_df = pd.DataFrame(comparison_data)
    prov_path = os.path.join(out_dir, 'turnover_by_province.csv')
    prov_df.to_csv(prov_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {prov_path}")


def main():
    for window, input_dir, base_output_dir in ec.iter_windows(sys.argv[1:]):
        print(f"\n--- Processing window: {window} ---")
        run_analysis(input_dir, base_output_dir, window)


if __name__ == '__main__':
    main()

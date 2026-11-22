#!/usr/bin/env python3
"""
trait_split_table.py
====================
Two small tables behind the thesis text, both built on the driver
(ecuador_price/decomposition.py) and the unmodified Kemp engine.

1. trait_split.csv
   The population-path trait is the log of TOTAL parish sales, and

       ln(sales) = ln(firms) + ln(sales per firm).

   Selection is a covariance, so it is linear in the trait and splits exactly
   into a "size" part (trait = ln firms) and a "per firm" part (trait =
   ln sales per firm). The engine already computes the size part: it is the
   Price (_pop) output of the driver's second run, the one that feeds the
   density path. The per-firm part is the difference between the two runs.

2. path_weights.csv
   What each path's TOTAL averages over. The population path weights a parish
   by its firm share. The income and density paths weight it by firm share x
   sales and firm share x firm count, so the largest parishes count twice.
   The table reports those weights for the two metropolitan parishes, next
   to the aggregate growth rates the totals are sometimes mistaken for.

Outputs -> results/trait_split/

Run:
    python -m ecuador_price.analysis.trait_split_table
"""
import numpy as np
import pandas as pd

from ecuador_price import decomposition as dec
from ecuador_price import paths

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
OUT = paths.TRAIT_SPLIT
FREQ = "yearly"

YEAR_START, YEAR_END = 2017, 2023
WINDOWS = [(YEAR_START, YEAR_END)] + [(y, y + 1) for y in range(YEAR_START, YEAR_END)]
METROS = {"170150": "Quito", "090150": "Guayaquil"}


def national_price_terms(results):
    """The national population-path terms from one engine run."""
    row = results["country"].iloc[0]
    sel_prov = row["Sel_country_from_province_pop"]
    sel_canton = row["Transmitted_Sel_province_to_country_pop"]
    sel_parish = row["Transmitted_Sel_canton_to_country_pop"]
    return pd.Series({
        "total": row["AvgG_pop_country"],
        "selection": sel_prov + sel_canton + sel_parish,
        "sel_between_provinces": sel_prov,
        "sel_canton_within_province": sel_canton,
        "sel_parish_within_canton": sel_parish,
        "transmission": row["Transmitted_AvgG_parish_to_country_pop"],
    })


def trait_split(pp, valid_cells, year0, year1):
    """Three rows for one window: the thesis trait and its two parts."""
    key0, key1 = str(year0), str(year1)
    survivors = (dec.valid_parishes(pp, key0, valid_cells)
                 & dec.valid_parishes(pp, key1, valid_cells))
    elapsed = float(year1 - year0)

    sales = national_price_terms(
        dec.run_engine(pp, survivors, key0, key1, elapsed, "log_sales"))
    size = national_price_terms(
        dec.run_engine(pp, survivors, key0, key1, elapsed, "firms"))
    per_firm = sales - size                       # exact: the trait is a sum

    table = pd.DataFrame({"ln_sales": sales, "ln_firms": size,
                          "ln_sales_per_firm": per_firm}).T
    table.insert(0, "window", f"{year0}_to_{year1}")
    table.insert(1, "parishes", len(survivors))
    return table.rename_axis("trait").reset_index()


def path_weights(pp, valid_cells, year0, year1):
    """Weights behind each path's total, and the aggregate growth rates."""
    key0, key1 = str(year0), str(year1)
    survivors = (dec.valid_parishes(pp, key0, valid_cells)
                 & dec.valid_parishes(pp, key1, valid_cells))
    start = pp[(pp["period"] == key0) & pp["cod_parr"].isin(survivors)].set_index("cod_parr")
    end = pp[(pp["period"] == key1) & pp["cod_parr"].isin(survivors)].set_index("cod_parr")
    end = end.loc[start.index]
    years = year1 - year0

    firm_share = start["firms"] / start["firms"].sum()
    sales_share = start["sales"] / start["sales"].sum()
    inc_weight = firm_share * start["sales"]
    inc_weight = inc_weight / inc_weight.sum()
    den_weight = firm_share * start["firms"]
    den_weight = den_weight / den_weight.sum()
    final_firm_share = end["firms"] / end["firms"].sum()

    sales_growth = np.log(end["sales"] / start["sales"]) / years
    firm_growth = np.log(end["firms"] / start["firms"]) / years
    metros = list(METROS)

    rows = [
        ("firm share of the two metros, start", firm_share[metros].sum()),
        ("firm share of the two metros, end", final_firm_share[metros].sum()),
        ("sales share of the two metros, start", sales_share[metros].sum()),
        ("income-path weight of the two metros", inc_weight[metros].sum()),
        ("density-path weight of the two metros", den_weight[metros].sum()),
        ("aggregate sales growth per year", np.log(end["sales"].sum() / start["sales"].sum()) / years),
        ("income-path total (flat, one level)", (inc_weight * sales_growth).sum()),
        ("aggregate firm growth per year", np.log(end["firms"].sum() / start["firms"].sum()) / years),
        ("density-path total (flat, one level)", (den_weight * firm_growth).sum()),
        ("firm-weighted corr(ln sales, ln firms)",
         weighted_corr(firm_share, np.log(start["sales"]), np.log(start["firms"]))),
    ]
    for code, name in METROS.items():
        rows.append((f"{name} sales growth per year", sales_growth[code]))
        rows.append((f"{name} firm growth per year", firm_growth[code]))
    out = pd.DataFrame(rows, columns=["quantity", "value"])
    out.insert(0, "window", f"{year0}_to_{year1}")
    return out


def weighted_corr(weights, a, b):
    def cov(x, y):
        return (weights * (x - (weights * x).sum()) * (y - (weights * y).sum())).sum()
    return cov(a, b) / np.sqrt(cov(a, a) * cov(b, b))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    monthly = dec.monthly_table(dec.load_real_panel())
    pp = dec.parish_period_table(monthly, FREQ)
    valid_cells = dec.annual_valid_cells(monthly)

    split = pd.concat([trait_split(pp, valid_cells, y0, y1) for y0, y1 in WINDOWS],
                      ignore_index=True)
    weights = path_weights(pp, valid_cells, YEAR_START, YEAR_END)

    split.to_csv(OUT / "trait_split.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(OUT / "path_weights.csv", index=False, encoding="utf-8-sig")

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:+.4f}"):
        print("\n=== population path split by trait (per year) ===")
        print(split.to_string(index=False))
        print(f"\n=== what the path totals average over, {YEAR_START}->{YEAR_END} ===")
        print(weights.to_string(index=False))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()

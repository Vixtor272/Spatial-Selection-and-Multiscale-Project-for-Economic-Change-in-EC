#!/usr/bin/env python3
"""
prepare_sales_extract.py
========================
Step 0 of the database build: from the sales extract exactly as the SRI
delivered it, to the table of company sales the rest of the build reads.

The extract (tab-separated, 11.8 million rows, 2015-2025) has one row per
SRI parish x 4-digit CIIU activity x month x taxpayer type. Three things
happen here, and nothing else:

  1. Keep companies only. ``tipo`` is PN (natural persons), SDD_NO_PUB
     (private companies) or SDD_PUB (public companies); natural persons are
     dropped. The analysis later keeps SDD_NO_PUB alone.
  2. Keep rows that report sales. A row whose V_VTX (total sales and net
     exports) is blank declares no sales that month and is dropped. No blank
     is ever turned into a zero.
  3. Add two convenience columns:
        total_sales     = V_VTX - V_X      (sales without net exports)
        total_purchases = C_CTM - C_M      (purchases without net imports)
     where a blank V_X, C_CTM or C_M counts as 0.
     Example: V_VTX = 1,307,576.21 and V_X = 402,374.50 give
     total_sales = 905,201.71.

Input   data/raw/SRI_2015-2025_respaldo/BDD_AGR_VENTAS_..._TR1170120260413045.csv
Output  data/processed/df_ciuu_total_sales_purchases.csv   (2.4 million rows)

Run:
    python -m ecuador_price.build_db.prepare_sales_extract
"""
import pandas as pd

from ecuador_price import paths

COMPANY_TYPES = ["SDD_NO_PUB", "SDD_PUB"]


def prepare(extract):
    """The three steps of the module docstring, on the extract as a DataFrame."""
    companies = extract["tipo"].isin(COMPANY_TYPES)
    reports_sales = extract["V_VTX"].notna()
    out = extract[companies & reports_sales].copy()
    out["total_sales"] = out["V_VTX"] - out["V_X"].fillna(0)
    out["total_purchases"] = out["C_CTM"].fillna(0) - out["C_M"].fillna(0)
    return out


def main():
    # parroquia is a 7-digit code; reading it as text keeps it exactly as delivered
    extract = pd.read_csv(paths.SRI_SALES_EXTRACT, sep="\t", dtype={"parroquia": str})
    print(f"extract: {len(extract):,} rows; by type "
          f"{extract['tipo'].value_counts().to_dict()}")

    out = prepare(extract)
    paths.SALES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.SALES_CSV, index=False)
    print(f"kept {len(out):,} company rows that report sales "
          f"({out['tipo'].value_counts().to_dict()})")
    print(f"saved -> {paths.SALES_CSV}")


if __name__ == "__main__":
    main()

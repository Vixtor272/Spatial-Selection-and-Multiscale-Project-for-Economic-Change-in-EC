"""The exact identities the thesis relies on, checked on an invented economy
(tests/conftest.py), so they run without any SRI data.

    total change  =  selection  +  transmission                (every path)
    selection     =  between provinces + cantons within province
                     + parishes within canton                  (every path)
"""
import numpy as np
import pytest

from conftest import YEAR0, YEAR1, make_monthly
from ecuador_price import decomposition as dec

FREQ = "yearly"
TOL = 1e-10


def run(monthly):
    """One window YEAR0 -> YEAR1 through the real engine; returns
    (the single country row, the yearly parish table, the survivor set)."""
    pp = dec.parish_period_table(monthly, FREQ)
    valid = dec.annual_valid_cells(monthly)
    results = dec.run_window(pp, valid, "test", (YEAR0, 1), (YEAR1, 1), FREQ)
    survivors = (dec.valid_parishes(pp, str(YEAR0), valid)
                 & dec.valid_parishes(pp, str(YEAR1), valid))
    return results["country"].iloc[0], pp, survivors


def start_and_end(pp, survivors):
    start = pp[(pp["period"] == str(YEAR0)) & pp["cod_parr"].isin(survivors)].set_index("cod_parr")
    end = pp[(pp["period"] == str(YEAR1)) & pp["cod_parr"].isin(survivors)].set_index("cod_parr")
    return start, end.loc[start.index]


@pytest.mark.parametrize("path", ["pop", "inc", "den"])
def test_total_is_selection_plus_transmission(monthly, path):
    c, _, _ = run(monthly)
    total = c[f"AvgG_{path}_country"]
    selection = c[f"Cumulative_Sel_{path}_country"]
    transmission = c[f"Transmitted_AvgG_parish_to_country_{path}"]
    assert total == pytest.approx(selection + transmission, abs=TOL)


@pytest.mark.parametrize("path", ["pop", "inc", "den"])
def test_selection_splits_exactly_by_scale(monthly, path):
    c, _, _ = run(monthly)
    by_scale = (c[f"Sel_country_from_province_{path}"]               # between provinces
                + c[f"Transmitted_Sel_province_to_country_{path}"]   # cantons within province
                + c[f"Transmitted_Sel_canton_to_country_{path}"])    # parishes within canton
    assert c[f"Cumulative_Sel_{path}_country"] == pytest.approx(by_scale, abs=TOL)


def test_pop_total_is_the_change_of_the_firm_weighted_mean_log_sales(monthly):
    c, pp, survivors = run(monthly)
    start, end = start_and_end(pp, survivors)
    mean0 = (start["firms"] / start["firms"].sum() * np.log(start["sales"])).sum()
    mean1 = (end["firms"] / end["firms"].sum() * np.log(end["sales"])).sum()
    assert c["AvgG_pop_country"] == pytest.approx((mean1 - mean0) / (YEAR1 - YEAR0), abs=TOL)


def test_pop_selection_is_the_covariance_of_fitness_and_trait(monthly):
    """Summed over the three scales, selection is one flat covariance across
    parishes: sum of p (w/W - 1)(z - Z), with p the initial firm share,
    w the firm-count ratio and z the initial log sales."""
    c, pp, survivors = run(monthly)
    start, end = start_and_end(pp, survivors)
    p = start["firms"] / start["firms"].sum()
    w = end["firms"] / start["firms"]
    z = np.log(start["sales"])
    flat = (p * (w / (p * w).sum() - 1) * (z - (p * z).sum())).sum() / (YEAR1 - YEAR0)
    assert c["Cumulative_Sel_pop_country"] == pytest.approx(flat, abs=TOL)


def test_no_selection_under_proportional_growth():
    """If every parish's firm count grows by the same factor (Gibrat), nothing
    is reallocated: population- and density-path selection are exactly zero."""
    c, _, _ = run(make_monthly(firm_growth=1.25))
    assert c["Cumulative_Sel_pop_country"] == pytest.approx(0.0, abs=TOL)
    assert c["Cumulative_Sel_den_country"] == pytest.approx(0.0, abs=TOL)


def test_trait_split_is_exact(monthly):
    """ln(sales) = ln(firms) + ln(sales per firm), and selection is linear in
    the trait, so selection on ln(sales) minus selection on ln(firms) is the
    selection on ln(sales per firm) (the thesis's trait-split table)."""
    pp = dec.parish_period_table(monthly, FREQ)
    valid = dec.annual_valid_cells(monthly)
    k0, k1 = str(YEAR0), str(YEAR1)
    survivors = dec.valid_parishes(pp, k0, valid) & dec.valid_parishes(pp, k1, valid)
    elapsed = float(YEAR1 - YEAR0)
    on_sales = dec.run_engine(pp, survivors, k0, k1, elapsed, "log_sales")["country"].iloc[0]
    on_firms = dec.run_engine(pp, survivors, k0, k1, elapsed, "firms")["country"].iloc[0]

    start, end = start_and_end(pp, survivors)
    p = start["firms"] / start["firms"].sum()
    w = end["firms"] / start["firms"]
    per_firm = np.log(start["sales"] / start["firms"])
    direct = (p * (w / (p * w).sum() - 1) * (per_firm - (p * per_firm).sum())).sum() / elapsed

    difference = on_sales["Cumulative_Sel_pop_country"] - on_firms["Cumulative_Sel_pop_country"]
    assert difference == pytest.approx(direct, abs=TOL)


def test_parishes_below_the_sales_floor_are_left_out(monthly):
    tiny = monthly["cod_parr"].iloc[0]
    monthly.loc[monthly["cod_parr"] == tiny, "sales"] = 0.5        # $6 a year
    pp = dec.parish_period_table(monthly, FREQ)
    valid = dec.annual_valid_cells(monthly)
    assert tiny not in dec.valid_parishes(pp, str(YEAR0), valid)
    assert len(dec.valid_parishes(pp, str(YEAR0), valid)) == monthly["cod_parr"].nunique() - 1

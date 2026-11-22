"""Which time windows a run compares, for each time grain and step mode."""
import pytest

from ecuador_price import decomposition as dec


def windows(**settings):
    return dec.generate_windows(dec.Settings(**settings))


def test_period_labels_round_trip():
    for freq, sub, label in [("monthly", 7, "2019M07"), ("quarterly", 3, "2019Q3"),
                             ("halfyearly", 2, "2019H2"), ("yearly", 1, "2019")]:
        assert dec.period_label(2019, sub, freq) == label
        assert dec.parse_period(label, freq) == (2019, sub)


def test_elapsed_time_is_in_years():
    assert dec.period_time(2020, 1, "monthly") - dec.period_time(2019, 1, "monthly") == 1.0
    one_month = dec.period_time(2019, 2, "monthly") - dec.period_time(2019, 1, "monthly")
    assert one_month == pytest.approx(1 / 12)


def test_monthly_rolling_year_over_year():
    w = windows(freq="monthly", step_mode="rolling", window_horizon=12, window_step=1)
    steps = [x for x in w if x[3] == "step"]
    assert len(steps) == 72                               # 2018M01 ... 2023M12
    assert steps[0][0] == "2017M01_to_2018M01"
    assert steps[-1][0] == "2022M12_to_2023M12"
    # every step compares the same calendar month one year apart
    assert all(p0[1] == p1[1] and p1[0] - p0[0] == 1 for _, p0, p1, _ in steps)
    # the headline is season-matched too
    assert [x[0] for x in w if x[3] == "headline"] == ["2017M01_to_2023M01"]


def test_yearly_consecutive_with_custom_windows():
    w = windows(freq="yearly", step_mode="consecutive",
                custom_windows=[("2017", "2019"), ("2020", "2023")])
    names = {kind: [x[0] for x in w if x[3] == kind] for kind in ("step", "headline", "custom")}
    assert names["step"] == [f"{y}_to_{y + 1}" for y in range(2017, 2023)]
    assert names["headline"] == ["2017_to_2023"]
    assert names["custom"] == ["2017_to_2019", "2020_to_2023"]


def test_quarterly_year_over_year():
    w = windows(freq="quarterly", step_mode="year_over_year")
    assert len([x for x in w if x[3] == "step"]) == 4 * 6          # 4 quarters x 6 year pairs
    assert [x[0] for x in w if x[3] == "headline"] == \
        [f"2017Q{q}_to_2023Q{q}" for q in (1, 2, 3, 4)]


def test_run_names():
    assert [s.run_name() for s in dec.THESIS_RUNS] == [
        "monthly_rolling12_2017_2023", "yearly_consecutive_2017_2023",
        "quarterly_year_over_year_2017_2023"]


def test_unknown_step_mode_is_rejected():
    with pytest.raises(ValueError):
        windows(step_mode="sideways")

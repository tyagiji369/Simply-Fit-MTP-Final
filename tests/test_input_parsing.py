"""Tests for user-input parsing (src/input_parsing.py)."""
import io

from src.input_parsing import parse_weight_csv, parse_weight_text


def test_text_basic_and_missing_tokens():
    vals, errs = parse_weight_text("78.4, 78.1, -, 78.3\n77.9; NA, 77.5,")
    assert errs == []
    assert vals == [78.4, 78.1, None, 78.3, 77.9, None, 77.5]


def test_text_empty_is_empty_not_fake_data():
    assert parse_weight_text("") == ([], [])
    assert parse_weight_text("   \n ") == ([], [])
    assert parse_weight_text(None) == ([], [])


def test_text_non_numeric_reports_error_and_position():
    vals, errs = parse_weight_text("78, seventy, 77")
    assert vals == [78.0, None, 77.0]
    assert len(errs) == 1 and "Entry 2" in errs[0]


def test_csv_with_dates_builds_daily_grid_with_gaps():
    csv = "date,weight\n2026-01-01,80.0\n2026-01-02,79.8\n2026-01-05,79.5\n"
    vals, errs = parse_weight_csv(io.StringIO(csv))
    assert errs == []
    assert vals == [80.0, 79.8, None, None, 79.5]


def test_csv_unsorted_dates_and_alternate_header():
    csv = "Date,Weight_kg\n2026-01-03,79.0\n2026-01-01,80.0\n"
    vals, _ = parse_weight_csv(io.StringIO(csv))
    assert vals == [80.0, None, 79.0]


def test_csv_without_date_column_is_consecutive_days():
    vals, errs = parse_weight_csv(io.StringIO("weight\n80\n\n79.5\n"))
    assert errs == [] and vals[0] == 80.0 and vals[-1] == 79.5


def test_csv_day_column():
    vals, _ = parse_weight_csv(io.StringIO("day,weight\n1,80\n2,79.9\n4,79.7\n"))
    assert vals == [80.0, 79.9, None, 79.7]


def test_csv_errors():
    assert parse_weight_csv(io.StringIO("a,b\n1,2\n"))[1]                       # no weight column
    assert parse_weight_csv(io.StringIO("date,weight\nnot-a-date,80\n"))[1]     # bad date
    assert parse_weight_csv(io.StringIO("date,weight\n2026-01-01,80\n2026-01-01,81\n"))[1]  # duplicate day
    vals, errs = parse_weight_csv(io.StringIO("weight\n80\nabc\n79\n"))
    assert errs and vals == [80.0, None, 79.0]


def test_csv_bytes_upload():
    vals, errs = parse_weight_csv(b"weight\n80\n79.9\n")
    assert errs == [] and vals == [80.0, 79.9]

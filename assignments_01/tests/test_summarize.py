import pytest
from weatherkit.records import HourlyReading
from weatherkit.summarize import DailyAggregator

@pytest.fixture
def readings():
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.5),
        HourlyReading("2026-04-08T01:00", 15.0, 1.0),
        HourlyReading("2026-04-08T02:00", 12.0, 0.5),
        HourlyReading("2026-04-09T00:00", 20.0, 2.0),
        HourlyReading("2026-04-09T01:00", 17.0, 1.5),
        HourlyReading("2026-04-09T02:00", 19.0, 0.5),
    ]

def test_grouping_produces_two_summaries(readings):
    aggregator = DailyAggregator(min_hours=3)
    summaries = aggregator.summarize(readings)

    assert len(summaries) == 2
    assert summaries[0].date == "2026-04-08"
    assert summaries[1].date == "2026-04-09"


def test_temp_max_and_min_are_correct(readings):
    aggregator = DailyAggregator(min_hours=3)
    summaries = aggregator.summarize(readings)
    assert summaries[0].temp_max == 15.0
    assert summaries[0].temp_min == 10.0
    assert summaries[1].temp_max == 20.0
    assert summaries[1].temp_min == 17.0


def test_precipitation_sum_is_correct(readings):
    aggregator = DailyAggregator(min_hours=3)
    summaries = aggregator.summarize(readings)

    assert summaries[0].precipitation_sum == pytest.approx(2.0)
    assert summaries[1].precipitation_sum == pytest.approx(4.0)

def test_incomplete_day_is_dropped(readings):
    aggregator = DailyAggregator(min_hours=4)
    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    assert summaries == []
    assert incomplete == ["2026-04-08", "2026-04-09"]


@pytest.mark.parametrize("min_hours, expected_count", [
    (2, 2),
    (3, 2),
    (4, 0),
])

def test_min_hours_controls_days_kept(readings, min_hours, expected_count):
    aggregator = DailyAggregator(min_hours=min_hours)
    summaries = aggregator.summarize(readings)
    assert len(summaries) == expected_count


# I delibarately broke the method summarize and the test def test_temp_max_and_min_are_correct(readings) caught it right away (I changed max to min in the code which caused it and returned then back to proper)





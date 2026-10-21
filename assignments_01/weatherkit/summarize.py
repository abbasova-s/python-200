from dataclasses import dataclass
from .records import HourlyReading

@dataclass
class DailySummary:
    """Summary of hourly weather observations for one calendar day."""
    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """The day's temperature swing in degrees Celsius"""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Aggregate hourly weather readings into daily summaries."""
    def __init__(self, min_hours: int = 24):
        """Initialize the aggregator with a minimum number of observations."""
        self.min_hours = min_hours

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Create daily summaries from hourly readings."""
        groups = {}

        for reading in readings:
            date = reading.timestamp[:10]

            if date not in groups:
                groups[date] = []

            groups[date].append(reading)

        summaries = []

        for date, day_readings in groups.items():
            hours_observed = len(day_readings)

            if hours_observed < self.min_hours:
                continue

            summaries.append(
                DailySummary(
                    date=date,
                    temp_max=max(r.temperature_c for r in day_readings),
                    temp_min=min(r.temperature_c for r in day_readings),
                    precipitation_sum=sum(r.precipitation_mm for r in day_readings),
                    hours_observed=hours_observed,
                )
            )
        return sorted(summaries, key=lambda summary:summary.date)

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return dates with fewer than the minimum required observations."""
        groups = {}

        for reading in readings:
            date = reading.timestamp[:10]

            if date not in groups:
                groups[date] = []

            groups[date].append(reading)

        return sorted(
            date
            for date, day_readings in groups.items()
            if len(day_readings) < self.min_hours
        )


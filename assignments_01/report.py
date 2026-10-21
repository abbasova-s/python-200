import json

from weatherkit.records import to_readings
from weatherkit.schemas import WeatherResponse
from weatherkit.summarize import DailyAggregator


def main():
    with open("weather_raw.json") as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)

    readings = to_readings(response)

    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    print(f"{'Date':<12} {'High':>8} {'Low':>8} {'Precip':>10} {'Range':>8}")
    print("-" * 52)

    for summary in summaries:
        print(
            f"{summary.date:<12} "
            f"{summary.temp_max:>8.1f} "
            f"{summary.temp_min:>8.1f} "
            f"{summary.precipitation_sum:>10.1f} "
            f"{summary.temp_range():>8.1f}"
        )

    if incomplete:
        print(f"\nWarning: incomplete days dropped: {', '.join(incomplete)}")
    else:
        print("\nWarning: incomplete days dropped: none")


# Without this guard, importing report.py would immediately execute the entire script, rather than allowing another module to reuse its functions.
if __name__ == "__main__":
    main()



# Reflection
# 1) WeatherResponse would reject the whole file when a temp is null. It's helpful when somplete temp data is required for a proper analysis. Example: report that calculates daily temp statistics must have daily values. But if it's a monitoring projectsm one hourly missed value can be ok. list[float | None] can be used to tolerate gaps

#2) If a pipeline runs at noon, this day wouldn't have enough hourly onservations and will be dropped as incompleted. incomplete_days() makes it visible by showing which days are dropeed 

#3) Making weatherkit a package makes it easier to import specific pieces of the pipeline; can copy needed classes for instance without copying the code
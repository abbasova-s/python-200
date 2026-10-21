from weatherkit.records import HourlyReading, to_readings
from weatherkit.schemas import HourlyBlock, WeatherResponse

def make_response():
    return WeatherResponse(
        latitude=35.2,
        longitude=-78.6,
        timezone="EST",
        elevation=100.0,
        hourly=HourlyBlock(
            time=[
                "2026-04-08T00:00",
                "2026-04-08T01:00",
                "2026-04-08T02:00",
            ],
            temperature_2m=[10.0, 11.5, 13.0],
            precipitation=[0.0, 0.5, 1.0],
        ),
    )

def test_to_readings_returns_one_reading_per_hour_in_order():
    response = make_response()
    readings = to_readings(response)

    assert len(readings) == 3
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-08T02:00"


def test_values_in_readin_match_index_of_each_input_list():
    response = make_response()
    readings = to_readings(response)

    for i, reading in enumerate(readings):
        assert reading.timestamp == response.hourly.time[i]
        assert reading.temperature_c == response.hourly.temperature_2m[i]
        assert reading.precipitation_mm == response.hourly.precipitation[i]

def test_identical_hourly_readings():
    reading1 = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )
    reading2 = HourlyReading(
        timestamp="2026-04-08T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )

    assert reading1 == reading2
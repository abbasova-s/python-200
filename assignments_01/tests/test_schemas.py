from pathlib import Path
import json
import pytest
from pydantic import ValidationError
from weatherkit.schemas import WeatherResponse

def test_response_validates():
    # A plain relative path depends on the directory pytest is run from
    path = Path(__file__).parent.parent / "weather_raw.json"

    with open(path) as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)

    assert len(response.hourly.time) == 168


def test_latitude_range():
    data = {
        "latitude": 200.0,
        "longitude": -78.6,
        "timezone": "EST",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [10.0],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_mismatched_list_lengths():
    data = {
        "latitude": 23.0,
        "longitude": -78.6,
        "timezone": "EST",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [10.0, 20.9, 11.3],
            "precipitation": [0.0, 1.1, 2.2],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_null_value_in_temperature():
    data = {
        "latitude": 30.0,
        "longitude": -78.6,
        "timezone": "EST",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [None],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)
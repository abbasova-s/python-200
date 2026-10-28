# Question 1
# Write a class ModelInfo that reads a JSON file once in __init__ and stores its contents.

from pathlib import Path
import json


class ModelInfo:
    """Reads a JSON file once and prints the features count"""

    def __init__(self, path: str | Path):
        with open(path, "r") as f:
            self.data = json.load(f)

    def feature_count(self) -> int:
        return len(self.data["features"])

# Create a small JSON file with a "features" list 
metadata_path = Path("model_info_example.json")
sample_features = {
    "features" : ["temperature_max", "temperature_min", "precipitation_sum", "wind_speed", "humidity"]
}

with open(metadata_path, "w") as f:
    json.dump(sample_features, f)

model_info = ModelInfo(metadata_path)
print(f"Feature count: {model_info.feature_count()}")

# Loading the file in __init__ is better than loading it inside feature_count() if the method is called many times because this way file is loaded once and it avoids reading the file repeatedly every time the method is called.


# Question 2
# Write a dataclass DayForecast with three type-hinted fields: date (str), high_c (float), and will_rain (bool)

from dataclasses import dataclass

@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        if self.will_rain:
            return f"{self.date}: high {self.high_c}C, rain expected"
        return f"{self.date}: high {self.high_c}C, rain not expected"
  

forecast1 = DayForecast("2023-06-15", 24.1, True)
forecast2 = DayForecast("2023-10-19", 27.5, False)

print(forecast1.summary())
print(forecast2.summary())

# @dataclass generated the whole __init__ section automatically


# Question 3
# Write a function parse_temperature(text: str) -> float that converts a string like "24.1" to a float, and raises a ValueError with a message containing the word "temperature" when the text is not a number
import pytest

def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"Invalid temperature: {text}")

def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)

def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")














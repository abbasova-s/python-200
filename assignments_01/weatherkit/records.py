from dataclasses import dataclass
from .schemas import WeatherResponse

@dataclass
class HourlyReading:
    """One hourly observation of the weather at the given location
    Temperature is measured in degrees Celsius.
    Precipitation is measured in millimeters.
    """
    timestamp: str
    temperature_c: float
    precipitation_mm: float

def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert columnar responses in one record per hour
    Args: response: A validated WeatherResponse from the weather API.
    Returns: A list of HourlyReading objects in the same order as the hourly data in the response.
    """
    h = response.hourly
    return[
        HourlyReading(
            timestamp= h.time[i],
            temperature_c= h.temperature_2m[i],
            precipitation_mm= h.precipitation[i],
            )
            for i in range (len(h.time))
        ]

# HourlyReading is a dataclass because it organizes internal records, while WeatherResponse is the Pydentic model since it's where the data crosses from outside; protects the API boundary







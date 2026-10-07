# --- Classes ---
# Q1

class Thermometer:
    def __init__(self, location: str, readings: list[float]=None):
        self.location = location
        self.readings = [] if readings is None else readings

    def add(self, reading):
        self.readings.append(reading)

    def average(self) -> float | None:
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self) -> float | None:
        if not self.readings:
            return None
        return max(self.readings)

    # Q2
    def __repr__(self):
        return (
            f"Thermometer(location={self.location!r}," 
            f"n_readings={len(self.readings)}," 
            f"average={self.average()})"
        )

raleigh_thermometer = Thermometer(location= "Raleigh", readings= [11.4, 25.6, 23.6, 17.9])

print(raleigh_thermometer.average())
print(raleigh_thermometer.hottest())

# Average() needs to handle the empty case because if threre are no values (readings), it would be dividing by 0 and raise an error (ZeroDivisionError)


# Q2
my_thermometer = Thermometer("Charlotte", [23.4, 25.7,13.0, 11.2])
other_thermometer = Thermometer("Cary", [11.0, 22.5])
print(my_thermometer)
print([my_thermometer, other_thermometer])

# When a class has no __repr__ the Python displays the memory container and not the actual content of the instance. For debugging purposes it's so much better to see what each instance contents right there in the terminal


# Q3
class TemperatureAlert:
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [
            reading 
            for reading in thermometer.readings 
            if reading > self.threshold
        ]
    

alert1 = TemperatureAlert(35)
alert2 = TemperatureAlert(22)

print(alert1.breaches(raleigh_thermometer))
print(alert2.breaches(raleigh_thermometer))

# The reason for storing the threshold on TemperatureAlert rather than passed as an argument to breaches() is because it can be created once and reused it for as many thermometers as needed, insted of having to create 20 different once



# --- Dataclasses, Type Hints, and Docstrings ---
# Q1
from dataclasses import dataclass, FrozenInstanceError, field

@dataclass(frozen=True)
class Station:
    """ One station information with information observed at it"""
    station_id: float
    name: str
    latitude: float
    longitude: float
    elevation: float

station_a = Station(11, "Panama", 22.4, 12.8, 7.0)
station_b = Station(11, "Panama", 22.4, 12.8, 7.0)
print(station_a == station_b)

# The result is "True" because the values are equal in both objects. Dataclasses automatically create __eq__ alowwing to compare two identical objects. Without eq it would compare memory location and never be true even if values were the same


# Q2
try:
    station_a.longitude = 50.0
except FrozenInstanceError as e:
    print("do not let the script crash", e)

stations = {
    Station(11, "Tokmok", 22.4, 12.8, 7.0),
    Station(12, "Kemin", 28.4, 12.0, 9.0),
    Station(12, "Kemin", 28.4, 12.0, 9.0),
}

print(len(stations))

# Besides immutability, frozen dataclass becomes hashable, and can be used as a dictionary key or put in a set, like in this example. This is useful here because the set can identify duplicate Stations based on their field values.


# Q3
@dataclass
class StationBatch:
    """Represents a group of stations in a region"""
    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Add a station to the batch"""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the highest elevation or None if empty"""
        if not self.stations:
            return None
        
        return max(self.stations, key=lambda station: station.elevation)

# raise ValueError(f'mutable default {type(f.default)} for field 'ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory


# --- Pydantic ---
# Q1
from pydantic import BaseModel, Field, ValidationError, model_validator

class Reading(BaseModel):
    station_id: str	= Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    # Q4
    @model_validator(mode="after")
    def check_failed_sensor(self):
        """Reject readings that indicate a failed sensor."""
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError(
                f"humidity is ({self.humidity}) and temperature ({self.temperature_c})is below 40"
            )
        return self

entry1 = Reading(
    station_id="001",
    timestamp="11:11",
    temperature_c=27,
    humidity=30,)
print(entry1)


# Q2
# 1. Missing required field
try:
    Reading(
    station_id="001",
    temperature_c=27,
    humidity=30,)
except ValidationError as e:
    print(e)

# 2. Invalid temperature
try:
    Reading(
    station_id="001",
    timestamp="11:11",
    temperature_c=150,
    humidity=30,)
except ValidationError as e:
    print(e)

# 3. Invalid humidity
try:
    Reading(
    station_id="001",
    timestamp="11:11",
    temperature_c=27,
    humidity="very humid",)
except ValidationError as e:
    print(e)


entry2 = Reading(
    station_id="003",
    timestamp="11:11",
    temperature_c="21.5",
    humidity=40,)

print(entry2)
print(type(entry2.temperature_c))
print(type(entry2.humidity))

# Pydantic accepts values that can be converted to the declared type, but rejects values that cannot be converted to that type.


#Q3
try:
    Reading(
    station_id="1",
    temperature_c="good",
    humidity="25",)
except ValidationError as e:
    for error in e.errors():
        print(error['loc'], error['msg'])

# Three errors were reported: ('station_id',) String should have at least 3 characters, ('timestamp',) Field require, ('temperature_c',) Input should be a valid number, unable to parse string as a number. It's helpful to see all the errors at once that need to be fixed instead of having to fix one and come back for another


# Q4
entry3 = Reading(
    station_id="003",
    timestamp="01:11",
    temperature_c=27,
    humidity=30,)
print(entry3)

try:
    entry4 = Reading(
        station_id="004",
        timestamp="11:50",
        temperature_c=-50,
        humidity=0.0,
    )
except ValidationError as e:
    print(e)

# This rule cannot be expressed with Field constraints alone because it depends on the combination of two fields: humidity and temperature.Field constraints validate each field independently.


# --- Pytest ---
# Q1
import pytest

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert celsius to fahrenheit"""
    return celsius * 9/5 + 32

def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)
    
# pytest.approx is used because the exact comparison might fail when comparing floating-point calculations


# Q2
def mean(values: list[float]) -> float:
    """Return the mean of a list of values."""
    if not values:
        raise ValueError("values cannot be empty")
    return sum(values)/len(values)

def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])

# pytest.raises(ValueError) would confirm the error happened, but match+ also checks that the error message contains expected words


# Q3
@pytest.mark.parametrize("values, expected", [
    ([5], 5),
    ([1, 2, 3], 2),
    ([-2, -4, -6], -4),
    ([10, 20, 30, 40], 25),
])

def test_mean_values(values, expected):
    assert mean(values) == expected

# ================= 2 failed, 4 passed in 0.10s =================
# one parametrized test helps to save from creating multiple (or more) identical functions for each case


# Q4
# FAILURES 
# test_celsius_to_fahrenheit
# def test_celsius_to_fahrenheit():
# assert celsius_to_fahrenheit(0) == 32 > assert celsius_to_fahrenheit(100) == 212 E assert 257.0 == 212 E+  where 257.0 = celsius_to_fahrenheit(100)
# assignments_01/warmup_01.py:244: AssertionError

# pytest showed the actual result was 257.0 when the expected result was 212. This is more useful that just a "assertion failed"






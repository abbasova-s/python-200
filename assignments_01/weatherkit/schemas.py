from pydantic import BaseModel, Field, model_validator

class HourlyBlock(BaseModel):
    """The columnar array returned under the 'hourly' key """
    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode='after')
    def check_lists_length(self):
        """The lengths of three lists cannot be different"""
        if not (
            len(self.time)
            == len(self.temperature_2m)
            == len(self.precipitation)
        ):
            raise ValueError("Arrays must be the same length")
        return self

class WeatherResponse(BaseModel):
    """Hourly historical weather response from Open-Meteo archive API """
    latitude: float = Field(ge=-90, le=90)
    longitude:float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock
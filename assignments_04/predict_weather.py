"""Classify a few sample days using the saved weather model."""

from pathlib import Path
from weather_model import WeatherClassifier

SAMPLE_DAYS = [
    {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12},
    {"temperature_2m_max": 22, "temperature_2m_min": 13, "precipitation_sum": 9.0, "wind_speed_10m_max": 14},
    {"temperature_2m_max": 25, "temperature_2m_min": 15, "precipitation_sum": 0.0, "wind_speed_10m_max": 33},
    {"temperature_2m_max": 10, "temperature_2m_min": 2, "precipitation_sum": 2.5, "wind_speed_10m_max": 25},
]


def main() -> None:
    """Load the component and print a prediction for each sample day."""
    model_path = Path(__file__).parent / "models" / "weather_classifier.pkl"
    classifier = WeatherClassifier(model_path)
    for day, result in zip(SAMPLE_DAYS, classifier.predict(SAMPLE_DAYS)):
        print(f"{day['temperature_2m_max']:.0f}C, {day['precipitation_sum']:.0f}mm rain, "
              f"{day['wind_speed_10m_max']:.0f} km/h wind -> {result.label} ({result.probability:.2f})")

# # Without this guard, importing this module would run main() and print
# predictions even when another script only wanted to reuse its helpers.

if __name__ == "__main__":
    main()


#Reflection
# If the features are in the wrong order, the model could use a value for the wrong feature and make an incorrect prediction.

# A tiny test model makes tests independent from my saved model, so they are repeatable and don't depend on a specific file being available.

# The component can predict multiple days at once and returns structured results. Before production, I would add better error handling for bad data.
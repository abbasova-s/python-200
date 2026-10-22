import joblib
import json
import pandas as pd

# Load saved data and metadata
clf = joblib.load("assignments_03/models/weather_classifier.pkl")
with open("assignments_03/models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

# Print key metadata
print("City:", metadata["city"])
print("Features:", metadata["features"])
print("Test accuracy:", round(metadata["test_accuracy"], 3))

new_days = pd.DataFrame(
    [
        {"temperature_2m_max": 20, "temperature_2m_min": 5, "precipitation_sum": 0.0, "wind_speed_10m_max": 10},
        {"temperature_2m_max": 0, "temperature_2m_min": -15, "precipitation_sum": 10.0, "wind_speed_10m_max": 10},
        {"temperature_2m_max": 26, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 20},
        {"temperature_2m_max": 10, "temperature_2m_min": 0, "precipitation_sum": 20.0, "wind_speed_10m_max": 5},
    ],
    columns=metadata["features"],
)

# Predict labels and probabilities
preds = clf.predict(new_days)
probs = clf.predict_proba(new_days)[:,1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "good for running" if pred == 1 else "skip"
    print(f"Day {i+1}: {label} (confidence: {prob:.2%})")


# Task 8: Reflect
# 1. My borderline day has a maximum temperature of 26 degrees. The model gave it a probability of 100% for "good for running". The model appears confident in this prediction, although the probability is not a guarantee that the prediction is correct. If the model predicts 0.52, I would consider it uncertain because the probability is close to 0.5. I could flag the day for review or use additional weather safety rules before making a recommendation.

# 2. If predict.py runs before train_classifier.py has created the model, joblib.load() will fail because the model file does not exist. I could check whether the file exists before loading it and display a helpful message telling the user to run train_classifier.py first.

# 3. For a cloud pipeline, I would change predict.py to accept new weather data automatically from the pipeline instead of relying on manually hardcoded hypothetical days. I would also make sure predictions can be saved or sent to the next pipeline step.






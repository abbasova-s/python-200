import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sklearn
import joblib
import sys

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score, 
    recall_score, 
    f1_score,
)



# Task 1: Fetch the Data
# I have chosen Bishkek city, that is located in Kyrgyzstan

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 42.87,
    "longitude": 74.59,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
        ],
    "timezone": "Asia/Bishkek",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Data shape:", df.shape)
print(df.head())


# Task 2: Engineer the Label
# Create a good_for_running column that is 1 when a day is good and 0 otherwise
df["good_for_running"] = (
    df["temperature_2m_max"].between(7, 26)
    & (df["temperature_2m_min"] >=0)
    & (df["precipitation_sum"] < 3)
    & (df["wind_speed_10m_max"] < 30)
).astype(int)

good_days = (df["good_for_running"] == 1).sum()
skip_days = (df["good_for_running"] == 0).sum()
total_days = len(df)

print("Good days:", good_days)
print("Skip days:", skip_days)
print("Fraction of good days:", round(good_days / total_days, 3))

# About 31.8% of days are good for running in Bishkek. This seems reasonable given Bishkek's cold winters and seasonal weather changes. Since the classes are somewhat imbalanced, I should consider precision, recall, and F1-score alongside accuracy when evaluating the classifier.



# Task 3: Build and Evaluate a KNN Classifier
# Split the data into training and test sets (80/20, random_state=42, stratify on the label). Build a Pipeline of StandardScaler and KNeighborsClassifier
features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

for k in [1, 4, 5, 6, 9, 11, 15, 19, 25]:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(knn, X_train, y_train, cv=5 )

    print(f"k={k:2d}: mean = {scores.mean():.3f} std = {scores.std():.3f}")

best_k = 5
model_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])
model_knn.fit(X_train, y_train)
y_pred = model_knn.predict(X_test)
print("KNN model accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["skip", "good"]))


# Task 4: Build and Evaluate a Logistic Regression Classifier
# Build a Pipeline of StandardScaler and LogisticRegression(max_iter=1000)

model_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred_lr))
print("Precision: ", precision_score(y_test, y_pred_lr))
print("Recall: ", recall_score(y_test, y_pred_lr))
print("F1: ", f1_score(y_test, y_pred_lr))

print(classification_report(y_test, y_pred_lr, target_names=["skip", "good"]))

logreg = model_lr.named_steps["logreg"]
for name, coef in zip(features, logreg.coef_[0]):
    print(f"{name:20s} : {coef:+.2f}")

# Negative coefficients push predictions toward "skip". Precipitation has the strongest negative coefficient, followed by wind speed. Positive coefficients push predictions toward "good". The balanced model identifies more good days but has lower overall accuracy.


# Task 5: Compare and Choose
# KNN had higher accuracy (86.3%) than Logistic Regression (57.5%). For a running app, I would prioritize recall for the "good" class if the main goal is to help users find opportunities to run. A false positive means recommending a run on a bad-weather day, while a false negative means missing a good day for running. However, if bad weather could create safety risks, precision also matters because we want to avoid unsafe recommendations.

# I would deploy KNN, because it performed better in accuracy on this dataset, so I would consider it the stronger predictive model. However, KNN stores the training data and may be slower at prediction time. Logistic Regression is smaller and easier to interpret. Both models can provide probability estimates. Before deployment, I would test both models on additional data.


# Task 6: Save the Model

os.makedirs("assignments_03/models", exist_ok=True)

y_pred_knn = model_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_knn)
test_f1 = f1_score(y_test, y_pred_knn)

joblib.dump(model_knn, "assignments_03/models/weather_classifier.pkl")
print("Model saved")

metadata = {
    "python_version": sys.version.split()[0],
    "scikit_learn_version": sklearn.__version__,
    "model_type": "KNeighborsClassifier",
    "features": features,
    "label_rule": {
        "good_for_running": (
            "1 if maximum temperature is between 7 and 26 degrees C "
            "(inclusive), minimum temperature is at least 0 degrees C, "
            "precipitation is under 3 mm, and maximum wind speed is "
            "under 30 km/h; otherwise 0."
        )
    },
    "city": "Bishkek, Kyrgyzstan",
    "test_accuracy": test_accuracy,
    "test_f1": test_f1,
}

with open("assignments_03/models/weather_classifier_metadata.json", "w") as file:
    json.dump(metadata, file, indent=4)
print("Metadata saved")

print(f"Test accuracy: {test_accuracy:.3f}")
print(f"Test F1: {test_f1:.3f}")



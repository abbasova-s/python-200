import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("assignments_03/weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# --- Preprocessing --- 
# Q1
# Fit a StandardScaler on X_train only, then transform both X_train and X_test

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("X_train_scaled means: ", X_train_scaled.mean(axis=0))
print("X_test_scaled means: ", X_test_scaled.mean(axis=0))

# Fitting scaler on X_train only helps preventing from leaking the test set


# Q2
# The season column in the dataset is categorical. One-hot encode it with OneHotEncoder(sparse_output=False) 

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[["season"]])
print("Encoded shape: ", encoded.shape)
print("Feature names: ", encoder.get_feature_names_out())
print(encoded)


# The single season column became 4 columns because dataset contains 4 separate seasons and each becomes its own binary column


# Q3 
# Build a ColumnTransformer that applies a StandardScaler to the four numeric columns and a OneHotEncoder to season. Wrap it with a LogisticRegression in a Pipeline
X_q3 = df[numeric + ["season"]]
X_train_q3, X_test_q3, y_train_q3, y_test_q3 = train_test_split(
    X_q3, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["season"]),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

model.fit(X_train_q3, y_train_q3)
predictions = model.predict(X_test_q3)

print("Logistic Regression Accuracy: ", accuracy_score(y_test_q3, predictions))

# Pipeline puts everything together since its a sequence of steps where the output of each one becomes the input to the next one. It fits the model with scaler, encoder, and classifier. Then it applies the same preprocessing to the test set and predicts



# --- KNN --- 
# Q1
# Build a Pipeline of StandardScaler and KNeighborsClassifier(n_neighbors=5)

knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("KNN Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["skip", "good"]))


# Q2
# Loop over k values [1, 3, 5, 7, 9, 11, 15, 21]. For each, build the scaled pipeline and compute the mean 5-fold cross-validation accuracy on the training data

for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}: mean={scores.mean():.3f}")

# I'd choose k=11: mean=0.94, because it scored best across all folds on cross validation


# Q3
# Fit KNN once without scaling (on raw X_train) and once with scaling, both using n_neighbors=5

# KNN without scaling: 
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
print("Unscaled accuracy: ", accuracy_score(y_test, knn_unscaled.predict(X_test)))

# KNN with scaling: 
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
knn_scaled.fit(X_train, y_train)
print("Scaled accuracy: ", accuracy_score(y_test, knn_scaled.predict(X_test)))

# Scaling helped a lot: it chabged accuracy from 0.925 to 0.958. Different features might have different ranges individually. Small range in one feature might not mean it's unimportant but without scaling this is exactly what the model would assume and that's why scaling is crucial in a distance-based model


# --- Logistic Regression --- 
# Q1
# Build a Pipeline of StandardScaler and LogisticRegression(max_iter=1000)

model_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision (good):", precision_score(y_test, y_pred_lr, pos_label=1))
print("Recall (good):", recall_score(y_test, y_pred_lr, pos_label=1))
print("F1 (good):", f1_score(y_test, y_pred_lr, pos_label=1))


# Q2
# Print each feature name alongside its coefficient 
logreg = model_lr.named_steps["logreg"]
for name, coef in zip(numeric, logreg.coef_[0]):
    print(f"{name:20s} : {coef:+.2f}")

# Wind speed and precipitation push most strongly toward "skip." This matches my intuition because windy and rainy weather can make running less comfortable. Minimum temperature pushes slightly toward "good."


# Q3
# Use predict_proba to print the probability of "good" for the first five test days

probabilities = model_lr.predict_proba(X_test)[:,1]
print(probabilities[:5].round(2))

# Probability around 0.5 (0.47 in our case) means its barely "skip", so it's basically gambling 


# --- Evaluation --- 
# Q1
# For your logistic regression pipeline, build a confusion matrix on the test set and display it with ConfusionMatrixDisplay, using display_labels=["skip", "good"]

cm = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["skip", "good"])
disp.plot(colorbar=False)
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("assignments_03/outputs/logreg_confusion_matrix.png", bbox_inches="tight")
plt.show()

# Model made 9 false positives and 7 false negatives. In a running app the false positives would be less favorable for sure since the app would recommend running on a day with bad weather



# --- joblib --- 
# Q1
# take your fitted logistic regression pipeline and save it to models/warmup_model.pkl with joblib.dump

os.makedirs("assignments_03/models", exist_ok=True)
joblib.dump(model_lr, "assignments_03/models/warmup_model.pkl")
print("Model saved")


loaded = joblib.load("assignments_03/models/warmup_model.pkl")
assert (loaded.predict(X_test) == model_lr.predict(X_test)).all()
print("Predictions match.")

# If I saved only LogisticRegression without the scaler, raw input data would not be standardized before prediction. The model would receive features on a different scale than it was trained on, potentially producing incorrect predictions.



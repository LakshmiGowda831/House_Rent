# train_model.py
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Load California housing dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Feature and target info
feature_names = list(X.columns)
target_name = "MedHouseVal"

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Model pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained — MSE: {mse:.2f}, R²: {r2:.2f}")

# Save model
joblib.dump(pipeline, "boston_best_model.pkl")

# Save feature metadata
meta = {
    "numeric_features": feature_names,
    "categorical_features": [],
    "all_features": feature_names,
    "target": target_name
}
with open("feature_columns.json", "w") as f:
    json.dump(meta, f)

# Optional: Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("residual_plot.png")

# Optional: Predicted vs Actual
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.savefig("pred_vs_actual.png")

# Optional: Model card
with open("model_card.md", "w") as f:
    f.write(f"# Model Card\n\n")
    f.write(f"- Model: RandomForestRegressor\n")
    f.write(f"- Features: {', '.join(feature_names)}\n")
    f.write(f"- Target: {target_name}\n")
    f.write(f"- R² Score: {r2:.2f}\n")
    f.write(f"- MSE: {mse:.2f}\n")
    f.write(f"- Trained on: {len(X_train)} samples\n")

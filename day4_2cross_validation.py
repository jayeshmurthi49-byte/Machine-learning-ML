import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    StratifiedKFold,
    LeaveOneOut
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data = fetch_california_housing()
X = data.data
y = data.target

model = LinearRegression()

print("=== Simple Cross Validation ===")
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"All 5 scores : {scores.round(4)}")
print(f"Mean R2      : {scores.mean():.4f}")
print(f"Std R2       : {scores.std():.4f}")

print("\n=== KFold Cross Validation ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
print(f"All 5 scores : {kf_scores.round(4)}")
print(f"Mean R2      : {kf_scores.mean():.4f}")
print(f"Std R2       : {kf_scores.std():.4f}")

print("\n=== Effect of K on Cross Validation ===")
for k in [3, 5, 10]:
    scores = cross_val_score(model, X, y, cv=k, scoring="r2")
    print(f"K={k:2d} → Mean R2={scores.mean():.4f} Std={scores.std():.4f}")

print("\n=== Cross Validation with Scaling Pipeline ===")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
pipe_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
print(f"All 5 scores : {pipe_scores.round(4)}")
print(f"Mean R2      : {pipe_scores.mean():.4f}")
print(f"Std R2       : {pipe_scores.std():.4f}")

print("\n=== Summary ===")
print(f"Simple train test split R2 : 0.5757")
print(f"Cross validation mean R2   : {scores.mean():.4f}")
print(f"More reliable score        : Cross Validation")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = fetch_california_housing()
X = data.data
y = data.target

# Split first — ALWAYS split before scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")

# Without scaling
model_no_scale = LinearRegression()
model_no_scale.fit(X_train, y_train)
y_pred_no_scale = model_no_scale.predict(X_test)
r2_no_scale = r2_score(y_test, y_pred_no_scale)

# With StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

print("\n=== Comparison ===")
print(f"R2 without scaling : {r2_no_scale:.4f}")
print(f"R2 with scaling    : {r2_scaled:.4f}")

print("\n=== Coefficients Comparison ===")
print(f"{'Feature':15} {'No Scale':12} {'Scaled':12}")
for feature, c1, c2 in zip(data.feature_names,
                             model_no_scale.coef_,
                             model_scaled.coef_):
    print(f"{feature:15} {c1:12.4f} {c2:12.4f}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test, y_pred_no_scale, alpha=0.3, color="steelblue")
axes[0].plot([0, 5], [0, 5], color="red", linestyle="--")
axes[0].set_title(f"Without Scaling R2={r2_no_scale:.4f}")
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")

axes[1].scatter(y_test, y_pred_scaled, alpha=0.3, color="green")
axes[1].plot([0, 5], [0, 5], color="red", linestyle="--")
axes[1].set_title(f"With Scaling R2={r2_scaled:.4f}")
axes[1].set_xlabel("Actual")
axes[1].set_ylabel("Predicted")

plt.tight_layout()
plt.savefig("scaling_comparison.png")
plt.show()
print("\nPlot saved as scaling_comparison.png")
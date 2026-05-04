import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression

# Load data

data = fetch_california_housing()
df = pd.DataFrame(data.data,columns=data.feature_names)
df["price"] = data.target


print("Shape:", df.shape)
print("\nFeatures:", data.feature_names)
print("\nFirst 5 rows:")
print(df.head())

# Use ALL 8 features this time
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining size : {X_train.shape[0]}")
print(f"Test size     : {X_test.shape[0]}")
print(f"Features used : {X_train.shape[1]}")


# Train
model = LinearRegression()
model.fit(X_train,y_train)

# Coefficients for each feature
print("\n=== Model Coefficients ===")
for feature,coef in zip(data.feature_names,model.coef_):
    print(f"{feature:15} : {coef:4f}")
print(f"{'Intercept':15} : {model.intercept_:.4f}")

#predict 
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

n  = len(y_test)
n_features = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


print("\n=== Evaluation ===")
print(f"MSE         : {mse:.4f}")
print(f"RMSE        : {np.sqrt(mse):.4f}")
print(f"R2 Score    : {r2:.4f}")
print(f"Adjusted R2 : {adjusted_r2:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.3, color="steelblue")
plt.plot([0, 5], [0, 5], color="red", linewidth=1, linestyle="--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted — Multiple Linear Regression (All Features)")
plt.tight_layout()
plt.savefig("multiple_regression_result.png")
plt.show()
print("\nPlot saved.")
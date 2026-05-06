import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = fetch_california_housing()
X = data.data
y = data.target

print("=== Effect of random_state on R2 ===")
for state in [0, 1, 42, 99, 123]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"random_state={state:4d} → R2={r2:.4f}")

print("\n=== Effect of test_size on R2 ===")
for size in [0.1, 0.2, 0.3, 0.4]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=size, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"test_size={size} → Train={len(X_train)} Test={len(X_test)} R2={r2:.4f}")
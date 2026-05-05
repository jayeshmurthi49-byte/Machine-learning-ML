import numpy as np
import pandas as pd

# Sample data — different scales
data = {
    "MedInc":     [8.32, 5.64, 3.87, 7.25, 2.19],
    "HouseAge":   [41.0, 21.0, 52.0, 35.0, 18.0],
    "Population": [322.0, 2401.0, 496.0, 558.0, 565.0]
}

df = pd.DataFrame(data)
print("=== Original Data ===")
print(df)
print(f"\nMean:\n{df.mean()}")
print(f"\nStd:\n{df.std()}")


class StandardScalerScratch:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MinMaxScalerScratch:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    X = df.values

    # Standard Scaler
    ss = StandardScalerScratch()
    X_standard = ss.fit_transform(X)
    print("\n=== After StandardScaler ===")
    print(np.round(X_standard, 4))
    print(f"Mean after scaling : {np.mean(X_standard, axis=0).round(4)}")
    print(f"Std  after scaling : {np.std(X_standard, axis=0).round(4)}")

    # MinMax Scaler
    mm = MinMaxScalerScratch()
    X_minmax = mm.fit_transform(X)
    print("\n=== After MinMaxScaler ===")
    print(np.round(X_minmax, 4))
    print(f"Min after scaling : {np.min(X_minmax, axis=0).round(4)}")
    print(f"Max after scaling : {np.max(X_minmax, axis=0).round(4)}")
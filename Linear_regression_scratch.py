import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionScratch:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X_train, y_train):
        x_mean = X_train.mean()
        y_mean = y_train.mean()

        numerator = np.sum((X_train - x_mean) * (y_train - y_mean))
        denominator = np.sum((X_train - x_mean) ** 2)

        self.slope = numerator / denominator
        self.intercept = y_mean - (self.slope * x_mean)

    def predict(self, X):
        return self.slope * X + self.intercept

    def mse(self, y_actual, y_predicted):
        return np.mean((y_actual - y_predicted) ** 2)

    def r2_score(self, y_actual, y_predicted):
        ss_res = np.sum((y_actual - y_predicted) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    df = pd.read_csv("placement.csv")
    print(df.head())

    X = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    model = LinearRegressionScratch()
    model.fit(X_train, y_train)

    print(f"Slope     : {model.slope:.4f}")
    print(f"Intercept : {model.intercept:.4f}")

    y_pred = model.predict(X_test)

    print(f"MSE       : {model.mse(y_test, y_pred):.4f}")
    print(f"R2 Score  : {model.r2_score(y_test, y_pred):.4f}")
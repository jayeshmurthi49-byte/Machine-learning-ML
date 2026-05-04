import numpy as np

y_actual = np.array([3.26, 1.98, 3.25, 3.67, 3.57,
                        2.99, 2.60, 2.48, 2.31, 3.51])

y_predicted = np.array([3.10, 2.10, 3.40, 3.50, 3.60,
                         3.00, 2.50, 2.60, 2.20, 3.40])


def mae(y_actual,y_predicted):
    return np.mean(np.abs(y_actual - y_predicted))

def mse(y_actual,y_predicted):
    return np.mean((y_actual - y_predicted) **2)

def rmse(y_actual,y_predicted):
    return np.sqrt(mse(y_actual,y_predicted))

def r2_score(y_actual,y_predicted):
    ss_res = np.sum((y_actual - y_predicted) **2)
    ss_tot = np.sum((y_actual - np.mean(y_predicted) **2))
    return 1 - (ss_res / ss_tot)

def adjusted_r2(y_actual,y_predicted,n_features):
    n = len(y_actual)
    r2 = r2_score(y_actual,y_predicted)
    return 1- (1 -r2) * (n -1) / (n-n_features-1)

if __name__ == "__main__":
    print("=== Regression Metrics ===")
    print(f"MAE            : {mae(y_actual, y_predicted):.4f}")
    print(f"MSE            : {mse(y_actual, y_predicted):.4f}")
    print(f"RMSE           : {rmse(y_actual, y_predicted):.4f}")
    print(f"R2 Score       : {r2_score(y_actual, y_predicted):.4f}")
    print(f"Adjusted R2    : {adjusted_r2(y_actual, y_predicted, n_features=1):.4f}")

    print("\n=== What each metric means ===")
    print(f"MAE  {mae(y_actual, y_predicted):.4f} → average error is {mae(y_actual, y_predicted):.2f} LPA")
    print(f"RMSE {rmse(y_actual, y_predicted):.4f} → errors in same unit as salary")
    print(f"R2   {r2_score(y_actual, y_predicted):.4f} → model explains {r2_score(y_actual, y_predicted)*100:.1f}% of variance")
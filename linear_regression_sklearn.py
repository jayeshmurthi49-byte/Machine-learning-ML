import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

# Step 1 — Load data
data = fetch_california_housing()

# step 2 - Explore
df = pd.DataFrame(data.data,columns=data.feature_names)
df["Prices"] = data.target
print("Shape:",df.shape)
print("\n First 5 rows:")
print(df.head())
print("\n Basic Stats")
print(df.describe())

X = data.data[:,0].reshape(-1,1)
y = data.target

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)
print(f"\n Training size :{X_train.shape[0]}")
print(f"Test size : {X_test.shape[0]}")

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f"\nMSE      : {mse:.4f}")
print(f"R2 Score : {r2:.4f}")



plt.figure(figsize=(8,5))
plt.scatter(y_test,y_pred,alpha=0.3,color="Steelblue")
plt.xlabel("Actual price")
plt.ylabel("Predicted prices")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted — Linear Regression (MedInc only)")
plt.tight_layout()
plt.savefig("linear_regression_result.png")
plt.show()
print("\nPlot saved as linear_regression_result.png")

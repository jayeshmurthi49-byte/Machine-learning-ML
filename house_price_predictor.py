import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
df = pd.DataFrame(data.data,columns=data.feature_names)
df["Price"] = data.target 

print("=== Data info ===")
print(f"Shape {df.shape}")
print(f"\n First five row")
print(df.head())
print(f"\n Missing values")
print(df.isnull().sum())
print(f"\nBasic stats:")
print(df.describe())

# Step 2 — Prepare data
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) 

print(f"\n Trainig size :{X_train.shape[0]}")
print(f"Test size :{X_test.shape[0]}") 

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

params = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, None]
}

grid = GridSearchCV(pipeline,params,cv=3,
                    scoring="r2",verbose=1)
grid.fit(X_train, y_train)

print(f"\nBest Params : {grid.best_params_}")
print(f"Best CV R2  : {grid.best_score_:.4f}")
 

y_pred = grid.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)


print(f"\n=== Evaluation ===")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {np.sqrt(mse):.4f}")
print(f"R2   : {r2:.4f}")

# Step 6 — Plot 
plt.figure(figsize=(8,5))
plt.scatter(y_test,y_pred,alpha=0.3,color="steelblue")
plt.plot([0,5],[0,5],color="red",linestyle="--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted price")
plt.title('House Price Predictor — Actual vs Predicted')
plt.tight_layout()
plt.savefig('prediction_plot.png')
plt.show()
print("\n Plot saved") 

# Step 7 — Save model
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(grid, f)
print("Model saved as house_price_model.pkl")

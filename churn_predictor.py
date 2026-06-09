import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Step 1 — Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("=== Dataset Info ===")
print(f"Shape : {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nChurn distribution:")
print(df['Churn'].value_counts())

# Step 2 — Clean data
# Drop customerID — not useful
df.drop('customerID', axis=1, inplace=True)

# TotalCharges has spaces — convert to float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Step 3 — Encode categorical columns
le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("\nAfter encoding:")
print(df.head())

# Step 4 — Split data
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining size : {X_train.shape[0]}")
print(f"Test size     : {X_test.shape[0]}")

# Step 5 — Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6 — Evaluate
y_pred = model.predict(X_test)
print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred,
      target_names=['Not Churn', 'Churn']))

# Step 7 — Feature importance
print("\n=== Top 5 Features ===")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns
for i in range(5):
    print(f"{feature_names[indices[i]]:20} : {importances[indices[i]]:.4f}")

# Step 8 — Save model and feature names
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("\nModel saved.")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
import numpy as np


data = load_breast_cancer()
X = data.data
y = data.target 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) 

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(f"Accuracy without scaling : {accuracy_score(y_test, y_pred):.4f}") 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = KNeighborsClassifier(n_neighbors=5)
model_scaled.fit(X_train_scaled,y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
print(f"Accuracy with scaling    : {accuracy_score(y_test, y_pred_scaled):.4f}")


for k in [3,5,7,9,11]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled,y_train)
    acc = accuracy_score(y_test,model.predict(X_test_scaled))
    print(f"k={k} Accuracy={acc}:.4f") 


# Compare all 3 algorithms
print("\n=== Algorithm Comparison ===")
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

algorithms = {
    "KNN           ": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree ": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest ": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name,algo in algorithms.items():
    scores = cross_val_score(algo,X,y,cv=5,scoring="accuracy")
    print(f"{name} : {scores.mean():.4f} +/- {scores.std():.4f}")
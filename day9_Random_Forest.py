from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,classification_report
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) 

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred,
      target_names=data.target_names))

importance = model.feature_importances_
indeices = np.argsort(importance)[::-1]
for i in range(5):
    print(f"{data.feature_names[indeices[i]]} : {importance[indeices[i]]}")


scores = cross_val_score(model,X,y,cv=5,scoring="accuracy")
print(f"\nCross Val : {scores.mean():.4f} +/- {scores.std():.4f}") 

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_scores = cross_val_score(dt, X, y, cv=5, scoring="accuracy")
print(f"\n=== Comparison ===")
print(f"Decision Tree  : {dt_scores.mean():.4f}")
print(f"Random Forest  : {scores.mean():.4f}")
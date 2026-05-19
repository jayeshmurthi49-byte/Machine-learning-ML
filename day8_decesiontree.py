from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Effect of max_depth 
print("=== Effect of max_depth ===")
for depth in [2, 3, 5, 10, None]:
    model = DecisionTreeClassifier(max_depth=depth,random_state=42)
    model.fit(X_train,y_train)
    train_acc = accuracy_score(y_train,model.predict(X_train))
    test_acc = accuracy_score(y_test,model.predict(X_test))
    print(f"max_depth={str(depth):5} train={train_acc:.4f} test={test_acc:.4f}")

#best model
print("\n=== Best Model max_depth=5 ===")
model = DecisionTreeClassifier(max_depth=5,random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(f"Accuracy :{accuracy_score(y_test,y_pred)}")
print(classification_report (y_test,y_pred,
                             target_names=data.target_names))

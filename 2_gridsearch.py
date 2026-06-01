from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score 



data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) 

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42 
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
]) 

params = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [3, 5, 10, None]
} 

grid = GridSearchCV(pipeline,params,cv=5,scoring="accuracy")
grid.fit(X_train,y_train)

print(f"\nBest Params : {grid.best_params_}")
print(f"Best Score  : {grid.best_score_:.4f}")

# Predict with best model
y_pred = grid.predict(X_test)
print(f"Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
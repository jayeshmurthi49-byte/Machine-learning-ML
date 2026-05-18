import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeScratch:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping conditions
        if (depth >= self.max_depth or
                n_classes == 1 or
                n_samples < self.min_samples_split):
            leaf_value = self._most_common(y)
            return Node(value=leaf_value)

        # Find best split
        best_feature, best_threshold = self._best_split(X, y, n_features)

        if best_feature is None:
            return Node(value=self._most_common(y))

        # Split data
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx

        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, n_features):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_gini = self._gini(y)

        left_idx = X_column <= threshold
        right_idx = ~left_idx

        if sum(left_idx) == 0 or sum(right_idx) == 0:
            return 0

        n = len(y)
        n_left = sum(left_idx)
        n_right = sum(right_idx)

        child_gini = ((n_left / n) * self._gini(y[left_idx]) +
                      (n_right / n) * self._gini(y[right_idx]))

        return parent_gini - child_gini

    def _gini(self, y):
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        return 1 - sum(p ** 2 for p in probabilities)

    def _most_common(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeScratch(max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
import numpy as np

class KNearestNeighbors:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        print(f"KNN: Stored {len(y)} training samples with k={self.k}")
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _predict_one(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_nearest = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_nearest]
        return 1 if np.sum(k_labels) > self.k / 2 else 0
        
    def predict(self, X):
        X = np.array(X)
        predictions = []
        for i in range(len(X)):
            pred = self._predict_one(X[i])
            predictions.append(pred)
            if (i + 1) % 100 == 0:
                print(f"  KNN predicted {i+1}/{len(X)} samples...")
        return np.array(predictions)



class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        print(f"Logistic Regression: Training on {num_samples} samples...")
        for iteration in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)
            error = predictions - y
            dw = (1 / num_samples) * np.dot(X.T, error)
            db = (1 / num_samples) * np.sum(error)
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            if (iteration + 1) % 200 == 0:
                loss = self._calculate_loss(y, predictions)
                print(f"  Iteration {iteration+1}/{self.num_iterations}, Loss: {loss:.4f}")
        print("  Training complete!")
    
    def _calculate_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def predict_proba(self, X):
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X):
        
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


class SimpleDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None
    
    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        p_1 = np.mean(y)
        p_0 = 1 - p_1
        return 1 - (p_0 ** 2 + p_1 ** 2)
    
    def _best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        num_features = X.shape[1]
        for feature_idx in range(num_features):
            thresholds = np.percentile(X[:, feature_idx], [25, 50, 75])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                left_y = y[left_mask]
                right_y = y[right_mask]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                weighted_gini = (
                    len(left_y) * self._gini_impurity(left_y) +
                    len(right_y) * self._gini_impurity(right_y)
                ) / len(y)
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth:
            return {'leaf': True, 'prediction': int(np.mean(y) >= 0.5)}
        if len(np.unique(y)) == 1:
            return {'leaf': True, 'prediction': int(y[0])}
        if len(y) < 2:
            return {'leaf': True, 'prediction': int(np.mean(y) >= 0.5)}
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {'leaf': True, 'prediction': int(np.mean(y) >= 0.5)}
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        print(f"Decision Tree: Building tree with max_depth={self.max_depth}...")
        self.tree = self._build_tree(X, y, depth=0)
        print("  Training complete!")
    
    def _predict_one(self, x, node):
        if node['leaf']:
            return node['prediction']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])
    
    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(x, self.tree) for x in X])



class SimpleRandomForest:
    """
    Random Forest Classifier (from scratch)

    How it works:
        1. Build N decision trees, each on a random bootstrap sample
        2. Each tree uses a random subset of features
        3. Final prediction = majority vote across all trees
    """

    def __init__(self, n_trees=10, max_depth=5, max_features=None):
        """
        n_trees:      number of trees in the forest
        max_depth:    max depth of each tree
        max_features: features considered per split (default: sqrt of total)
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []  # list of (tree, feature_indices)

    def _bootstrap_sample(self, X, y):
        """Random sample WITH replacement, same size as original"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y,random_state=42):
        np.random.seed(random_state)
        X = np.array(X)
        y = np.array(y)
        n_features = X.shape[1]

        if self.max_features is None:
            self.max_features = max(1, int(np.sqrt(n_features)))

        print(f"Random Forest: Training {self.n_trees} trees "
              f"(max_depth={self.max_depth}, max_features={self.max_features})...")

        self.trees = []
        for i in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)
            X_subset = X_sample[:, feature_indices]

            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_subset, y_sample)
            self.trees.append((tree, feature_indices))
            print(f"  Tree {i+1}/{self.n_trees} done.")

        print("  Random Forest training complete!")

    def predict(self, X):
        X = np.array(X)
        all_preds = np.zeros((len(self.trees), len(X)), dtype=int)

        for i, (tree, feature_indices) in enumerate(self.trees):
            all_preds[i] = tree.predict(X[:, feature_indices])

        # Majority vote
        final = []
        for j in range(len(X)):
            votes = all_preds[:, j]
            final.append(1 if np.sum(votes) > len(votes) / 2 else 0)

        return np.array(final)


if __name__ == "__main__":
    print("This file defines machine learning models.")
import numpy as np
from collections import Counter
import random

class DecisionNode:
    """
    Simple node class for our decision tree
    I could have made this more complex but keeping it minimal for now
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # which feature to split on
        self.threshold = threshold  # threshold value for the split
        self.left = left           # left subtree
        self.right = right         # right subtree  
        self.value = value         # prediction value (for leaf nodes)

class DecisionTree:
    """
    Basic decision tree implementation
    Not the most optimized but gets the job done
    """
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # number of features to consider at each split
        self.root = None
    
    def fit(self, X, y):
        # Set number of features to consider if not specified
        # Using sqrt(total_features) is a common heuristic
        self.n_features = self.n_features or int(np.sqrt(X.shape[1]))
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria - learned these the hard way through trial and error
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            # Return most common class as leaf value
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)
        
        # Random feature selection - this is what makes it "random" forest
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        # Find the best split among random features
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)
        
        # Create child splits
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return DecisionNode(best_feature, best_thresh, left, right)
    
    def _best_split(self, X, y, feat_idxs):
        """
        Find the best feature and threshold to split on
        Using information gain - there are other metrics but this works well
        """
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)  # try all unique values as thresholds
            
            for thr in thresholds:
                # Calculate information gain for this split
                gain = self._information_gain(y, X_column, thr)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        
        return split_idx, split_threshold
    
    def _information_gain(self, y, X_column, threshold):
        """
        Calculate information gain using entropy
        Math is a bit involved but basically measures how much "disorder" we reduce
        """
        # Parent entropy
        parent_entropy = self._entropy(y)
        
        # Create children
        left_idxs, right_idxs = self._split(X_column, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0  # no split happened
        
        # Calculate weighted average of children entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        
        # Information gain is reduction in entropy
        return parent_entropy - child_entropy
    
    def _split(self, X_column, split_thresh):
        """Simple split function - left gets values <= threshold"""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _entropy(self, y):
        """
        Calculate entropy - measure of disorder/uncertainty
        Lower entropy = more pure/certain
        """
        hist = np.bincount(y)
        ps = hist / len(y)  # probabilities
        # Avoid log(0) by filtering out zero probabilities
        return -np.sum([p * np.log(p) for p in ps if p > 0])
    
    def _most_common_label(self, y):
        """Helper to find most frequent class"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """Make predictions for input samples"""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction for single sample"""
        if node.value is not None:  # leaf node
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForest:
    """
    Random Forest implementation - ensemble of decision trees
    The magic happens by combining multiple "weak" learners
    """
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
    
    def fit(self, X, y):
        """
        Train the forest by building multiple trees on bootstrap samples
        Bootstrap sampling = sampling with replacement
        """
        self.trees = []
        
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            
            # Bootstrap sampling - this adds diversity to our trees
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def _bootstrap_samples(self, X, y):
        """
        Create bootstrap sample (sampling with replacement)
        This is key to reducing overfitting - each tree sees slightly different data
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def predict(self, X):
        """
        Make predictions by majority voting across all trees
        More democratic than a single tree!
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Transpose so we have predictions for each sample across trees
        tree_preds = np.swapaxes(predictions, 0, 1)
        # Take majority vote for each sample
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
    
    def _most_common_label(self, y):
        """Find the most common prediction among trees"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]

# Example usage and testing
if __name__ == "__main__":
    # Let's test it with some dummy data
    # In practice you'd use real datasets like iris, wine, etc.
    
    # Generate some random classification data
    np.random.seed(42)  # for reproducibility
    n_samples, n_features = 1000, 4
    X = np.random.randn(n_samples, n_features)
    # Create some pattern in the data so it's actually learnable
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # simple linear combination
    
    # Add some noise to make it more realistic
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_idx] = 1 - y[noise_idx]  # flip 10% of labels
    
    # Split into train/test (doing it manually to avoid sklearn dependency)
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train our random forest
    print("Training Random Forest...")
    rf = RandomForest(n_trees=50, max_depth=10)  # 50 trees should be plenty
    rf.fit(X_train, y_train)
    
    # Make predictions
    predictions = rf.predict(X_test)
    
    # Calculate accuracy (simple metric but good enough for demo)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print(f"Accuracy: {accuracy:.3f}")
    
    # Also test single decision tree for comparison
    print("\nTraining single Decision Tree for comparison...")
    single_tree = DecisionTree(max_depth=10)
    single_tree.fit(X_train, y_train)
    single_predictions = single_tree.predict(X_test)
    single_accuracy = np.sum(single_predictions == y_test) / len(y_test)
    print(f"Single tree accuracy: {single_accuracy:.3f}")
    
    print(f"\nImprovement from ensemble: {accuracy - single_accuracy:.3f}")
    print("Random Forest should generally perform better due to ensemble effect!")

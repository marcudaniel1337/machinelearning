import math
from collections import defaultdict, Counter

class NaiveBayesClassifier:
    def __init__(self):
        # Store prior probabilities for each class
        self.class_priors = {}
        # For each class, store feature likelihood info
        self.feature_probs = {}
        # Track if feature is numeric or categorical
        self.feature_types = []

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.

        Args:
            X (list of lists): Dataset features.
            y (list): Class labels.
        """
        n_samples = len(X)
        n_features = len(X[0])
        self.feature_types = self._detect_feature_types(X)

        # Calculate prior probabilities P(class)
        class_counts = Counter(y)
        self.class_priors = {cls: count / n_samples for cls, count in class_counts.items()}

        # Initialize feature_probs as dict: {class: [{value: prob} or (mean, var)]}
        self.feature_probs = {cls: [{} for _ in range(n_features)] for cls in class_counts}

        # Separate data by class
        separated = defaultdict(list)
        for features, label in zip(X, y):
            separated[label].append(features)

        # For each class, calculate likelihoods for each feature
        for cls, instances in separated.items():
            for feature_index in range(n_features):
                feature_values = [instance[feature_index] for instance in instances]

                if self.feature_types[feature_index] == 'categorical':
                    # Calculate frequency of each categorical value
                    counts = Counter(feature_values)
                    total = len(feature_values)
                    # Store probability P(feature_value | class)
                    self.feature_probs[cls][feature_index] = {val: count / total for val, count in counts.items()}

                else:
                    # Numeric feature: calculate mean and variance for Gaussian likelihood
                    mean = sum(feature_values) / len(feature_values)
                    var = sum((x - mean) ** 2 for x in feature_values) / len(feature_values)
                    self.feature_probs[cls][feature_index] = (mean, var)

    def predict(self, X):
        """
        Predict the class labels for multiple instances.

        Args:
            X (list of lists): Data points.

        Returns:
            list: Predicted class labels.
        """
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        """
        Predict the class label for a single instance x.
        """
        posteriors = {}

        for cls, prior in self.class_priors.items():
            # Start with the prior probability
            posterior = math.log(prior)  # use log to avoid underflow

            for feature_index, value in enumerate(x):
                if self.feature_types[feature_index] == 'categorical':
                    # Get conditional probability P(feature=value | class)
                    probs = self.feature_probs[cls][feature_index]
                    # Use a small smoothing value if unseen value
                    prob = probs.get(value, 1e-6)
                    posterior += math.log(prob)

                else:
                    # Numeric feature: Gaussian likelihood
                    mean, var = self.feature_probs[cls][feature_index]
                    prob = self._gaussian_probability(value, mean, var)
                    posterior += math.log(prob)

            posteriors[cls] = posterior

        # Return class with highest posterior probability
        return max(posteriors, key=posteriors.get)

    @staticmethod
    def _gaussian_probability(x, mean, var):
        """
        Calculate Gaussian probability density function.
        """
        if var == 0:
            return 1.0 if x == mean else 1e-6  # avoid division by zero
        exponent = math.exp(- ((x - mean) ** 2) / (2 * var))
        return (1 / math.sqrt(2 * math.pi * var)) * exponent

    @staticmethod
    def _detect_feature_types(X):
        """
        Detect whether each feature is categorical or numeric.

        Simple heuristic: if all values can be cast to float, treat numeric.
        Otherwise, categorical.
        """
        feature_types = []
        n_features = len(X[0])
        for i in range(n_features):
            try:
                for row in X:
                    float(row[i])
                feature_types.append('numeric')
            except ValueError:
                feature_types.append('categorical')
        return feature_types


if __name__ == "__main__":
    # Example dataset (categorical and numeric features)
    X_train = [
        [5.1, 3.5, 'setosa'],
        [4.9, 3.0, 'setosa'],
        [7.0, 3.2, 'versicolor'],
        [6.4, 3.2, 'versicolor'],
        [6.3, 3.3, 'virginica'],
        [5.8, 2.7, 'virginica'],
    ]
    # Separate features and labels
    X = [row[:-1] for row in X_train]
    y = [row[-1] for row in X_train]

    clf = NaiveBayesClassifier()
    clf.fit(X, y)

    X_test = [
        [5.0, 3.4],
        [6.5, 3.0],
    ]

    predictions = clf.predict(X_test)
    print("Predictions:", predictions)

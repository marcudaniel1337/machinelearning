import math
from collections import Counter

def entropy(labels):
    """
    Calculate the entropy of a list of class labels.
    Entropy measures impurity; lower entropy = more pure.
    """
    total = len(labels)
    counts = Counter(labels)
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def split_dataset(dataset, feature_index, threshold):
    """
    Split dataset into two parts based on a feature and threshold.
    Left side: feature value <= threshold
    Right side: feature value > threshold
    """
    left = []
    right = []
    for row in dataset:
        if row[feature_index] <= threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right

def information_gain(parent_labels, left_labels, right_labels):
    """
    Calculate information gain from splitting a node into left and right subsets.
    """
    parent_entropy = entropy(parent_labels)
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    if n_left == 0 or n_right == 0:
        # No split actually happened
        return 0

    weighted_entropy = (n_left / n) * entropy(left_labels) + (n_right / n) * entropy(right_labels)
    gain = parent_entropy - weighted_entropy
    return gain

def best_split(dataset):
    """
    Find the best feature and threshold to split on to maximize information gain.
    """
    best_gain = 0
    best_feature = None
    best_threshold = None
    best_splits = None

    n_features = len(dataset[0]) - 1  # last column is label
    parent_labels = [row[-1] for row in dataset]

    for feature_index in range(n_features):
        # Consider all unique values as possible thresholds
        values = sorted(set(row[feature_index] for row in dataset))

        for i in range(len(values) - 1):
            # Threshold between two consecutive values
            threshold = (values[i] + values[i + 1]) / 2

            left, right = split_dataset(dataset, feature_index, threshold)
            left_labels = [row[-1] for row in left]
            right_labels = [row[-1] for row in right]

            gain = information_gain(parent_labels, left_labels, right_labels)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold
                best_splits = (left, right)

    return best_feature, best_threshold, best_splits, best_gain

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes, store class label

def build_tree(dataset, max_depth=None, min_size=1, depth=0):
    """
    Recursively build the decision tree.
    """
    labels = [row[-1] for row in dataset]
    # If all labels are same, create a leaf node
    if len(set(labels)) == 1:
        return DecisionNode(value=labels[0])

    # If reached max depth or dataset too small, create leaf with majority label
    if max_depth is not None and depth >= max_depth or len(dataset) <= min_size:
        majority_label = Counter(labels).most_common(1)[0][0]
        return DecisionNode(value=majority_label)

    feature, threshold, splits, gain = best_split(dataset)

    if gain == 0 or splits is None:
        majority_label = Counter(labels).most_common(1)[0][0]
        return DecisionNode(value=majority_label)

    left_branch = build_tree(splits[0], max_depth, min_size, depth + 1)
    right_branch = build_tree(splits[1], max_depth, min_size, depth + 1)

    return DecisionNode(feature_index=feature, threshold=threshold, left=left_branch, right=right_branch)

def predict(node, row):
    """
    Predict the class label for a single data row by traversing the tree.
    """
    if node.value is not None:
        return node.value
    if row[node.feature_index] <= node.threshold:
        return predict(node.left, row)
    else:
        return predict(node.right, row)

if __name__ == "__main__":
    # Simple dataset: [feature1, feature2, ..., label]
    dataset = [
        [2.7, 2.5, 0],
        [1.3, 1.5, 0],
        [3.6, 4.0, 0],
        [7.6, 2.8, 1],
        [5.3, 2.7, 1],
        [6.9, 1.7, 1],
        [8.0, 2.0, 1]
    ]

    tree = build_tree(dataset, max_depth=3)

    print("Predictions:")
    for row in dataset:
        pred = predict(tree, row)
        print(f"Actual: {row[-1]}, Predicted: {pred}")

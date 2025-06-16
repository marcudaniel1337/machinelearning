import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

class Perceptron:
    """
    Simple Perceptron implementation - the granddaddy of neural networks!
    
    This is probably the simplest "learning" algorithm you can implement.
    The math is straightforward but the concept is profound - a machine that
    can learn from mistakes and gradually improve its performance.
    
    Frank Rosenblatt would be proud (and probably amazed at what this led to).
    """
    
    def __init__(self, learning_rate=0.01, max_epochs=1000, random_state=None):
        """
        Initialize the perceptron
        
        learning_rate: how big steps to take when updating weights
        I usually start with 0.01 and adjust based on how training goes.
        Too high = overshooting, too low = slow convergence
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state
        
        # These get set during training
        self.weights = None
        self.bias = None
        self.errors_per_epoch = []  # track learning progress
        self.converged = False
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
    
    def _activation_function(self, z):
        """
        Step function activation - the classic perceptron activation
        
        Returns 1 if input >= 0, else 0
        This is what makes it a "hard" classifier - no middle ground!
        """
        return np.where(z >= 0.0, 1, 0)
    
    def _net_input(self, X):
        """
        Calculate net input (weighted sum + bias)
        
        This is the core computation: z = w·x + b
        Simple dot product but it's doing all the heavy lifting
        """
        return np.dot(X, self.weights) + self.bias
    
    def fit(self, X, y):
        """
        Train the perceptron using the classic perceptron learning rule
        
        The algorithm is beautifully simple:
        1. Make a prediction
        2. If wrong, adjust weights in the right direction
        3. Repeat until convergence (or give up after max_epochs)
        
        This only works for linearly separable data - learned that the hard way!
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        # Small random weights work better than zeros (avoids symmetry issues)
        self.weights = np.random.normal(0.0, 0.01, n_features)
        self.bias = 0.0
        
        self.errors_per_epoch = []
        
        for epoch in range(self.max_epochs):
            errors = 0
            
            # Go through each training example
            for xi, target in zip(X, y):
                # Forward pass: compute prediction
                net_input = self._net_input(xi)
                prediction = self._activation_function(net_input)
                
                # Calculate error (target - prediction)
                error = target - prediction
                
                # Update weights only if there's an error
                # This is the key insight: only learn from mistakes!
                if error != 0:
                    # Weight update rule: w = w + η * error * x
                    # If error > 0: we predicted too low, increase weights for positive features
                    # If error < 0: we predicted too high, decrease weights for positive features
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
                    errors += 1
            
            self.errors_per_epoch.append(errors)
            
            # Check for convergence - no errors means we got everything right!
            if errors == 0:
                self.converged = True
                print(f"Converged after {epoch + 1} epochs!")
                break
        
        if not self.converged:
            print(f"Did not converge after {self.max_epochs} epochs.")
            print("Data might not be linearly separable, or try increasing max_epochs")
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Just compute net input and apply activation function
        """
        if self.weights is None:
            raise ValueError("Model not trained yet! Call fit() first.")
        
        return self._activation_function(self._net_input(X))
    
    def predict_proba(self, X):
        """
        Return "probabilities" - well, sort of...
        
        Since perceptron uses step function, we don't get real probabilities.
        This returns the raw net input transformed to [0,1] range as a proxy.
        Not mathematically rigorous but sometimes useful for confidence estimates.
        """
        if self.weights is None:
            raise ValueError("Model not trained yet! Call fit() first.")
        
        net_input = self._net_input(X)
        # Use sigmoid to convert to [0,1] range
        # This is a bit of a hack - real perceptron doesn't do probabilities
        proba_1 = 1 / (1 + np.exp(-net_input))
        proba_0 = 1 - proba_1
        
        return np.column_stack([proba_0, proba_1])
    
    def score(self, X, y):
        """Calculate accuracy on test data"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def decision_function(self, X):
        """
        Return raw decision function values (before activation)
        
        Positive values = class 1, negative values = class 0
        The magnitude tells you how "confident" the decision is
        """
        if self.weights is None:
            raise ValueError("Model not trained yet! Call fit() first.")
        
        return self._net_input(X)

class MultiClassPerceptron:
    """
    Multi-class perceptron using one-vs-all strategy
    
    The original perceptron only handles binary classification.
    For multiple classes, we train one perceptron per class.
    Not the most elegant solution but it works!
    """
    
    def __init__(self, learning_rate=0.01, max_epochs=1000, random_state=None):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.perceptrons = {}
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Train one perceptron per class
        
        For each class, we create a binary problem: "this class vs all others"
        Then combine the results during prediction
        """
        self.classes_ = np.unique(y)
        
        for class_label in self.classes_:
            print(f"Training perceptron for class {class_label}")
            
            # Create binary labels: 1 for this class, 0 for all others
            binary_y = np.where(y == class_label, 1, 0)
            
            # Train perceptron for this class
            perceptron = Perceptron(
                learning_rate=self.learning_rate,
                max_epochs=self.max_epochs,
                random_state=self.random_state
            )
            perceptron.fit(X, binary_y)
            
            self.perceptrons[class_label] = perceptron
        
        return self
    
    def predict(self, X):
        """
        Predict by choosing class with highest decision function value
        
        Each perceptron gives a "confidence" score. We pick the most confident one.
        """
        if not self.perceptrons:
            raise ValueError("Model not trained yet!")
        
        # Get decision function values from all perceptrons
        decision_scores = np.zeros((X.shape[0], len(self.classes_)))
        
        for idx, class_label in enumerate(self.classes_):
            decision_scores[:, idx] = self.perceptrons[class_label].decision_function(X)
        
        # Choose class with highest score
        predicted_indices = np.argmax(decision_scores, axis=1)
        return self.classes_[predicted_indices]
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

def plot_decision_boundary(perceptron, X, y, title="Perceptron Decision Boundary"):
    """
    Plot the decision boundary for 2D data
    
    This really helps visualize what the perceptron learned
    The decision boundary is just a straight line - that's all it can do!
    """
    if X.shape[1] != 2:
        print("Can only plot decision boundary for 2D data")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Create a mesh to plot the decision boundary
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Get predictions for each point in the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    
    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    
    # Plot the decision line more clearly
    if hasattr(perceptron, 'weights') and perceptron.weights is not None:
        # Decision boundary: w1*x1 + w2*x2 + bias = 0
        # Solve for x2: x2 = -(w1*x1 + bias) / w2
        w1, w2 = perceptron.weights
        bias = perceptron.bias
        
        if abs(w2) > 1e-6:  # avoid division by zero
            x_line = np.array([x_min, x_max])
            y_line = -(w1 * x_line + bias) / w2
            plt.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Boundary')
            plt.legend()
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)

def plot_learning_curve(perceptron, title="Perceptron Learning Curve"):
    """
    Plot how errors decrease over epochs
    
    This shows the learning process - hopefully errors go down over time!
    """
    if not hasattr(perceptron, 'errors_per_epoch') or not perceptron.errors_per_epoch:
        print("No learning curve data available")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(perceptron.errors_per_epoch) + 1), 
             perceptron.errors_per_epoch, 'b-', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Errors')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add some annotations
    if perceptron.converged:
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Perfect Classification')
        plt.legend()

def generate_linearly_separable_data(n_samples=100, n_features=2, 
                                   random_state=42, margin=1.0):
    """
    Generate linearly separable data for testing
    
    The key is making sure the classes can actually be separated by a line.
    I spent way too much time debugging before realizing my test data wasn't separable!
    """
    np.random.seed(random_state)
    
    if n_features == 2:
        # Create two well-separated clusters
        n_per_class = n_samples // 2
        
        # Class 0: left side
        X0 = np.random.randn(n_per_class, 2) * 0.3 + np.array([-margin, 0])
        y0 = np.zeros(n_per_class, dtype=int)
        
        # Class 1: right side  
        X1 = np.random.randn(n_per_class, 2) * 0.3 + np.array([margin, 0])
        y1 = np.ones(n_per_class, dtype=int)
        
        X = np.vstack([X0, X1])
        y = np.hstack([y0, y1])
        
    else:
        # For higher dimensions, create random separable data
        # This is trickier to guarantee separability, so we use a simple approach
        X = np.random.randn(n_samples, n_features)
        
        # Create a random hyperplane and assign labels based on which side points fall
        w_true = np.random.randn(n_features)
        b_true = 0
        y = (np.dot(X, w_true) + b_true >= 0).astype(int)
        
        # Add some margin by moving points away from the boundary
        distances = np.abs(np.dot(X, w_true) + b_true) / np.linalg.norm(w_true)
        close_to_boundary = distances < margin
        
        # Move points that are too close to the boundary
        direction = np.sign(np.dot(X, w_true) + b_true).reshape(-1, 1)
        X[close_to_boundary] += direction[close_to_boundary] * w_true.reshape(1, -1) * margin
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

def generate_non_separable_data(n_samples=100, noise=0.3, random_state=42):
    """
    Generate data that's NOT linearly separable
    
    Good for demonstrating when perceptron fails to converge
    """
    np.random.seed(random_state)
    
    # Create XOR-like pattern that's not linearly separable
    n_per_quadrant = n_samples // 4
    
    # Four clusters in different quadrants with mixed labels
    X1 = np.random.randn(n_per_quadrant, 2) * noise + np.array([-1, -1])
    X2 = np.random.randn(n_per_quadrant, 2) * noise + np.array([1, 1])
    X3 = np.random.randn(n_per_quadrant, 2) * noise + np.array([-1, 1])
    X4 = np.random.randn(n_per_quadrant, 2) * noise + np.array([1, -1])
    
    X = np.vstack([X1, X2, X3, X4])
    y = np.array([1] * n_per_quadrant + [1] * n_per_quadrant + 
                [0] * n_per_quadrant + [0] * n_per_quadrant)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

# Example usage and comprehensive testing
if __name__ == "__main__":
    print("Testing Perceptron Implementation")
    print("=" * 40)
    
    # Test 1: Linearly separable data
    print("\n1. Testing with linearly separable data:")
    X_sep, y_sep = generate_linearly_separable_data(n_samples=100, margin=1.5, random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X_sep))
    X_train, X_test = X_sep[:split_idx], X_sep[split_idx:]
    y_train, y_test = y_sep[:split_idx], y_sep[split_idx:]
    
    # Train perceptron
    perceptron = Perceptron(learning_rate=0.1, max_epochs=100, random_state=42)
    perceptron.fit(X_train, y_train)
    
    # Evaluate
    train_acc = perceptron.score(X_train, y_train)
    test_acc = perceptron.score(X_test, y_test)
    
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Final weights: {perceptron.weights}")
    print(f"Final bias: {perceptron.bias:.3f}")
    
    # Visualize results
    plot_decision_boundary(perceptron, X_sep, y_sep, "Linearly Separable Data")
    plt.tight_layout()
    plt.show()
    
    plot_learning_curve(perceptron, "Learning Curve - Separable Data")
    plt.tight_layout()
    plt.show()
    
    # Test 2: Non-linearly separable data
    print("\n2. Testing with non-linearly separable data:")
    X_nonsep, y_nonsep = generate_non_separable_data(n_samples=100, noise=0.2, random_state=42)
    
    perceptron_fail = Perceptron(learning_rate=0.1, max_epochs=50, random_state=42)
    perceptron_fail.fit(X_nonsep, y_nonsep)
    
    acc_nonsep = perceptron_fail.score(X_nonsep, y_nonsep)
    print(f"Accuracy on non-separable data: {acc_nonsep:.3f}")
    print("As expected, perceptron struggles with non-separable data")
    
    plot_decision_boundary(perceptron_fail, X_nonsep, y_nonsep, 
                          "Non-Linearly Separable Data (XOR-like)")
    plt.tight_layout()
    plt.show()
    
    plot_learning_curve(perceptron_fail, "Learning Curve - Non-Separable Data")
    plt.tight_layout()
    plt.show()
    
    # Test 3: Multi-class classification
    print("\n3. Testing multi-class perceptron:")
    
    # Generate 3-class data
    np.random.seed(42)
    n_per_class = 30
    
    # Three well-separated clusters
    X1 = np.random.randn(n_per_class, 2) * 0.3 + np.array([0, 2])
    X2 = np.random.randn(n_per_class, 2) * 0.3 + np.array([-2, -1])
    X3 = np.random.randn(n_per_class, 2) * 0.3 + np.array([2, -1])
    
    X_multi = np.vstack([X1, X2, X3])
    y_multi = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)
    
    # Shuffle
    indices = np.random.permutation(len(X_multi))
    X_multi, y_multi = X_multi[indices], y_multi[indices]
    
    # Train multi-class perceptron
    multi_perceptron = MultiClassPerceptron(learning_rate=0.1, max_epochs=100, random_state=42)
    multi_perceptron.fit(X_multi, y_multi)
    
    multi_acc = multi_perceptron.score(X_multi, y_multi)
    print(f"Multi-class accuracy: {multi_acc:.3f}")
    
    # Visualize multi-class results
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green']
    for class_idx in range(3):
        mask = y_multi == class_idx
        plt.scatter(X_multi[mask, 0], X_multi[mask, 1], 
                   c=colors[class_idx], label=f'Class {class_idx}',
                   alpha=0.7, edgecolors='black')
    
    plt.title("Multi-class Classification Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Test 4: Effect of learning rate
    print("\n4. Testing different learning rates:")
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    
    for lr in learning_rates:
        p = Perceptron(learning_rate=lr, max_epochs=100, random_state=42)
        p.fit(X_train, y_train)
        acc = p.score(X_test, y_test)
        epochs_to_converge = len(p.errors_per_epoch) if p.converged else "Did not converge"
        print(f"LR={lr:5.3f}: Accuracy={acc:.3f}, Epochs={epochs_to_converge}")
    
    print("\nKey Insights:")
    print("- Perceptron only works on linearly separable data")
    print("- Learning rate affects convergence speed but not final accuracy (for separable data)")
    print("- Multi-class extension works but isn't the most elegant solution")
    print("- This algorithm laid the foundation for modern neural networks!")
    print("\nHistorical note: The perceptron sparked both AI optimism in the 1950s")
    print("and the 'AI winter' when its limitations were discovered in the 1960s.")

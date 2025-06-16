import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable

class SVM:
    """
    Support Vector Machine implementation using Sequential Minimal Optimization (SMO)
    
    Honestly, SVM was one of the trickier algorithms to implement from scratch.
    The math is pretty heavy but the intuition is simple: find the best line/hyperplane
    that separates classes with maximum margin.
    """
    
    def __init__(self, C=1.0, kernel='linear', gamma=1.0, degree=3, coef0=0.0, 
                 tol=1e-3, max_iter=1000):
        """
        Initialize SVM with hyperparameters
        
        C: regularization parameter - higher C means less regularization
        Had to tune this a lot in practice. Too high = overfitting, too low = underfitting
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma  # for RBF kernel
        self.degree = degree  # for polynomial kernel
        self.coef0 = coef0  # for polynomial/sigmoid kernels
        self.tol = tol  # tolerance for optimization
        self.max_iter = max_iter
        
        # These get set during training
        self.alphas = None
        self.b = 0  # bias term
        self.X_train = None
        self.y_train = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
    
    def _kernel_function(self, X1, X2):
        """
        Compute kernel function between data points
        
        The kernel trick is brilliant - lets us work in higher dimensions
        without explicitly computing the transformation. Math magic!
        """
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        
        elif self.kernel == 'rbf' or self.kernel == 'gaussian':
            # RBF (Radial Basis Function) kernel
            # This one is my favorite - works well for non-linear data
            if X1.ndim == 1:
                X1 = X1.reshape(1, -1)
            if X2.ndim == 1:
                X2 = X2.reshape(1, -1)
            
            # Compute squared euclidean distance
            # Using broadcasting to compute all pairwise distances efficiently
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                      np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * sq_dists)
        
        elif self.kernel == 'poly':
            # Polynomial kernel - good for non-linear data with polynomial relationships
            return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        
        elif self.kernel == 'sigmoid':
            # Sigmoid kernel - sometimes called "neural network" kernel
            return np.tanh(self.gamma * np.dot(X1, X2.T) + self.coef0)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """
        Train the SVM using SMO algorithm
        
        SMO (Sequential Minimal Optimization) breaks the big QP problem
        into smaller 2-variable problems. Much more tractable!
        """
        n_samples, n_features = X.shape
        
        # Convert labels to -1/+1 format (SVM standard)
        y = np.where(y <= 0, -1, 1)
        
        # Initialize Lagrange multipliers (alphas)
        self.alphas = np.zeros(n_samples)
        self.b = 0
        self.X_train = X
        self.y_train = y
        
        # Precompute kernel matrix - this can get memory intensive for large datasets  
        # In practice, you'd want to compute kernel values on-demand for large data
        self.K = self._kernel_function(X, X)
        
        # SMO main loop
        # This is where the magic happens - iteratively optimize pairs of alphas
        for iteration in range(self.max_iter):
            num_changed_alphas = 0
            
            for i in range(n_samples):
                # Check KKT conditions - these tell us if we're at optimum
                Ei = self._decision_function_single(i) - y[i]
                
                # KKT violation check (this took me a while to get right)
                if ((y[i] * Ei < -self.tol and self.alphas[i] < self.C) or
                    (y[i] * Ei > self.tol and self.alphas[i] > 0)):
                    
                    # Choose second alpha randomly (could be smarter about this)
                    j = self._select_j(i, n_samples)
                    Ej = self._decision_function_single(j) - y[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alphas[i].copy()
                    alpha_j_old = self.alphas[j].copy()
                    
                    # Compute bounds L and H for alpha_j
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta (second derivative of objective function)
                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue  # skip this pair
                    
                    # Update alpha_j
                    self.alphas[j] -= (y[j] * (Ei - Ej)) / eta
                    
                    # Clip alpha_j to bounds
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    
                    # Check if change is significant
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    # Update bias term b
                    # This part is crucial but easy to mess up
                    b1 = (self.b - Ei - y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, i] -
                          y[j] * (self.alphas[j] - alpha_j_old) * self.K[i, j])
                    
                    b2 = (self.b - Ej - y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, j] -
                          y[j] * (self.alphas[j] - alpha_j_old) * self.K[j, j])
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            # If no alphas changed, we're done
            if num_changed_alphas == 0:
                break
        
        # Extract support vectors (non-zero alphas)
        # These are the "important" points that define the decision boundary
        sv_indices = self.alphas > 1e-5
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_vector_alphas = self.alphas[sv_indices]
        
        print(f"Training completed after {iteration + 1} iterations")
        print(f"Found {len(self.support_vectors)} support vectors out of {n_samples} samples")
    
    def _select_j(self, i, n_samples):
        """
        Select second alpha index j != i
        
        This is simplified - better implementations use heuristics
        to choose j more intelligently (like largest |Ei - Ej|)
        """
        j = i
        while j == i:
            j = np.random.randint(0, n_samples)
        return j
    
    def _decision_function_single(self, i):
        """
        Compute decision function for single training example
        Used during training - more efficient than computing for all points
        """
        result = 0
        for j in range(len(self.alphas)):
            if self.alphas[j] > 0:
                result += self.alphas[j] * self.y_train[j] * self.K[i, j]
        return result + self.b
    
    def decision_function(self, X):
        """
        Compute decision function values
        
        This is the "raw" output before applying sign for classification
        Distance from hyperplane - positive means one class, negative the other
        """
        if self.support_vectors is None:
            raise ValueError("Model not trained yet!")
        
        # Compute kernel between test points and support vectors
        K_test = self._kernel_function(X, self.support_vectors)
        
        # Decision function: sum over support vectors
        decision = np.sum(self.support_vector_alphas * self.support_vector_labels * K_test.T, axis=0)
        return decision + self.b
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Simple: just take the sign of the decision function
        """
        return np.sign(self.decision_function(X))
    
    def predict_proba(self, X):
        """
        Predict class probabilities using Platt scaling approximation
        
        Note: This is a rough approximation. Real probability calibration
        would require additional training data and more sophisticated methods
        """
        decision = self.decision_function(X)
        # Simple sigmoid approximation - not perfectly calibrated but reasonable
        proba_positive = 1 / (1 + np.exp(-decision))
        return np.column_stack([1 - proba_positive, proba_positive])
    
    def score(self, X, y):
        """Compute accuracy on test data"""
        y_pred = self.predict(X)
        y_true = np.where(y <= 0, -1, 1)  # convert to -1/+1 format
        return np.mean(y_pred == y_true)

# Utility functions for visualization and testing
def make_classification_data(n_samples=200, n_features=2, n_classes=2, 
                           random_state=42, noise=0.1):
    """
    Generate simple 2D classification data for testing
    
    I got tired of importing sklearn just for toy datasets
    """
    np.random.seed(random_state)
    
    if n_classes == 2:
        # Create two clusters
        n_per_class = n_samples // 2
        
        # Class 0: centered around (-1, -1)
        X0 = np.random.randn(n_per_class, n_features) * 0.5 + np.array([-1, -1])
        y0 = np.zeros(n_per_class)
        
        # Class 1: centered around (1, 1)  
        X1 = np.random.randn(n_per_class, n_features) * 0.5 + np.array([1, 1])
        y1 = np.ones(n_per_class)
        
        X = np.vstack([X0, X1])
        y = np.hstack([y0, y1])
        
        # Add some noise to make it more interesting
        X += np.random.randn(*X.shape) * noise
        
    else:
        raise NotImplementedError("Only binary classification implemented")
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

def plot_svm_decision_boundary(svm, X, y, title="SVM Decision Boundary"):
    """
    Plot SVM decision boundary and support vectors
    
    Visualization really helps understand what the SVM is doing
    """
    plt.figure(figsize=(10, 8))
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Get decision function values
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.decision_function(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5, 
               linestyles=['--', '-', '--'], colors=['red', 'black', 'red'])
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, alpha=0.8)
    
    # Highlight support vectors
    if hasattr(svm, 'support_vectors') and svm.support_vectors is not None:
        plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                   s=100, facecolors='none', edgecolors='black', linewidth=2,
                   label='Support Vectors')
        plt.legend()
    
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)

# Example usage and testing
if __name__ == "__main__":
    print("Testing SVM implementation...")
    
    # Generate test data
    X, y = make_classification_data(n_samples=100, noise=0.15, random_state=42)
    
    # Split into train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Test different kernels
    kernels = ['linear', 'rbf', 'poly']
    
    for kernel in kernels:
        print(f"\n--- Testing {kernel.upper()} kernel ---")
        
        # Adjust hyperparameters based on kernel
        if kernel == 'rbf':
            svm = SVM(C=1.0, kernel=kernel, gamma=1.0, max_iter=1000)
        elif kernel == 'poly':
            svm = SVM(C=1.0, kernel=kernel, degree=2, gamma=1.0, max_iter=1000)
        else:
            svm = SVM(C=1.0, kernel=kernel, max_iter=1000)
        
        # Train the model
        svm.fit(X_train, y_train)
        
        # Make predictions
        train_score = svm.score(X_train, y_train)
        test_score = svm.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        print(f"Number of support vectors: {len(svm.support_vectors)}")
        
        # Plot decision boundary (only for 2D data)
        if X.shape[1] == 2:
            plot_svm_decision_boundary(svm, X, y, 
                                     title=f"SVM with {kernel.upper()} kernel")
            plt.tight_layout()
            plt.show()
    
    print("\n--- Comparing different C values (regularization) ---")
    # Test effect of regularization parameter C
    C_values = [0.1, 1.0, 10.0, 100.0]
    
    for C in C_values:
        svm = SVM(C=C, kernel='rbf', gamma=1.0, max_iter=1000)
        svm.fit(X_train, y_train)
        
        train_score = svm.score(X_train, y_train)
        test_score = svm.score(X_test, y_test)
        
        print(f"C={C:5.1f}: Train={train_score:.3f}, Test={test_score:.3f}, "
              f"Support Vectors={len(svm.support_vectors)}")
    
    print("\nDone! Higher C usually means more complex models (more overfitting risk)")
    print("Lower C means more regularization (smoother decision boundaries)")

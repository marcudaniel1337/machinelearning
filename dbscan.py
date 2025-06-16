import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize DBSCAN clustering algorithm
        
        eps: maximum distance between two points to be considered neighbors
             (think of it as the radius of our neighborhood search)
        min_samples: minimum number of points required to form a dense region
                    (including the point itself - so if min_samples=5, we need 
                     the point plus 4 neighbors to call it a core point)
        """
        self.eps = eps
        self.min_samples = min_samples
        
        # We'll keep track of labels for each point
        # -1 means noise, 0+ means cluster ID
        self.labels_ = None
        
        # Internal tracking - let's use constants to make code readable
        self.UNVISITED = -2  # Haven't looked at this point yet
        self.NOISE = -1      # Point is considered noise/outlier
    
    def _euclidean_distance(self, point1, point2):
        """
        Calculate straight-line distance between two points
        Could use other distance metrics here if needed
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def _get_neighbors(self, X, point_idx):
        """
        Find all points within eps distance of the given point
        
        This is where we do the heavy lifting - for each point, we need to
        check every other point to see if it's close enough to be a neighbor.
        In a real implementation, you'd probably use a spatial index like
        KDTree to make this faster, but let's keep it simple for understanding.
        """
        neighbors = []
        
        # Check every point in our dataset
        for i, point in enumerate(X):
            # Don't include the point as its own neighbor in our count
            if i != point_idx:
                # If it's close enough, add it to neighbors list
                if self._euclidean_distance(X[point_idx], point) <= self.eps:
                    neighbors.append(i)
        
        return neighbors
    
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        """
        Grow a cluster starting from a core point
        
        This is the heart of DBSCAN - we start with a core point and its neighbors,
        then recursively add more points if they're also core points.
        We use a queue (breadth-first search) to systematically explore.
        """
        # Mark the starting point as part of this cluster
        self.labels_[point_idx] = cluster_id
        
        # Use a queue to process neighbors systematically
        # We'll check each neighbor, and if it turns out to be a core point too,
        # we'll add ITS neighbors to our queue for processing
        neighbor_queue = deque(neighbors)
        
        while neighbor_queue:
            current_neighbor = neighbor_queue.popleft()
            
            # If we haven't visited this neighbor yet, let's explore it
            if self.labels_[current_neighbor] == self.UNVISITED:
                self.labels_[current_neighbor] = cluster_id
                
                # Check if this neighbor is also a core point
                neighbor_neighbors = self._get_neighbors(X, current_neighbor)
                
                # If it has enough neighbors to be a core point,
                # add its neighbors to our exploration queue
                if len(neighbor_neighbors) >= self.min_samples - 1:
                    # Only add neighbors we haven't already processed
                    for nn in neighbor_neighbors:
                        if self.labels_[nn] == self.UNVISITED:
                            neighbor_queue.append(nn)
            
            # Special case: if this neighbor was previously marked as noise,
            # but now we're reaching it from a core point, it should join the cluster
            # (border point behavior)
            elif self.labels_[current_neighbor] == self.NOISE:
                self.labels_[current_neighbor] = cluster_id
    
    def fit(self, X):
        """
        Main DBSCAN algorithm - this is where everything comes together
        
        The algorithm is beautifully simple:
        1. For each unvisited point, check if it's a core point
        2. If yes, start a new cluster and grow it
        3. If no, mark it as noise (might get rescued later by a core point)
        """
        n_points = len(X)
        
        # Initialize all points as unvisited
        self.labels_ = np.full(n_points, self.UNVISITED)
        
        cluster_id = 0  # Start numbering clusters from 0
        
        # Go through each point systematically
        for point_idx in range(n_points):
            # Skip if we've already processed this point
            if self.labels_[point_idx] != self.UNVISITED:
                continue
            
            # Find all neighbors of this point
            neighbors = self._get_neighbors(X, point_idx)
            
            # Check if this point qualifies as a core point
            # (remember: min_samples includes the point itself, so we need
            #  at least min_samples-1 actual neighbors)
            if len(neighbors) >= self.min_samples - 1:
                # This is a core point! Start a new cluster
                self._expand_cluster(X, point_idx, neighbors, cluster_id)
                cluster_id += 1  # Next cluster will get the next ID
            else:
                # Not enough neighbors - mark as noise for now
                # (might get rescued later if a core point reaches it)
                self.labels_[point_idx] = self.NOISE
        
        return self
    
    def fit_predict(self, X):
        """
        Convenience method: fit the model and return cluster labels
        """
        self.fit(X)
        return self.labels_

# Let's test our implementation with some sample data
def demo_dbscan():
    """
    Create some sample data and show how DBSCAN works
    """
    print("Creating sample data with 3 distinct blobs...")
    
    # Generate sample data - 3 clusters with some noise
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, 
                      center_box=(-10.0, 10.0), random_state=42)
    
    # Add some noise points
    noise_points = np.random.uniform(-12, 12, (20, 2))
    X = np.vstack([X, noise_points])
    
    print(f"Total data points: {len(X)}")
    
    # Apply our DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    
    # Count clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Found {n_clusters} clusters")
    print(f"Identified {n_noise} noise points")
    
    # Visualize results
    plt.figure(figsize=(10, 8))
    
    # Plot each cluster in a different color
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Black for noise points
            color = 'black'
            marker = 'x'
            label_name = 'Noise'
        else:
            marker = 'o'
            label_name = f'Cluster {label}'
        
        # Get points belonging to this cluster/noise
        cluster_points = X[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=[color], marker=marker, s=50, label=label_name)
    
    plt.title(f'DBSCAN Clustering (eps={dbscan.eps}, min_samples={dbscan.min_samples})')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some explanatory text
    plt.figtext(0.02, 0.02, 
                f"Algorithm found {n_clusters} clusters and {n_noise} noise points.\n"
                f"Points within distance {dbscan.eps} are considered neighbors.\n"
                f"Regions need at least {dbscan.min_samples} points to form a cluster.",
                fontsize=9, ha='left')
    
    plt.tight_layout()
    plt.show()
    
    return X, labels

# Run the demo if this script is executed directly
if __name__ == "__main__":
    print("DBSCAN Implementation Demo")
    print("=" * 50)
    
    # Run our demo
    X, labels = demo_dbscan()
    
    print("\nHow DBSCAN works:")
    print("1. For each point, find all neighbors within 'eps' distance")
    print("2. If a point has >= 'min_samples' neighbors, it's a 'core point'")
    print("3. Core points start clusters and recursively add their neighbors")
    print("4. Points reachable from core points become part of the cluster")
    print("5. Points not reachable from any core point are marked as noise")
    
    print(f"\nIn our example:")
    print(f"- eps = 1.5 (neighborhood radius)")
    print(f"- min_samples = 5 (minimum points needed for a dense region)")
    print(f"- Result: {len(set(labels)) - (1 if -1 in labels else 0)} clusters found")

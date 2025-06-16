import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from collections import defaultdict
import heapq

class HierarchicalClustering:
    """
    Hierarchical Clustering implementation using agglomerative approach
    
    This was one of the first clustering algorithms I really "got" - the concept
    is so intuitive! Start with each point as its own cluster, then keep merging
    the closest ones until you have the structure you want.
    
    The dendrograms you get are beautiful and actually tell a story about your data.
    Much more interpretable than k-means in my opinion.
    """
    
    def __init__(self, n_clusters=2, linkage='ward', metric='euclidean'):
        """
        Initialize hierarchical clustering
        
        linkage: how to measure distance between clusters
        - 'single': minimum distance (can create long chains - not always great)
        - 'complete': maximum distance (creates compact clusters)
        - 'average': average distance (good balance)
        - 'ward': minimizes within-cluster variance (my personal favorite)
        
        Took me a while to understand the differences, but ward usually works best
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        
        # These get filled during fitting
        self.labels_ = None
        self.linkage_matrix_ = None  # for dendrogram plotting
        self.distances_ = []  # track merge distances
        self.cluster_history_ = []  # track how clusters merged
        
    def _calculate_distance(self, point1, point2):
        """
        Calculate distance between two points
        
        Starting simple with euclidean - could add other metrics later
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(point1 - point2))
        elif self.metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            dot_product = np.dot(point1, point2)
            norms = np.linalg.norm(point1) * np.linalg.norm(point2)
            if norms == 0:
                return 0
            return 1 - (dot_product / norms)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _cluster_distance(self, cluster1_points, cluster2_points):
        """
        Calculate distance between two clusters based on linkage method
        
        This is where the different linkage methods really matter.
        Each one gives a different "personality" to the clustering.
        """
        if self.linkage == 'single':
            # Single linkage: minimum distance between any two points
            # Can create long, stretched clusters (chaining effect)
            min_dist = float('inf')
            for p1 in cluster1_points:
                for p2 in cluster2_points:
                    dist = self._calculate_distance(p1, p2)
                    min_dist = min(min_dist, dist)
            return min_dist
            
        elif self.linkage == 'complete':
            # Complete linkage: maximum distance between any two points
            # Creates more compact, spherical clusters
            max_dist = 0
            for p1 in cluster1_points:
                for p2 in cluster2_points:
                    dist = self._calculate_distance(p1, p2)
                    max_dist = max(max_dist, dist)
            return max_dist
            
        elif self.linkage == 'average':
            # Average linkage: average distance between all pairs
            # Good compromise between single and complete
            total_dist = 0
            count = 0
            for p1 in cluster1_points:
                for p2 in cluster2_points:
                    total_dist += self._calculate_distance(p1, p2)
                    count += 1
            return total_dist / count if count > 0 else 0
            
        elif self.linkage == 'ward':
            # Ward linkage: minimizes within-cluster sum of squares
            # Usually produces the most balanced, interpretable clusters
            
            # Calculate centroids
            centroid1 = np.mean(cluster1_points, axis=0)
            centroid2 = np.mean(cluster2_points, axis=0)
            
            # Calculate current within-cluster sum of squares
            ss1 = np.sum([(self._calculate_distance(p, centroid1)) ** 2 for p in cluster1_points])
            ss2 = np.sum([(self._calculate_distance(p, centroid2)) ** 2 for p in cluster2_points])
            
            # Calculate merged cluster properties
            merged_points = np.vstack([cluster1_points, cluster2_points])
            merged_centroid = np.mean(merged_points, axis=0)
            merged_ss = np.sum([(self._calculate_distance(p, merged_centroid)) ** 2 for p in merged_points])
            
            # Ward distance is the increase in sum of squares
            return merged_ss - (ss1 + ss2)
        
        else:
            raise ValueError(f"Unknown linkage method: {self.linkage}")
    
    def fit(self, X):
        """
        Perform hierarchical clustering
        
        The algorithm is straightforward but can be slow for large datasets:
        1. Start with each point as its own cluster
        2. Find the two closest clusters
        3. Merge them
        4. Repeat until desired number of clusters
        
        Time complexity is O(n³) - gets expensive quickly!
        """
        n_samples = len(X)
        
        # Initialize: each point is its own cluster
        # Using dict to map cluster_id -> list of point indices
        clusters = {i: [i] for i in range(n_samples)}
        cluster_points = {i: [X[i]] for i in range(n_samples)}
        
        # For tracking the dendrogram structure
        linkage_matrix = []
        next_cluster_id = n_samples  # new cluster IDs start after original points
        
        # Keep merging until we have the desired number of clusters
        while len(clusters) > self.n_clusters:
            # Find the two closest clusters
            # This is the expensive part - checking all pairs
            min_distance = float('inf')
            merge_clusters = None
            
            cluster_ids = list(clusters.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    cluster1_id = cluster_ids[i]
                    cluster2_id = cluster_ids[j]
                    
                    points1 = cluster_points[cluster1_id]
                    points2 = cluster_points[cluster2_id]
                    
                    distance = self._cluster_distance(points1, points2)
                    
                    if distance < min_distance:
                        min_distance = distance
                        merge_clusters = (cluster1_id, cluster2_id)
            
            # Merge the two closest clusters
            cluster1_id, cluster2_id = merge_clusters
            
            # Create new merged cluster
            merged_indices = clusters[cluster1_id] + clusters[cluster2_id]
            merged_points = cluster_points[cluster1_id] + cluster_points[cluster2_id]
            
            # Record merge for dendrogram
            # Format: [cluster1_id, cluster2_id, distance, size_of_new_cluster]
            linkage_matrix.append([
                cluster1_id,
                cluster2_id, 
                min_distance,
                len(merged_indices)
            ])
            
            # Update cluster structures
            clusters[next_cluster_id] = merged_indices
            cluster_points[next_cluster_id] = merged_points
            
            # Remove old clusters
            del clusters[cluster1_id]
            del clusters[cluster2_id]
            del cluster_points[cluster1_id]
            del cluster_points[cluster2_id]
            
            # Track progress
            self.distances_.append(min_distance)
            self.cluster_history_.append({
                'merged': merge_clusters,
                'distance': min_distance,
                'n_clusters': len(clusters)
            })
            
            next_cluster_id += 1
        
        # Assign labels to points
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_label, (cluster_id, point_indices) in enumerate(clusters.items()):
            for point_idx in point_indices:
                self.labels_[point_idx] = cluster_label
        
        # Store linkage matrix for dendrogram plotting
        self.linkage_matrix_ = np.array(linkage_matrix)
        
        print(f"Clustering completed! Found {len(clusters)} clusters.")
        print(f"Final merge distances: {self.distances_[-5:]}")  # show last few merges
        
        return self
    
    def fit_predict(self, X):
        """Convenience method to fit and return labels"""
        return self.fit(X).labels_
    
    def plot_dendrogram(self, X=None, figsize=(12, 8), **kwargs):
        """
        Plot dendrogram showing cluster hierarchy
        
        This is the coolest part of hierarchical clustering - you can actually
        see how the clusters were built! Much more interpretable than k-means.
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Must fit the model first!")
        
        plt.figure(figsize=figsize)
        
        # Plot dendrogram using scipy's function (they did the hard layout work)
        dendrogram_data = dendrogram(
            self.linkage_matrix_,
            **kwargs
        )
        
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage} linkage)')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        
        # Add some helpful annotations
        if len(self.distances_) > 0:
            plt.axhline(y=self.distances_[-1], color='r', linestyle='--', 
                       alpha=0.7, label=f'Cut for {self.n_clusters} clusters')
            plt.legend()
        
        plt.tight_layout()
        return dendrogram_data
    
    def plot_clusters(self, X, figsize=(10, 8)):
        """
        Plot clusters in 2D space
        
        Only works for 2D data, but great for visualizing results
        """
        if X.shape[1] != 2:
            print("Can only plot 2D data")
            return
        
        if self.labels_ is None:
            raise ValueError("Must fit the model first!")
        
        plt.figure(figsize=figsize)
        
        # Use different colors for each cluster
        colors = plt.cm.Set1(np.linspace(0, 1, self.n_clusters))
        
        for cluster_id in range(self.n_clusters):
            mask = self.labels_ == cluster_id
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=[colors[cluster_id]], 
                       label=f'Cluster {cluster_id}',
                       alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        plt.title(f'Hierarchical Clustering Results ({self.linkage} linkage)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

class DivisiveClustering:
    """
    Divisive (top-down) hierarchical clustering
    
    Less common than agglomerative, but sometimes useful.
    Start with all points in one cluster, then recursively split.
    
    Honestly, this is more complex to implement well and usually
    agglomerative works better. But it's cool to have both approaches!
    """
    
    def __init__(self, n_clusters=2, metric='euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.labels_ = None
    
    def _calculate_distance(self, point1, point2):
        """Same distance function as agglomerative version"""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(point1 - point2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _find_best_split(self, X, indices):
        """
        Find the best way to split a cluster into two
        
        Using a simple approach: try splitting along each dimension
        at the median. More sophisticated methods exist but this works.
        """
        if len(indices) <= 1:
            return None, None, 0
        
        best_split = None
        best_score = -1
        
        # Try splitting along each dimension
        for dim in range(X.shape[1]):
            values = X[indices, dim]
            median = np.median(values)
            
            left_mask = values <= median
            right_mask = values > median
            
            # Skip if split doesn't actually divide the points
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            left_indices = np.array(indices)[left_mask]
            right_indices = np.array(indices)[right_mask]
            
            # Score the split based on within-cluster variance reduction
            # Lower variance = tighter clusters = better split
            left_center = np.mean(X[left_indices], axis=0)
            right_center = np.mean(X[right_indices], axis=0)
            
            left_var = np.sum([(self._calculate_distance(X[i], left_center)) ** 2 
                              for i in left_indices])
            right_var = np.sum([(self._calculate_distance(X[i], right_center)) ** 2 
                               for i in right_indices])
            
            # Original variance
            center = np.mean(X[indices], axis=0)
            orig_var = np.sum([(self._calculate_distance(X[i], center)) ** 2 
                              for i in indices])
            
            # Score is reduction in variance
            score = orig_var - (left_var + right_var)
            
            if score > best_score:
                best_score = score
                best_split = (list(left_indices), list(right_indices))
        
        return best_split, best_score
    
    def fit(self, X):
        """
        Perform divisive clustering using recursive splitting
        
        This is basically a binary tree construction problem.
        Keep splitting until we have enough clusters.
        """
        n_samples = len(X)
        
        # Start with all points in one cluster
        clusters = [list(range(n_samples))]
        
        # Keep splitting until we have enough clusters
        while len(clusters) < self.n_clusters:
            # Find the largest cluster to split
            # Could use other criteria like highest variance
            largest_cluster_idx = max(range(len(clusters)), 
                                    key=lambda i: len(clusters[i]))
            
            cluster_to_split = clusters[largest_cluster_idx]
            
            # Find best split for this cluster
            split_result, score = self._find_best_split(X, cluster_to_split)
            
            if split_result is None:
                print(f"Cannot split further. Stopping at {len(clusters)} clusters.")
                break
            
            left_cluster, right_cluster = split_result
            
            # Replace the original cluster with two new ones
            clusters[largest_cluster_idx] = left_cluster
            clusters.append(right_cluster)
        
        # Assign labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster_indices in enumerate(clusters):
            for point_idx in cluster_indices:
                self.labels_[point_idx] = cluster_id
        
        print(f"Divisive clustering completed! Found {len(clusters)} clusters.")
        return self
    
    def fit_predict(self, X):
        return self.fit(X).labels_

def generate_hierarchical_data(n_samples=150, n_centers=3, random_state=42):
    """
    Generate data that's good for hierarchical clustering
    
    Creating nested, tree-like structure that hierarchical clustering should handle well
    """
    np.random.seed(random_state)
    
    if n_centers == 3:
        # Create three main clusters with subclusters
        n_per_main = n_samples // 3
        
        # Main cluster 1: two subclusters
        subcluster1a = np.random.randn(n_per_main//2, 2) * 0.3 + np.array([-3, -3])
        subcluster1b = np.random.randn(n_per_main//2, 2) * 0.3 + np.array([-2, -3])
        
        # Main cluster 2: compact single cluster
        cluster2 = np.random.randn(n_per_main, 2) * 0.4 + np.array([3, 0])
        
        # Main cluster 3: three subclusters
        n_sub = n_per_main // 3
        subcluster3a = np.random.randn(n_sub, 2) * 0.2 + np.array([0, 3])
        subcluster3b = np.random.randn(n_sub, 2) * 0.2 + np.array([1, 3.5])
        subcluster3c = np.random.randn(n_per_main - 2*n_sub, 2) * 0.2 + np.array([0.5, 2.5])
        
        X = np.vstack([subcluster1a, subcluster1b, cluster2, 
                      subcluster3a, subcluster3b, subcluster3c])
        
        # True labels for reference (though hierarchical clustering is unsupervised)
        y_true = np.array([0] * len(subcluster1a) + [0] * len(subcluster1b) +
                         [1] * len(cluster2) + 
                         [2] * len(subcluster3a) + [2] * len(subcluster3b) + [2] * len(subcluster3c))
        
    else:
        # Fallback: simple Gaussian clusters
        centers = np.random.randn(n_centers, 2) * 3
        n_per_cluster = n_samples // n_centers
        
        X_list = []
        y_list = []
        
        for i, center in enumerate(centers):
            cluster_size = n_per_cluster if i < n_centers - 1 else n_samples - i * n_per_cluster
            cluster_points = np.random.randn(cluster_size, 2) * 0.5 + center
            X_list.append(cluster_points)
            y_list.extend([i] * cluster_size)
        
        X = np.vstack(X_list)
        y_true = np.array(y_list)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    return X[indices], y_true[indices]

def compare_linkage_methods(X, n_clusters=3):
    """
    Compare different linkage methods on the same data
    
    This is really helpful for understanding how each method behaves differently
    """
    linkage_methods = ['single', 'complete', 'average', 'ward']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, linkage in enumerate(linkage_methods):
        # Fit hierarchical clustering
        hc = HierarchicalClustering(n_clusters=n_clusters, linkage=linkage)
        labels = hc.fit_predict(X)
        
        # Plot results
        ax = axes[i]
        colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
        
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            ax.scatter(X[mask, 0], X[mask, 1], 
                      c=[colors[cluster_id]], 
                      label=f'Cluster {cluster_id}',
                      alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_title(f'{linkage.capitalize()} Linkage')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.suptitle('Comparison of Linkage Methods', y=1.02, fontsize=16)

# Example usage and comprehensive testing
if __name__ == "__main__":
    print("Testing Hierarchical Clustering Implementation")
    print("=" * 50)
    
    # Generate test data with nested structure
    print("\n1. Generating hierarchical test data...")
    X, y_true = generate_hierarchical_data(n_samples=120, n_centers=3, random_state=42)
    
    print(f"Generated {len(X)} points with {len(np.unique(y_true))} true clusters")
    
    # Test agglomerative clustering with Ward linkage
    print("\n2. Testing Agglomerative Clustering (Ward linkage)...")
    hc_ward = HierarchicalClustering(n_clusters=3, linkage='ward')
    labels_ward = hc_ward.fit_predict(X)
    
    print(f"Cluster sizes: {np.bincount(labels_ward)}")
    
    # Plot results
    hc_ward.plot_clusters(X)
    plt.show()
    
    # Plot dendrogram
    print("\n3. Plotting dendrogram...")
    hc_ward.plot_dendrogram()
    plt.show()
    
    # Compare different linkage methods
    print("\n4. Comparing linkage methods...")
    compare_linkage_methods(X, n_clusters=3)
    plt.show()
    
    # Test different numbers of clusters
    print("\n5. Testing different numbers of clusters...")
    for n_clust in [2, 3, 4, 5]:
        hc = HierarchicalClustering(n_clusters=n_clust, linkage='ward')
        labels = hc.fit_predict(X)
        
        # Calculate simple within-cluster sum of squares as quality metric
        wcss = 0
        for cluster_id in range(n_clust):
            cluster_points = X[labels == cluster_id]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                wcss += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
        
        print(f"n_clusters={n_clust}: WCSS={wcss:.2f}, Sizes={np.bincount(labels)}")
    
    # Test divisive clustering
    print("\n6. Testing Divisive Clustering...")
    dc = DivisiveClustering(n_clusters=3)
    labels_divisive = dc.fit_predict(X)
    
    print(f"Divisive cluster sizes: {np.bincount(labels_divisive)}")
    
    # Compare agglomerative vs divisive
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Agglomerative
    colors = plt.cm.Set1(np.linspace(0, 1, 3))
    for cluster_id in range(3):
        mask = labels_ward == cluster_id
        ax1.scatter(X[mask, 0], X[mask, 1], 
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                   alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax1.set_title('Agglomerative Clustering')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Divisive
    for cluster_id in range(3):
        mask = labels_divisive == cluster_id
        ax2.scatter(X[mask, 0], X[mask, 1], 
                   c=[colors[cluster_id]], label=f'Cluster {cluster_id}',
                   alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    ax2.set_title('Divisive Clustering')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance analysis
    print("\n7. Performance considerations...")
    sizes = [50, 100, 200]
    
    import time
    for size in sizes:
        X_test, _ = generate_hierarchical_data(n_samples=size, random_state=42)
        
        start_time = time.time()
        hc = HierarchicalClustering(n_clusters=3, linkage='ward')
        hc.fit(X_test)
        end_time = time.time()
        
        print(f"n_samples={size:3d}: {end_time - start_time:.3f} seconds")
    
    print("\nKey Insights:")
    print("- Ward linkage usually produces the most balanced, interpretable clusters")
    print("- Single linkage can create long chains (chaining effect)")
    print("- Complete linkage creates compact, spherical clusters")
    print("- Dendrograms are incredibly useful for understanding cluster structure")
    print("- Algorithm is O(n³) - gets slow for large datasets")
    print("- Unlike k-means, no need to specify number of clusters upfront!")
    print("- Great for exploratory data analysis and understanding data structure")

import random
import math

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in any dimension.
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def initialize_centroids(data, k):
    """
    Randomly pick k unique points from data as initial centroids.
    """
    return random.sample(data, k)

def assign_clusters(data, centroids):
    """
    Assign each data point to the closest centroid, forming clusters.
    Returns a list where index corresponds to the data point,
    and value is the index of the assigned centroid.
    """
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_centroid_idx = distances.index(min(distances))
        clusters.append(closest_centroid_idx)
    return clusters

def update_centroids(data, clusters, k):
    """
    Calculate new centroids as the mean of points assigned to each cluster.
    """
    new_centroids = []
    for cluster_idx in range(k):
        # Collect all points assigned to this cluster
        cluster_points = [point for point, c_idx in zip(data, clusters) if c_idx == cluster_idx]
        
        if cluster_points:
            # Compute mean for each dimension
            mean = [sum(dim) / len(cluster_points) for dim in zip(*cluster_points)]
        else:
            # If a cluster lost all points, randomly reinitialize its centroid
            mean = random.choice(data)
        new_centroids.append(mean)
    return new_centroids

def k_means(data, k, max_iterations=100, tolerance=1e-4):
    """
    Perform K-Means clustering on data.

    Args:
        data (list of lists): Dataset where each element is a point.
        k (int): Number of clusters.
        max_iterations (int): Limit on number of iterations to prevent infinite loops.
        tolerance (float): Threshold to declare convergence (centroid movement).

    Returns:
        centroids (list): Final centroid positions.
        clusters (list): Cluster assignment for each data point.
    """
    # Step 1: Initialize centroids randomly from data points
    centroids = initialize_centroids(data, k)
    
    for iteration in range(max_iterations):
        # Step 2: Assign points to the nearest centroid
        clusters = assign_clusters(data, centroids)
        
        # Step 3: Update centroids based on current clusters
        new_centroids = update_centroids(data, clusters, k)
        
        # Step 4: Check how much centroids have moved (convergence)
        movements = [euclidean_distance(c_old, c_new) for c_old, c_new in zip(centroids, new_centroids)]
        max_movement = max(movements)
        
        # Debug print to watch progress (comment out if too verbose)
        print(f"Iteration {iteration + 1}, max centroid movement: {max_movement:.6f}")
        
        # If centroids have barely moved, we've converged
        if max_movement < tolerance:
            print("Convergence reached!")
            break
        
        centroids = new_centroids
    
    return centroids, clusters


if __name__ == "__main__":
    # Example dataset: 2D points
    data_points = [
        [1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0],
        [1.0, 0.6], [9.0, 11.0], [8.0, 2.0], [10.0, 2.0],
        [9.0, 3.0],
    ]

    k = 3  # Number of clusters

    centroids, clusters = k_means(data_points, k)

    print("\nFinal centroids:")
    for idx, centroid in enumerate(centroids):
        print(f"Centroid {idx}: {centroid}")

    print("\nData point assignments:")
    for point, cluster_idx in zip(data_points, clusters):
        print(f"Point {point} is in cluster {cluster_idx}")

import math

def mean_vector(data):
    """
    Calculate the mean of each feature (column) in the dataset.
    """
    n_samples = len(data)
    n_features = len(data[0])
    means = [0] * n_features

    for i in range(n_features):
        means[i] = sum(row[i] for row in data) / n_samples
    return means

def center_data(data, means):
    """
    Subtract the mean from each feature to center the data around zero.
    """
    centered = []
    for row in data:
        centered.append([value - mean for value, mean in zip(row, means)])
    return centered

def covariance_matrix(data):
    """
    Calculate the covariance matrix of the dataset.
    data is assumed to be mean-centered.
    """
    n_samples = len(data)
    n_features = len(data[0])
    
    # Initialize covariance matrix with zeros
    cov_matrix = [[0 for _ in range(n_features)] for _ in range(n_features)]
    
    for i in range(n_features):
        for j in range(n_features):
            cov_ij = 0
            for k in range(n_samples):
                cov_ij += data[k][i] * data[k][j]
            cov_matrix[i][j] = cov_ij / (n_samples - 1)
    return cov_matrix

def transpose(matrix):
    """
    Transpose a matrix.
    """
    return list(map(list, zip(*matrix)))

def matrix_vector_mult(matrix, vector):
    """
    Multiply a matrix by a vector.
    """
    result = []
    for row in matrix:
        s = sum(r * v for r, v in zip(row, vector))
        result.append(s)
    return result

def vector_norm(vector):
    """
    Calculate Euclidean norm of a vector.
    """
    return math.sqrt(sum(x ** 2 for x in vector))

def normalize_vector(vector):
    """
    Normalize a vector to have length 1.
    """
    norm = vector_norm(vector)
    return [x / norm for x in vector]

def power_iteration(matrix, num_iters=1000, tolerance=1e-9):
    """
    Find the dominant eigenvector and eigenvalue of matrix using power iteration.
    """
    n = len(matrix)
    b_k = [1.0] * n  # initial guess vector

    for _ in range(num_iters):
        # Multiply matrix by current vector
        b_k1 = matrix_vector_mult(matrix, b_k)

        # Normalize
        b_k1_norm = vector_norm(b_k1)
        b_k1 = [x / b_k1_norm for x in b_k1]

        # Check for convergence
        diff = sum(abs(x - y) for x, y in zip(b_k, b_k1))
        if diff < tolerance:
            break
        b_k = b_k1

    # Approximate eigenvalue using Rayleigh quotient
    Av = matrix_vector_mult(matrix, b_k)
    eigenvalue = sum(x * y for x, y in zip(b_k, Av))

    return eigenvalue, b_k

def deflate_matrix(matrix, eigenvalue, eigenvector):
    """
    Deflate matrix by removing the outer product of eigenvector scaled by eigenvalue.
    """
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]
    return matrix

def pca(data, num_components):
    """
    Perform PCA on the dataset and return the top num_components principal components.
    """
    means = mean_vector(data)
    centered = center_data(data, means)
    cov_mat = covariance_matrix(centered)

    components = []
    eigenvalues = []

    for _ in range(num_components):
        eigval, eigvec = power_iteration(cov_mat)
        components.append(eigvec)
        eigenvalues.append(eigval)
        cov_mat = deflate_matrix(cov_mat, eigval, eigvec)

    return components, eigenvalues, means

def project_data(data, components, means):
    """
    Project original data onto the principal components.
    """
    centered = center_data(data, means)
    projections = []
    for row in centered:
        proj = []
        for comp in components:
            # Dot product between row and component vector
            val = sum(x * y for x, y in zip(row, comp))
            proj.append(val)
        projections.append(proj)
    return projections

if __name__ == "__main__":
    # Example data: rows are samples, columns are features
    data = [
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9]
    ]

    num_components = 2
    components, eigenvalues, means = pca(data, num_components)

    print("Principal Components:")
    for i, comp in enumerate(components):
        print(f"Component {i+1}: {comp} with eigenvalue {eigenvalues[i]:.4f}")

    projected = project_data(data, components, means)
    print("\nProjected data:")
    for proj in projected:
        print(proj)

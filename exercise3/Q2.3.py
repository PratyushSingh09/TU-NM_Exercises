#Solution 3 

import numpy as np

def power_method(A, max_iterations=1000, tolerance=1e-5):

    n = A.shape[0]
    x = np.random.rand(n)  # Initial random guess
    x = x / np.linalg.norm(x)  # Here, we normalize initial vector

    eigenvalue = 0
    for iteration in range(max_iterations):
        # Apply matrix to the vector
        Ax = np.dot(A, x)
        
        # Calculate the eigenvalue approximation
        new_eigenvalue = np.dot(x, Ax)
        
        # Normalization
        x = Ax / np.linalg.norm(Ax)
        
        # Convergence check
        if np.abs(new_eigenvalue - eigenvalue) < tolerance:
            break
        
        eigenvalue = new_eigenvalue
    
    return eigenvalue, x

def aitken_acceleration(eigenvalues):

    if len(eigenvalues) < 3:
        raise ValueError("We need at least three successive iterates for Aitken's method")
    
    # Use the last three estimates
    lambda_k1, lambda_k2, lambda_k3 = eigenvalues[-3:]
    
    # Calculate the accelerated eigenvalue
    # accelerated_lambda = lambda_k3 - (lambda_k3 - lambda_k2)**2 / (lambda_k3 - 2*lambda_k2 + lambda_k1)
    accelerated_lambda = (lambda_k1*lambda_k3 - lambda_k2**2) / (lambda_k3 - 2*lambda_k2 + lambda_k1)
    
    return accelerated_lambda

def deflate_matrix(A, eigenvalue, eigenvector):
    """
    Removes the contribution of the found eigenvalue and eigenvector from matrix A.
    """
    # Deflation formula: A' = A - eigenvalue * (eigenvector @ eigenvector.T)
    deflated_A = A - eigenvalue * np.outer(eigenvector, eigenvector)
    return deflated_A


# Define the corrected matrix A
A = np.array([
    [6, 5, -5],
    [2, 6, -2],
    [2, 5, -1]
], dtype=float)

# Step 1: Find the largest eigenvalue and eigenvector using the Power Method
eigenvalue, eigenvector = power_method(A)

print("Largest Eigenvalue (Power Method):", eigenvalue)
print("Corresponding Eigenvector (Power Method):", eigenvector)

# Step 2: Use Aitken’s acceleration to improve the eigenvalue estimate
# Run the Power Method multiple times to collect successive eigenvalues
iterations = 1000
eigenvalues = []

for _ in range(iterations):
    eigenvalue, _ = power_method(A)
    eigenvalues.append(eigenvalue)

# Apply Aitken’s acceleration on the last three estimates
improved_eigenvalue = aitken_acceleration(eigenvalues)
print("Improved Largest Eigenvalue (Aitken's Acceleration):", improved_eigenvalue)

# Deflate the matrix
deflated_A = deflate_matrix(A, eigenvalue, eigenvector)

# Find the second eigenvalue and eigenvector using the Power Method on the deflated matrix
second_eigenvalue, second_eigenvector = power_method(deflated_A)
print(f"Second Eigenvalue: {second_eigenvalue}")
print(f"Corresponding Eigenvector: {second_eigenvector}")

# Deflate again for the third eigenvalue
deflated_A2 = deflate_matrix(deflated_A, second_eigenvalue, second_eigenvector)

# Find the third eigenvalue and eigenvector using the Power Method on the second deflated matrix
third_eigenvalue, third_eigenvector = power_method(deflated_A2)
print(f"Third Eigenvalue: {third_eigenvalue}")
print(f"Corresponding Eigenvector: {third_eigenvector}")

# Step 3 (Optional): Calculate all eigenvalues using numpy's built-in function (Verification)
all_eigenvalues, _ = np.linalg.eig(A)
print("All Eigenvalues (numpy):", all_eigenvalues)








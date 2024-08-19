import numpy as np

# Define the data points
X = np.array([[1, 1], [1, 0], [0, 0]])  # Feature matrix with 3 data points
y = np.array([1, -1, 1])  # Corresponding labels, assume y can take values 1 or -1

# Regularization parameter lambda (use a large value for lambda)
lmbda = 10  # Example value, this should be considered "large"

# Calculate the weights using the approximation for large lambda
# We solve (X^T * Y) / lambda = w
# where Y is a diagonal matrix with y on its diagonal
Y = np.diag(y)  # Create a diagonal matrix from y
W = np.linalg.solve(lmbda * np.eye(X.shape[1]), np.dot(X.T, Y))  # Solve for weights

# Extracting the weights
w1, w2 = W[:, 0]  # Unpack the weights
print(f"w1: {w1}, w2: {w2}")  # Print the weights

# Function to apply inverse transform to a similarity measure
def inverse_transform(similarity):
    if similarity <= 0:
        return np.inf  # Return infinity if similarity is non-positive
    return 1 / similarity - 1  # Apply the inverse transformation

# Function to apply negative logarithmic transform to a similarity measure
def negative_log_transform(similarity):
    if similarity <= 0:
        return np.inf  # Return infinity if similarity is non-positive
    return -np.log(similarity)  # Apply the negative log transformation

# Example similarity value
similarity_value = 0.5

# Gradient Descent function to optimize model parameters
def gradient_descent(X, y, alpha, beta, learning_rate, epochs):
    N = X.shape[0]  # Number of samples
    for epoch in range(epochs):
        grad_alpha = 0
        grad_beta = 0
        for n in range(N):
            # Forward pass to calculate prediction
            y_n = forward_pass(X[n], alpha, beta)
            # Calculate gradients for the sample
            grad_alpha_n, grad_beta_n = backward_pass(X[n], y[n], y_n, alpha, beta)
            # Accumulate the gradients over the dataset
            grad_alpha += grad_alpha_n
            grad_beta += grad_beta_n
        # Update parameters
        alpha -= learning_rate * grad_alpha / N
        beta -= learning_rate * grad_beta / N
    return alpha, beta  # Return updated parameters

# Stochastic Gradient Descent function to optimize model parameters
def stochastic_gradient_descent(X, y, alpha, beta, learning_rate, epochs):
    N = X.shape[0]  # Number of samples
    for epoch in range(epochs):
        # Shuffle the dataset
        indices = np.arange(N)
        np.random.shuffle(indices)  # Shuffle the indices of the dataset
        for n in indices:
            # Forward pass to calculate prediction
            y_n = forward_pass(X[n], alpha, beta)
            # Calculate gradients for the sample
            grad_alpha_n, grad_beta_n = backward_pass(X[n], y[n], y_n, alpha, beta)
            # Update parameters
            alpha -= learning_rate * grad_alpha_n
            beta -= learning_rate * grad_beta_n
    return alpha, beta  # Return updated parameters

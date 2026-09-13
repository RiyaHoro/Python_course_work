import numpy as np


def generate_logistic_data(theta, n, m):

    #Generate m continuous independent variables
    X_variables = np.random.randn(n, m)

    #Add a column of 1s for the intercept
    ones = np.ones((n, 1))
    X = np.hstack((ones, X_variables))

    #  Generate random coefficients
    beta = np.random.randn(m + 1)

    # Calculate X · beta
    linear_output = X @ beta

    # Apply sigmoid function
    probability = 1 / (1 + np.exp(-linear_output))

    # Generate binary labels
    Y = (probability > 0.5).astype(int)

    # Generate Bernoulli noise
    flip = np.random.binomial(1, theta, n)

    # Flip labels where flip = 1
    Y = np.where(flip == 1, 1 - Y, Y)

    # Make Y an n × 1 array
    Y = Y.reshape(-1, 1)

    return X, Y, beta


if __name__ == "__main__":

    theta = 0.1
    n = 100
    m = 2

    X, Y, beta = generate_logistic_data(theta, n, m)

    print("X:")
    print(X)

    print("\nY:")
    print(Y)

    print("\nBeta:")
    print(beta)
import numpy as np


def generate_logistic_data(theta, n, m):

   
    X_variables = np.random.randn(n, m)
    ones = np.ones((n, 1))
    X = np.hstack((ones, X_variables))

    beta = np.random.randn(m + 1)


    linear_output = X @ beta

    probability = 1 / (1 + np.exp(-linear_output))
    Y = (probability > 0.5).astype(int)


    flip = np.random.binomial(1, theta, n)

    Y = np.where(flip == 1, 1 - Y, Y)

    
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
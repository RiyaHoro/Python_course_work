import numpy as np


def logistic_regression_gd(X, Y, k, tau, learning_rate):

    Y = np.asarray(Y).reshape(-1)

    X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

    n = X_with_intercept.shape[0]
    beta = np.random.randn(X_with_intercept.shape[1])

    previous_cost = float("inf")

    for _ in range(k):
       
        z = X_with_intercept @ beta
        probabilities = 1 / (1 + np.exp(-z))

        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

        cost = -np.mean(
            Y * np.log(probabilities)
            + (1 - Y) * np.log(1 - probabilities)
        )

        gradient = (X_with_intercept.T @ (probabilities - Y)) / n

        beta = beta - learning_rate * gradient

       
        if abs(previous_cost - cost) < tau:
            break

        previous_cost = cost

    return beta, cost


if __name__ == "__main__":
    # Example usage
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [5.0, 5.0]
    ])

    Y = np.array([
        [0],
        [0],
        [1],
        [1],
        [1]
    ])

    k = 1000
    tau = 1e-6
    learning_rate = 0.1

    beta, final_cost = logistic_regression_gd(
        X, Y, k, tau, learning_rate
    )

    print("Learned coefficients:", beta)
    print("Final cost:", final_cost)

import numpy as np
import matplotlib.pyplot as plt


def logistic_regression_gd(X, Y, k, tau, learning_rate):

    Y = np.asarray(Y).reshape(-1)

    # Add intercept column
    X = np.column_stack((np.ones(X.shape[0]), X))

    n = X.shape[0]

    # Random initialization of beta
    beta = np.random.randn(X.shape[1])

    previous_cost = float("inf")

    for _ in range(k):

        # Prediction
        z = X @ beta
        probability = 1 / (1 + np.exp(-z))

        # Avoid log(0)
        probability = np.clip(probability, 1e-15, 1 - 1e-15)

        # Cost function
        cost = -np.mean(
            Y * np.log(probability)
            + (1 - Y) * np.log(1 - probability)
        )

        # Gradient
        gradient = (X.T @ (probability - Y)) / n

        # Update beta
        beta = beta - learning_rate * gradient

        # Stopping condition
        if abs(previous_cost - cost) < tau:
            break

        previous_cost = cost

    return beta, cost


def generate_logistic_data(n, theta):

    m = len(theta) - 1

    # Generate independent variables
    X = np.random.randn(n, m)

    # Add intercept
    X_with_intercept = np.column_stack((np.ones(n), X))

    # Generate probabilities
    z = X_with_intercept @ theta
    probability = 1 / (1 + np.exp(-z))

    # Generate binary Y
    Y = np.random.binomial(1, probability, size=n)

    return X, Y


def calculate_coefficient_error(true_beta, learned_beta):

    return np.mean((true_beta - learned_beta) ** 2)


def run_experiments():

    # Different sample sizes
    n_values = [50, 100, 200, 500, 1000]

    # Different theta values
    theta_values = [
        np.array([0.5, 1.0, -1.0]),
        np.array([1.0, 2.0, -2.0]),
        np.array([2.0, 3.0, -3.0])
    ]

    results_n = {}
    results_theta = {}

    # Experiment 1: Effect of n
    theta = theta_values[0]

    for n in n_values:

        X, Y = generate_logistic_data(n, theta)

        learned_beta, final_cost = logistic_regression_gd(
            X,
            Y,
            k=5000,
            tau=1e-7,
            learning_rate=0.05
        )

        error = calculate_coefficient_error(theta, learned_beta)

        results_n[n] = error

    # Experiment 2: Effect of theta
    n = 500

    for theta in theta_values:

        X, Y = generate_logistic_data(n, theta)

        learned_beta, final_cost = logistic_regression_gd(
            X,
            Y,
            k=5000,
            tau=1e-7,
            learning_rate=0.05
        )

        error = calculate_coefficient_error(theta, learned_beta)

        theta_name = str(theta)

        results_theta[theta_name] = error

    return results_n, results_theta


def plot_n_vs_error(results_n):

    n_values = list(results_n.keys())
    errors = list(results_n.values())

    plt.figure(figsize=(8, 5))

    plt.plot(n_values, errors, marker="o")

    plt.xlabel("Sample Size (n)")
    plt.ylabel("Coefficient MSE")

    plt.title("Effect of Sample Size on Coefficient Learning")

    plt.grid(True)

    plt.show()


def plot_theta_vs_error(results_theta):

    theta_names = list(results_theta.keys())
    errors = list(results_theta.values())

    plt.figure(figsize=(9, 5))

    plt.bar(theta_names, errors)

    plt.xlabel("Theta Values")
    plt.ylabel("Coefficient MSE")

    plt.title("Effect of Theta on Coefficient Learning")

    plt.xticks(rotation=20)

    plt.grid(axis="y")

    plt.show()


if __name__ == "__main__":

    results_n, results_theta = run_experiments()

    print("Effect of n:")
    for n, error in results_n.items():
        print("n =", n, "Coefficient MSE =", error)

    print("\nEffect of theta:")
    for theta, error in results_theta.items():
        print("theta =", theta, "Coefficient MSE =", error)

    # Generate graphs
    plot_n_vs_error(results_n)

    plot_theta_vs_error(results_theta)
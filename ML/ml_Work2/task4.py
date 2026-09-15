import numpy as np
import matplotlib.pyplot as plt


def generate_logistic_data(theta, n, m):

    
    X_variables = np.random.randn(n, m)

    ones = np.ones((n, 1))
    X = np.hstack((ones, X_variables))
    beta = np.random.randn(m + 1)

    linear_output = X @ beta

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


def sigmoid(z):

    # Prevent overflow in exp()
    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


def logistic_regression(X, Y, learning_rate, iterations,
                        regularization_type=None, lambda_value=0):

    n = X.shape[0]
    m = X.shape[1]

    # Start all beta values at zero
    beta = np.zeros((m, 1))

    for _ in range(iterations):

        # Calculate predictions
        z = X @ beta
        predictions = sigmoid(z)

        # Calculate gradient of normal logistic regression cost
        gradient = (1 / n) * (X.T @ (predictions - Y))

        # Add regularization gradient
        #
        # Do NOT regularize beta[0], because beta[0]
        # is the intercept.
        if regularization_type == "L2":

            regularization_gradient = (lambda_value / n) * beta

            regularization_gradient[0] = 0

            gradient = gradient + regularization_gradient

        elif regularization_type == "L1":

            regularization_gradient = (lambda_value / n) * np.sign(beta)

            regularization_gradient[0] = 0

            gradient = gradient + regularization_gradient

        # Update beta
        beta = beta - learning_rate * gradient

    return beta


def calculate_cost(X, Y, beta, regularization_type=None,
                   lambda_value=0):

    n = X.shape[0]

    # Calculate predicted probabilities
    predictions = sigmoid(X @ beta)

    # Avoid log(0)
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)

    # Normal logistic regression cost
    cost = -(1 / n) * np.sum(
        Y * np.log(predictions) +
        (1 - Y) * np.log(1 - predictions)
    )

    # Add L2 penalty
    if regularization_type == "L2":

        penalty = (lambda_value / (2 * n)) * np.sum(beta[1:] ** 2)

        cost = cost + penalty

    # Add L1 penalty
    elif regularization_type == "L1":

        penalty = (lambda_value / n) * np.sum(np.abs(beta[1:]))

        cost = cost + penalty

    return cost



if __name__ == "__main__":

    theta = 0.1
    n = 100
    m = 2

    X, Y, true_beta = generate_logistic_data(theta, n, m)

    print("True Beta used to generate the data:")
    print(true_beta)

    learning_rate = 0.1
    iterations = 5000

    lambda_values = [0, 0.01, 0.1, 1, 10, 100]


    l1_betas = []
    l2_betas = []

    
    for lambda_value in lambda_values:

        # L1 model
        beta_l1 = logistic_regression(
            X,
            Y,
            learning_rate,
            iterations,
            regularization_type="L1",
            lambda_value=lambda_value
        )

        # L2 model
        beta_l2 = logistic_regression(
            X,
            Y,
            learning_rate,
            iterations,
            regularization_type="L2",
            lambda_value=lambda_value
        )

        l1_betas.append(beta_l1.flatten())
        l2_betas.append(beta_l2.flatten())

    # Convert lists to NumPy arrays
    l1_betas = np.array(l1_betas)
    l2_betas = np.array(l2_betas)

   

    print("\n" + "=" * 60)
    print("L1 REGULARIZATION")
    print("=" * 60)

    for i, lambda_value in enumerate(lambda_values):

        print(
            f"Lambda = {lambda_value:<6} "
            f"Beta = {l1_betas[i]}"
        )

    print("\n" + "=" * 60)
    print("L2 REGULARIZATION")
    print("=" * 60)

    for i, lambda_value in enumerate(lambda_values):

        print(
            f"Lambda = {lambda_value:<6} "
            f"Beta = {l2_betas[i]}"
        )

   

    print("\n" + "=" * 60)
    print("COST VALUES")
    print("=" * 60)

    for i, lambda_value in enumerate(lambda_values):

        beta_l1 = l1_betas[i].reshape(-1, 1)
        beta_l2 = l2_betas[i].reshape(-1, 1)

        cost_l1 = calculate_cost(
            X,
            Y,
            beta_l1,
            regularization_type="L1",
            lambda_value=lambda_value
        )

        cost_l2 = calculate_cost(
            X,
            Y,
            beta_l2,
            regularization_type="L2",
            lambda_value=lambda_value
        )

        print(
            f"Lambda = {lambda_value:<6} "
            f"L1 Cost = {cost_l1:.4f}   "
            f"L2 Cost = {cost_l2:.4f}"
        )

    

    plt.figure(figsize=(9, 6))

    plt.plot(
        lambda_values,
        l1_betas[:, 0],
        marker="o",
        label="Beta 0 (Intercept)"
    )

    plt.plot(
        lambda_values,
        l1_betas[:, 1],
        marker="o",
        label="Beta 1"
    )

    plt.plot(
        lambda_values,
        l1_betas[:, 2],
        marker="o",
        label="Beta 2"
    )

    plt.xscale("symlog", linthresh=0.01)

    plt.xlabel("Regularization Constant (Lambda)")
    plt.ylabel("Learned Beta Values")
    plt.title("Effect of L1 Regularization on Beta Vector")

    plt.legend()
    plt.grid(True)

    plt.show()

    

    plt.figure(figsize=(9, 6))

    plt.plot(
        lambda_values,
        l2_betas[:, 0],
        marker="o",
        label="Beta 0 (Intercept)"
    )

    plt.plot(
        lambda_values,
        l2_betas[:, 1],
        marker="o",
        label="Beta 1"
    )

    plt.plot(
        lambda_values,
        l2_betas[:, 2],
        marker="o",
        label="Beta 2"
    )

    plt.xscale("symlog", linthresh=0.01)

    plt.xlabel("Regularization Constant (Lambda)")
    plt.ylabel("Learned Beta Values")
    plt.title("Effect of L2 Regularization on Beta Vector")

    plt.legend()
    plt.grid(True)

    plt.show()

    

    l1_magnitude = np.linalg.norm(l1_betas[:, 1:], axis=1)
    l2_magnitude = np.linalg.norm(l2_betas[:, 1:], axis=1)

    plt.figure(figsize=(9, 6))

    plt.plot(
        lambda_values,
        l1_magnitude,
        marker="o",
        label="L1"
    )

    plt.plot(
        lambda_values,
        l2_magnitude,
        marker="o",
        label="L2"
    )

    plt.xscale("symlog", linthresh=0.01)

    plt.xlabel("Regularization Constant (Lambda)")
    plt.ylabel("Magnitude of Beta Vector")
    plt.title("Effect of Regularization on Beta Magnitude")

    plt.legend()
    plt.grid(True)

    plt.show()
import numpy as np


class Regression:

    def __init__(self, k, tau, learning_rate):
        self.k = k
        self.tau = tau
        self.learning_rate = learning_rate
        self.beta = None

    def fit(self, X, y):

        n = X.shape[0]

        self.beta = np.random.randn(X.shape[1])

        previous_cost = float("inf")

        for i in range(self.k):

            y_pred = self.predict(X)

            gradient = self.gradient(X, y, y_pred)

            self.beta = self.beta - self.learning_rate * gradient

            cost = self.cost(y, y_pred)

            if abs(previous_cost - cost) < self.tau:
                break

            previous_cost = cost

        return self.beta, cost


class LinearRegression(Regression):

    def predict(self, X):
        return X @ self.beta

    def gradient(self, X, y, y_pred):

        n = X.shape[0]

        error = y_pred - y

        return (X.T @ error) / n

    def cost(self, y, y_pred):

        error = y_pred - y

        return np.mean(error ** 2) / 2


class LogisticRegression(Regression):

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):

        return self.sigmoid(X @ self.beta)

    def gradient(self, X, y, y_pred):

        n = X.shape[0]

        return (X.T @ (y_pred - y)) / n

    def cost(self, y, y_pred):

        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

        return -np.mean(
            y * np.log(y_pred) +
            (1 - y) * np.log(1 - y_pred)
        )


# ============================================================
# LINEAR DATA
# ============================================================

def generate_linear_data(sigma, n, m):

    X = np.random.rand(n, m)

    X = np.column_stack((np.ones(n), X))

    beta = np.random.randn(m + 1)

    e = np.random.normal(0, sigma, n)

    y = X @ beta + e

    return X, y, beta


def generate_logistic_data(theta, n, m):

    X = np.random.randn(n, m)

    X = np.column_stack((np.ones(n), X))

    beta = np.random.randn(m + 1)

    probability = 1 / (1 + np.exp(-(X @ beta)))

    Y = (probability > 0.5).astype(int)

    flip = np.random.binomial(1, theta, n)

    Y = np.where(flip == 1, 1 - Y, Y)

    return X, Y

if __name__ == "__main__":

    # Linear Regression

    X, y, true_beta = generate_linear_data(
        sigma=1,
        n=10,
        m=3
    )

    linear_model = LinearRegression(
        k=10000,
        tau=0.000001,
        learning_rate=0.01
    )

    learned_beta, cost = linear_model.fit(X, y)

    print("LINEAR REGRESSION")
    print("True beta:", true_beta)
    print("Learned beta:", learned_beta)
    print("Final cost:", cost)


    # Logistic Regression

    X, Y = generate_logistic_data(
        theta=0.1,
        n=100,
        m=2
    )

    logistic_model = LogisticRegression(
        k=5000,
        tau=0.000001,
        learning_rate=0.1
    )

    learned_beta, cost = logistic_model.fit(X, Y)

    print("\nLOGISTIC REGRESSION")
    print("Learned beta:", learned_beta)
    print("Final cost:", cost)
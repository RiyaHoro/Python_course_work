import numpy as np

def generate_data(sigma, n, m):
    X = np.random.rand(n, m)
    X = np.column_stack((np.ones(n), X))

    beta = np.random.randn(m + 1)

    e = np.random.normal(0, sigma, n)

    y = X @ beta + e

    return X, y, beta


X, y, beta = generate_data(sigma=1, n=10, m=3)

print("X:")
print(X)
print("\n")
print("y:")
print(y)
print("\n")
print("beta:")
print(beta)
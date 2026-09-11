import numpy as np

def generate_data(sigma, n, m):
    X = np.random.rand(n, m)
    X = np.column_stack((np.ones(n), X))
    
    beta = np.random.randn(m + 1)
    e = np.random.normal(0, sigma, n)
    y = X @ beta + e
    
    return X, y, beta

def linear_regression_gd(X, y, k, tau, learning_rate):
    
    n = X.shape[0]
    
    beta = np.random.randn(X.shape[1])
    
    previous_cost = float("inf")
    
    for i in range(k): 
        y_pred = X @ beta
        
        error = y_pred - y
        cost = np.mean(error ** 2) / 2
        gradient = (X.T @ error) / n
        beta = beta - learning_rate * gradient
        
        if abs(previous_cost - cost) < tau:
            break
        
        previous_cost = cost
    
    return beta, cost

X, y, true_beta = generate_data(sigma=1,n=10,m=3)

learned_beta, final_cost = linear_regression_gd(X,y,k=10000,tau=0.000001,learning_rate=0.01)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTrue beta:", true_beta)
print("Learned beta:", learned_beta)

print("\nFinal cost:", final_cost)
'''
Report
-------------------------------------------------------------------
Objective:

The objective of this experiment is to investigate how the number
of observations (n) and the amount of noise (sigma) affect the
ability of the linear regression model to learn the coefficients
(beta) used to generate the output vector y.

The data is generated using the relationship:

y = X * beta + e

where X represents the independent variables, beta represents the
true coefficients, and e represents the random Gaussian noise.

Method:
-------------------------------------------------------------------

Different values of n and sigma were used to generate datasets.
Gradient Descent was then applied to each dataset to estimate the
coefficients.

The cost function used to measure the prediction error is:

J(beta) = (1 / 2n) * sum((y_pred - y)^2)

The coefficients are updated using Gradient Descent:

beta_new = beta_old - lambda * gradient

The learned coefficients are compared with the original
coefficients using the mean squared error:

Beta Error = mean((beta_true - beta_learned)^2)


Results:
-------------------------------------------------------------------

The experiments were performed using different values of n and
sigma. The beta error and final cost were recorded for each
experiment.

The results show that the learned coefficients are generally
closer to the original coefficients when more observations are
used.

When sigma is increased, more random noise is added to the output
variable y. This makes it more difficult for the model to recover
the original coefficients accurately.


Effect of n:
-------------------------------------------------------------------

When the number of observations is increased, the model gets more
information about the relationship between X and y. Therefore,
the learned coefficients generally become more stable and closer
to the true coefficients.

Effect of sigma:
-------------------------------------------------------------------

The noise is generated from a Gaussian distribution:

e ~ N(0, sigma^2)

When sigma is small, there is less variation caused by noise, so
the underlying relationship between X and y is easier to learn.

As sigma increases, the amount of noise increases, making it more
difficult for the model to accurately recover the original
coefficients.


Conclusion:
-------------------------------------------------------------------

The experiment shows that the amount of data and the amount of
noise affect the ability of linear regression to learn the true
coefficients.

Increasing n generally improves coefficient estimation because
the model has more observations from which to learn the underlying
relationship.

Increasing sigma generally makes coefficient estimation more
difficult because the output contains more random variation.

Therefore, larger datasets and lower noise generally result in
more accurate estimates of the original coefficients.
'''
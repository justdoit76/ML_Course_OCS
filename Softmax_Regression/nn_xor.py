import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

from nn_xor_func import gradient_descent, mlp_training, mlp_forward
m, n = X.shape

# Single Layer Perceptron
W = np.random.randn(n) * 0.01
b = 0
lr = 0.05
epochs = 5000
Lambda = 0
W_final, b_final = gradient_descent(X, y, W, b, lr, epochs, Lambda)


# Multiple Layer Perceptiron(add hidden layer)
"""
Z1 = XW1 + b1
A1 = sigmoid(Z1)
Z2 = A1*W2 + b2
A2 = sigmoid(Z2)
"""
hidden_dim = 2
W1 = np.random.randn(n, hidden_dim)
b1 = np.zeros((1, hidden_dim)) 

W2 = np.random.randn(hidden_dim, 1)
b2 = np.zeros((1, 1))
lr = 0.05
epochs = 20000

y = y.reshape(-1, 1)

W1_f, b1_f, W2_f, b2_f = mlp_training(X, y, W1, b1, W2, b2, lr, epochs)

# predict
X_test = np.array([    
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0]
])
y_pred, _ = mlp_forward(X_test, W1_f, b1_f, W2_f, b2_f)
print(y_pred.round())

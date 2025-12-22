import numpy as np
import matplotlib.pyplot as plt

# X, 공부시간
X = np.array([1, 5, 10]).reshape(-1, 1)
# y, 실제성적
y = np.array([10, 50, 100])
 
m, n = X.shape
print(m, n)
 
plt.xlabel('Learning time')
plt.ylabel('Score')
plt.plot(X, y)
plt.scatter(X, y, color='red')
plt.show()


# init parameters
W = np.zeros(n)
b = 0
lr = 0.001
epochs = 1000
Lambda = 0
 
# machine learning
from Linear_Func import gradient_descent 
W_final, b_final = gradient_descent(X, y, W, b, lr, epochs, Lambda)
print(f'Final w: {np.round(W_final, 3)}, Final b: {b_final:.3f}')
 
# prediction
def predict(X, W, b):
    return np.dot(X, W) + b
 
X_test = np.array([2, 4, 6, 8]).reshape(-1, 1)
y_pred = predict(X_test, W_final, b_final)

for i in range(len(X_test)):
    print(f'X_test={X_test[i]}, y={y_pred[i]:.3f}')
# Non vectorization with list
X = [[1,2], [3,4], [5,6]]
W = [2,1]
b = 1

# y_hat = WX+b
y = []
for r in range(len(X)):
    wx = 0
    for c in range(len(X[0])):
        wx += X[r][c]*W[c]
    y.append(wx+b)

# [3x2] X [1*2] => 1D array (no good)
print(y)


# Non vectorization with numpy
import numpy as np

X = np.array([[1,2],
              [3,4],
              [5,6]]) 
w = np.array([2,1])

y = X.dot(w)+1 
#y = X @ w + 1
print(y)


# vectorization with numpy
import numpy as np

X = np.array([[1,2],
              [3,4],
              [5,6]]) 
w = np.array([2,1]) # 1x2
# -1은 원소갯수에 맞춰 자동으로 행 계산,생성
w_col = w.reshape(-1, 1)

y = X.dot(w_col)+1
print(y)
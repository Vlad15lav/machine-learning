import numpy as np
import matplotlib.pyplot as plt

x = np.load('dataset.npy')
t = np.load('target.npy')
z = np.load('realmodel.npy')

N = x.shape[0]

# Build Design matrix
def DesignMatrix(M):
    Plan_Matrix = np.zeros((N, M))
    for i in range(M):
        Plan_Matrix[:, i] = x ** i
    return Plan_Matrix
# Train w
def GetW(Plan_Matrix):
    Mura_Penrose = np.linalg.inv(Plan_Matrix.T @ Plan_Matrix) @ Plan_Matrix.T
    w = Mura_Penrose @ t
    return w

# Regression model
print('Write degree of the polynomial: ', end='')
Step = int(input())

Plan_Matrix = GetMatrix_Plan(Step)
w = GetW(Plan_Matrix)
y = Plan_Matrix @ w
E = np.sum((t - y) ** 2) / 2
print('Error = ' + str(E))
plt.figure('M = ' + str(Step) + ', Error = ' + str(E))
plt.plot(x, z, 'r-', label='Real model')
plt.plot(x, t, 'b,', label='DataSet')
plt.plot(x, y, 'b-,', label='Train model')
plt.legend()
plt.show()

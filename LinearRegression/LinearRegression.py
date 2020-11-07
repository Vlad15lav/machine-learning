import numpy as np
import matplotlib.pyplot as plt

N = 1000
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

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
y = Plan_Matrix @ GetW(Plan_Matrix)
E = np.sum((t - y) ** 2) / 2
plt.figure('M = ' + str(Step) + ', Error = ' + str(E))
plt.plot(x, z, 'r-', label='Real model')
plt.plot(x, t, 'b,', label='DataSet')
plt.plot(x, y, 'b-,', label='Train model')
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.__weight = np.zeros(0)
        self.__complexity = 0
    
    def get_parametrs(self):
        return {'Weight': self.__weight, 'Complexity': self.__complexity}
    
    def __DesignMatrix(self, x_set, M):
        Plan_Matrix = np.zeros((x_set.shape[0], M))
        for i in range(M):
            Plan_Matrix[:, i] = x_set ** i
        return Plan_Matrix

    def fit(self, x_set, t_set, complexity):
        Plan_Matrix = self.__DesignMatrix(x_set, complexity)
        self.__complexity = complexity
        self.__weight = np.linalg.inv(Plan_Matrix.T @ Plan_Matrix) @ Plan_Matrix.T @ t_set
    
    def predict(self, x_set):
        Plan_Matrix = self.__DesignMatrix(x_set, self.__complexity)
        return Plan_Matrix @ self.__weight
    
    def score(self, x_set, t_set):
        y = self.__DesignMatrix(x_set, self.__complexity) @ self.__weight
        return np.sum((t_set - y) ** 2) / 2


x = np.load('dataset.npy')
t = np.load('target.npy')
z = np.load('realmodel.npy')

N = x.shape[0]

# Regression model
print('Write degree of the polynomial: ', end='')
Step = int(input())

model = LinearRegression()
model.fit(x, t, Step)
y = model.predict(x)
E = model.score(x, t)

print('Error = ' + str(E))
plt.figure('M = ' + str(Step) + ', Error = ' + str(E))
plt.plot(x, z, 'r-', label='Real model')
plt.plot(x, t, 'b,', label='DataSet')
plt.plot(x, y, 'b-,', label='Train model')
plt.legend()
plt.show()

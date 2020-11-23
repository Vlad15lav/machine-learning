import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.__weight = np.zeros(0)
    
    def get_parametrs(self):
        return {'Weight': self.__weight}
    
    def fit(self, x_set, t_set, bias=True, isQR=False, isSVD=False):
        DesignMatrix = np.ones((x_set.shape[0], x_set.shape[1] + bias))
        DesignMatrix[:, bias:] = x_set
        
        if isQR:
            Q, R = np.linalg.qr(DesignMatrix)
            self.__weight = np.linalg.inv(R) @ Q.T @ t_set
        elif isSVD:
            U, S, Vt = np.linalg.svd(DesignMatrix, full_matrices=False)
            self.__weight = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ t_set
        else:
            self.__weight = np.linalg.inv(DesignMatrix.T @ DesignMatrix) @ DesignMatrix.T @ t_set
    
    def fit_grad(self, x_set, t_set, learning_rate=0.01, iters=100, std=0.01):
        self.__weight = np.random.randn(x_set.shape[1], t_set.shape[1]) * std
        W = np.copy(self.__weight)
        best_mse = pow(10, 25)
        for iter in range(iters):
            pred = self.predict(x_set)
            gradW = -(2 / x_set.shape[0]) * (x_set.T @ (t_set - pred))
            W -= learning_rate * gradW

            cur_mse = np.mean((pred - t_set) ** 2)
            if cur_mse < best_mse:
                best_mse = cur_mse
                self.__weight = W
            print('Iter - {} | MSE - {}'.format(iter + 1, cur_mse))
    
    def predict(self, x_set):
        return x_set @ self.__weight
    
    def score(self, x_set, t_set):
        pred_set = self.predict(x_set)
        return np.mean((pred_set - t_set) ** 2)

def class2int(data, feature, code, begin=0):
    t = np.zeros((data.shape[0], code.shape[0]), dtype=data.dtype)
    t[:] = data[:, feature].reshape(data.shape[0], -1)
    t = (t == code)
    data[:, feature] = t.argmax(axis=1) + begin

#f = open('flare.names', 'r')
f = open('flare.data', 'r')
lines = f.readlines()
f.close()

dataset = []
for line in lines:
    row = line.split(' ')
    row[-1] = row[-1].replace("\n", "")
    dataset.append(row)
dataset.pop(0)
dataset = np.array(dataset)

Class = np.array(['A','B','C','D','E','F','H'])
LargestSpotSize = np.array(['X','R','S','A','H','K'])
Distribution = np.array(['X','O','I','C'])

class2int(dataset, 0, Class)
class2int(dataset, 1, LargestSpotSize)
class2int(dataset, 2, Distribution)
dataset = np.int_(dataset)

X = dataset[:, :-3]
Y = dataset[:, -3:]

N = X.shape[0]
index_data = np.arange(N)
np.random.shuffle(index_data)
index_train = index_data[:int(N * 0.8)]
index_test = index_data[int(N * 0.8):]
x_train = X[index_train]
t_train = Y[index_train]
x_test = X[index_test]
t_test = Y[index_test]

model = LinearRegression()
model.fit(x_set=x_train, t_set=t_train, bias=False, isQR=False, isSVD=False)
train_MSE = model.score(x_train, t_train)
test_MSE = model.score(x_test, t_test)
print('(Train) Mean squared error - {}\n(Test) Mean squared error - {}'.format(train_MSE, test_MSE))

model.fit_grad(x_train, t_train)
train_MSE = model.score(x_train, t_train)
test_MSE = model.score(x_test, t_test)
print('(Train) Mean squared error - {}\n(Test) Mean squared error - {}'.format(train_MSE, test_MSE))
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import sklearn as sk
from sklearn.datasets import load_digits

class KNN:
    def __init__(self, k):
        self.__k = k
    
    def params(self):
    	return {'K': self.__k}

    def fit(self, x_train, t_train, x_val=None, t_val=None):
        self.X = x_train
        self.Y = t_train
        if not x_val is None and not t_val is None:
            best_k, k_max, best_acc = 0, self.__k, 0
            for k_i in range(1, k_max + 1):
                self.__k = k_i
                acc = self.score(x_val, t_val)
                print('K - {} | Val accuracy - {}'.format(k_i, acc))
                if acc > best_acc:
                    best_acc = acc
                    best_k = k_i
            self.__k = best_k
            print('Best k - {} | Best val accuracy - {}'.format(best_k, best_acc))
    
    def predict(self, x_set):
        num_train = self.X.shape[0]
        num_test = x_set.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        # L1 loss
        dists = np.sum(np.abs(np.float32(x_set[:, np.newaxis] - self.X)), axis=2)

        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            pred[i] = mode(self.Y[dists[i].argsort()][:self.k]).mode
        return pred
    
    def score(self, x_set, t_set):
        pred_set = self.predict(x_set)
        return np.sum(pred_set == t_set) / pred_set.shape[0]


# Data Set Digits
digits = load_digits()
N = len(digits.data)
data = np.copy(digits.data)
target = digits.target

# Split data
index_data = np.arange(N)
np.random.shuffle(index_data)
index_train = index_data[:int(N * 0.8)]
index_val = index_data[int(N * 0.8):]
x_train = data[index_train]
t_train = target[index_train]
x_val = data[index_val]
t_val = target[index_val]

# Fit model
model = KNN(10)
model.fit(x_train, t_train, x_val, t_val)
print('Val accuracy - {}'.format(model.score(x_val, t_val)))
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_boston

class GradientBoost:
    def __init__(self, n_trees=10, lr=0.001,
                 max_depth=3, min_data=5,
                 loss_function='mse', randmon_state=5):
        self.__n_trees = n_trees
        self.__lr = lr
        self.__max_depth = max_depth
        self.__min_data = min_data
        self.__random_state = randmon_state
        self.__loss_function = loss_function
        if loss_function == 'mse':
            self.__loss = self.mse_loss
            self.__grad = self.__mse_grad
        elif loss_function == 'log_loss':
            self.__loss = self.log_loss
            self.__grad = self.__log_grad
        else:
            raise ValueError('Incorrect loss function!')
        self.__trees = []

    def mse_loss(self, pred, target):
        return np.mean((pred - target) ** 2)
    
    def __mse_grad(self, pred, target):
        return (pred - target.reshape([target.shape[0], 1])) * 2 / pred.shape[0]
    
    def log_loss(self, pred, target):
        return np.nansum(-np.log(pred) * target - np.log(1 - pred) * (1 - target))
    
    def __log_grad(self, pred, target):
        return (pred - target) / (pred * (1 - pred))
    
    def __softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def fit(self, x_train, t_train):
        self.init_labels = t_train
        # First initial predict
        vec_b = np.mean(t_train) * np.ones([t_train.shape[0], 1])
        pred = vec_b.copy() # prediction

        self.__trees = []

        for t in range(self.__n_trees):
            pred_agrad = -self.__grad(pred, t_train)

            if self.__loss_function == 'mse':
                tree = DecisionTreeRegressor(max_depth=self.__max_depth,
                                             random_state=self.__random_state,
                                             min_samples_leaf=self.__min_data)
            else:
                tree = DecisionTreeClassifier(max_depth=self.__max_depth,
                                             random_state=self.__random_state,
                                             min_samples_leaf=self.__min_data)
            tree.fit(x_train, pred_agrad)
            vec_b = tree.predict(x_train).reshape([x_train.shape[0], 1])

            self.__trees.append(tree)
            pred += self.__lr * vec_b
        
        self.train_pred = pred
        if self.__loss_function != 'mse':
            self.train_pred = self.__softmax(pred)

    def predict(self, x_set):
        pred = np.ones((x_set.shape[0], 1)) * np.mean(self.init_labels)
        for tree in self.__trees:
            pred += self.__lr * tree.predict(x_set).reshape(x_set.shape[0], 1)
        
        if self.__loss_function == 'log_loss':
            pred = self.__softmax(pred)
            max_acc, best_t = 0, 0
            for t_val in np.linspace(0.01, 1, 100):
                acc = np.mean(self.init_labels == (self.train_pred > t_val))
                if acc > max_acc:
                    max_acc, best_t = acc, t_val
            return pred > best_t
        else:
            return pred
    
    def score(self, x_set, t_set):
        pred = self.predict(x_set)
        if self.__loss_function == 'mse':
            return self.mse_loss(pred, t_set)
        else:
            return np.sum(pred == t_set) / t_set.shape[0]


# Load dataset
dataset = load_boston()

# Split data
N = dataset.data.shape[0]
data = dataset.data
targets = dataset.target
index_data = np.arange(N)
np.random.shuffle(index_data)
index_train = index_data[:int(N * 0.8)]
index_test = index_data[int(N * 0.8):]
x_train = data[index_train]
t_train = targets[index_train]
x_test = data[index_test]
t_test = targets[index_test]

# Train model
model = GradientBoosting(lr=0.01)
model.fit(x_train, t_train)
print('Test score - {}'.format(model.score(x_test, t_test)))
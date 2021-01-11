import numpy as np
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm

def read_data(path_data):
    with open(path_data, 'r') as file:
        txt = [x.split(',') for x in file.read().splitlines()]
    data = np.array(txt, dtype=np.float32)
    return data[:, :-1], data[:, -1]

class SVM:
    def __init__(self, epochs=200, lr=0.001, reg=0, sigma_init=0.05, splitVal=0):
        self.__epochs = epochs
        self.__lr = lr
        self.__reg = reg
        self.__sigma_init = sigma_init
        self.__splitVal = splitVal
        self.__W = np.zeros(0)
    
    def hinge_loss(self, x_set, t_set):
        if x_set.shape[1] != self.__W.shape[0]:
            x_set = np.append(x_set, np.ones((x_set.shape[0], 1)), axis=1)
        penalty = 1 - t_set * (x_set @ self.__W)
        penalty = np.append(penalty[:, None], np.ones((penalty.shape[0], 1)), axis=1)
        return np.mean(np.max(penalty, axis=1))

    def soft_margin_loss(self, x_set, t_set):
        return self.hinge_loss(x_set, t_set) + 0.5 * self.__reg * (self.__W @ self.__W)

    def fit(self, x_train, t_train):
        x_train = np.append(x_train, np.ones((x_train.shape[0], 1)), axis=1)
        self.__W = np.random.randn(x_train.shape[1]) * self.__sigma_init

        splitVal = int(x_train.shape[0] * self.__splitVal)
        x_val, t_val = x_train[:splitVal], t_train[:splitVal]
        x_train, t_train = x_train[splitVal:], t_train[splitVal:]
        num_train, features = x_train.shape

        train_loss, val_loss, weights_history = [], [], []
        for epoch in tqdm(range(self.__epochs)):
            # Train model 
            for i in range(num_train):
                margin = t_train[i] * (x_train[i] @ self.__W)
                gradW = (self.__reg * self.__W / self.__epochs - \
                                         (margin < 1) * t_train[i] * x_train[i])
                self.__W -= self.__lr * gradW
            # Train loss
            train_loss.append(self.hinge_loss(x_train, t_train))
            # Val loss
            if splitVal > 0:
                val_loss.append(self.hinge_loss(x_val, t_val))
            # Save weight
            weights_history.append(np.copy(self.__W))
        
        best_epoch = np.argmin(val_loss) if splitVal > 0 else np.argmin(train_loss)
        self.__W = weights_history[best_epoch]
        
        plt.figure('Training')
        plt.plot(train_loss, label='Train Set')
        if splitVal > 0:
            plt.plot(val_loss, label='Valid Set')
            plt.plot(best_epoch, val_loss[best_epoch], '*', label='Best epoch')
        else:
            plt.plot(best_epoch, train_loss[best_epoch], '*', label='Best epoch')
        plt.legend()
        plt.show()

    def predict(self, x_set):
        x_set = np.append(x_set, np.ones((x_set.shape[0], 1)), axis=1)
        return np.sign(x_set @ self.__W)
    
    def score(self, x_set, t_set):
        pred_set = self.predict(x_set)
        return np.sum(pred_set == t_set) / t_set.shape[0]
    
    def save_model(self, path='SVM_weights'):
        f = open(r'{}.pickle'.format(path), 'wb')
        pickle.dump(self.__W, f)
        print('The weights is saved to file - {}.pickle!'.format(path))
        f.close()

    def load_model(self, path='SVM_weights.pickle'):
        f = open(r'{}'.format(path), 'rb')
        self.__W = pickle.load(f)
        f.close()

# Load dataset
data, targets = read_data('data/spambase.data')

# Split data
N = data.shape[0]
index_data = np.arange(N)
np.random.shuffle(index_data)
index_train = index_data[:int(N * 0.8)]
index_test = index_data[int(N * 0.8):]
targets[targets == 0] = -1
x_train = data[index_train]
t_train = targets[index_train]
x_test = data[index_test]
t_test = targets[index_test]

model = SVM(epochs=120, lr=0.001, reg=0.005, splitVal=0.2)
model.fit(x_train, t_train)
test_acc = model.score(x_test, t_test)
print('Test Accuracy - {}'.format(test_acc))
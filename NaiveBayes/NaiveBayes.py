import numpy as np
import matplotlib.pyplot as plt

def read_data(path_data):
    with open(path_data, 'r') as file:
        txt = [x.split(',') for x in file.read().splitlines()]
    data = np.array(txt, dtype=np.float32)
    return data[:, :-1], data[:, -1]

class NaiveBayes:
    def __init__(self):
        self.__statistics = None
    
    def __NormProb(self, x, mean, stdev):
        exp = np.exp(-((x - mean) ** 2)/(2 * (stdev ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exp

    def fit(self, x_set, t_set):
        splitByClass, stats = {}, {}
        for cls in np.unique(t_set): # split featurs by class
            splitByClass[cls] = x_set[t_set == cls]
        for cls, attrs in splitByClass.items(): # parameters by attrs
            stats[cls] = (np.mean(attrs, axis=0), np.std(attrs, axis=0, ddof=1))
        self.__statistics = stats

    def predict(self, x_set):
        pred = []
        for cls, attrStat in self.__statistics.items():
            mean, std = attrStat
            exp = np.exp(-((x_set - mean) ** 2)/(2 * (std ** 2)))
            prob = (1. / (np.sqrt(2 * np.pi) * std)) * exp
            prob = np.prod(prob, axis=1) # P(x_1,...,x_d | C)
            pred.append(prob)
        return np.stack(pred, axis=1)

    def score(self, x_set, t_set):
        pred = self.predict(x_set)
        return np.sum(pred.argmax(axis=1) == t_set) / t_set.shape[0]

data, targets = read_data('data/spambase.data')

# Split data
N = data.shape[0]
index_data = np.arange(N)
np.random.shuffle(index_data)
index_train = index_data[:int(N * 0.8)]
index_test = index_data[int(N * 0.8):]
x_train = data[index_train]
t_train = targets[index_train]
x_test = data[index_test]
t_test = targets[index_test]

model = NaiveBayes()
model.fit(x_train, t_train)
accuracy = model.score(x_test, t_test)
print('Test Accuracy: {}'.format(accuracy))

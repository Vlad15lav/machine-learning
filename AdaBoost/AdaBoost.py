import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import load_digits

class AdaBoost:
    def __init__(self, size, threshold=[0, 16], stepThreshold=0.5):
        self.__size = size
        self.__weights = []
        self.__classifiers = []

        self.__threshold = threshold
        self.__stepThreshold = stepThreshold
    
    # Loss of weak classifier
    def __Error(self, t_set, y, w_set):
        y = np.array(y)
        I = np.zeros(len(t_set))
        I[t_set != y] = 1
        return np.sum(I * w_set)

    # Training weak classifier
    def __GetParams(self, x_set, t_set, w_set):
        threshold = 0
        coord_ind = 0
        minJ = pow(10, 30)
        predict = list()
        for j in range(x_set.shape[1]):
            for tau in np.arange(self.__threshold[0], self.__threshold[1], self.__stepThreshold):
                predict.clear()

                for i in range(len(x_set)):
                    if x_set[i][j] <= tau:
                        predict.append(-1)
                    else:
                        predict.append(1)

                J = self.__Error(t_set, predict, w_set)
                if J < minJ: # Best parametrs
                    threshold = tau
                    coord_ind = j
                    minJ = J
        return {'Index': coord_ind, 'Threshold': threshold}, minJ

    # Create internal node
    def __CreateSplitNode(self, params):
        return {'Index_Cord': params['Index'], 'Threshold': params['Threshold'], 'isTerminal': False}
    
    # Classification of a weak classifier
    def __WeakClassifier(self, x_set, node):
        predict = list()
        for i in range(len(x_set)):
            if x_set[i][node['Index_Cord']] <= node['Threshold']:
                predict.append(-1)
            else:
                predict.append(1)
        return np.array(predict)

    # Create weak classifier
    def __CreateTree(self, x_set, t_set, w_set):
        params, J = self.__GetParams(x_set, t_set, w_set)
        node = self.__CreateSplitNode(params)
        predict = self.__WeakClassifier(x_set, node)
        node['left'] = {'Class': -1, 'isTerminal': True}
        node['right'] = {'Class': 1, 'isTerminal': True}

        return node, J, predict
    
    # Update Weight
    def __WeightUpdate(self, w_set, alpha, y, t_set):
        I = np.zeros(len(t_set))
        I[t_set != y] = 1
        Z = np.sum(w_set * np.exp(alpha * I))
        return (w_set * np.exp(alpha * I)) / Z
    
    # Fit model
    def fit(self, x_set, t_set):
        #w_train = np.ones(len(x_train)) / len(x_train)
        w_train = np.ones(t_set.shape[0])
        w_train[w_train == -1] /= (2 * np.sum(t_set == -1))
        w_train[w_train == 1] /= (2 * np.sum(t_set == 1)) 


        y_i = list()
        alpha_i = list()

        for i in range(self.__size):
            Cur_y, J_i, Predict = self.__CreateTree(x_set, t_set, w_train)
            if J_i > 0.5:
                break
            Cur_alpha = np.log((1 - J_i) / J_i)
            w_train = self.__WeightUpdate(w_train, Cur_alpha, Predict, t_train)

            self.__weights.append(Cur_alpha)
            self.__classifiers.append(Cur_y)
    
    # Output of the weak classifier
    def __DecisionTree(self, X, Node):
        while not Node['isTerminal']:
            if X[Node['Index_Cord']] <= Node['Threshold']:
                Node = Node['left']
            else:
                Node = Node['right']
        return Node['Class']

    # Predict
    def predict(self, x):
        if len(x.shape) == 1:
            for i in range(self.__size):
                Y += self.__weights[i] * self.__DecisionTree(x, self.__classifiers[i])
            return np.sign(Y)
        else:
            predict = []
            for data in x:
                Y = 0
                for i in range(self.__size):
                    Y += self.__weights[i] * self.__DecisionTree(data, self.__classifiers[i])
                predict.append(np.sign(Y))
            return np.array(predict)
    
    def metrics(self, x_set, t_set):
        if self.__weights == []:
            return
        
        x_pred = self.predict(x_set)
        TP = np.sum(x_pred[x_pred == t_set] == -1)
        TN = np.sum(x_pred[x_pred == t_set] == 1)
        FN = np.sum(t_set == -1) - TP
        FP = np.sum(t_set == 1) - TN
        return {'Accuracy': (TP + TN) / len(x_pred), 'Alpha error': FP / (FP + TN), 'Beta error': FN / (FN + TP)}


# Data Set Digits
digits = load_digits()
Data = digits.data
Target = digits.target
K = 2

# Choice of class C-1 and C1
print('Choose the number C{-1}: ', end='')
FirstDigit = int(input())
print('Choose the number C{1}: ', end='')
SecondDigit = int(input())
indexSet = np.arange(len(Data))
indexSet = np.concatenate((indexSet[Target == FirstDigit], indexSet[Target == SecondDigit]))
x = Data[indexSet]
N = len(x)
t = np.ones(N)
t[:len(Target[Target == FirstDigit])] = -1

# Split dataset
index_data = np.arange(N)
np.random.shuffle(index_data)
index_train = index_data[:int(N * 0.8)]
index_test = index_data[int(N * 0.8):]
x_train = x[index_train]
t_train = t[index_train]
x_test = x[index_test]
t_test = t[index_test]

# Parametrs
MaxThr = 16 # Max threshold
StepThr = 0.5 # Step threshold
sizeModel = 5 # Count weak classifiers

# Training model
model = AdaBoost(size=sizeModel, threshold=[0, MaxThr], stepThreshold=StepThr)
model.fit(x_train, t_train)

# Result model
predTrain = model.predict(x_train)
predTest = model.predict(x_test)

print('Train set - {}'.format(model.metrics(x_train, t_train)))
print('Test set - {}'.format(model.metrics(x_test, t_test)))
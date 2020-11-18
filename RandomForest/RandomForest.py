import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import load_digits

class RandomForest:
    def __init__(self, maxDeep=10, minData=20, minCriterion=0.05, typeCriterion=['Entropy', 'Gini', 'Error'], threshold=[0, 16], stepThreshold=0.5, countTrees=10, Bagging=True, RandNodeOpt=True):
        self.__countTrees = countTrees
        self.__nodes = []
        self.__paramsTrees = []
        
        self.__maxDeep = maxDeep
        self.__minData = minData
        self.__minCriterion = minCriterion
        self.__typeCriterion = typeCriterion

        self.__threshold = threshold
        self.__stepThreshold = stepThreshold

        self.__RandNodeOpt = RandNodeOpt
        self.__Bagging = Bagging

    # Get parametrs
    def get(self):
        return self.__paramsTrees

    # Get value criterion
    def __Criterion(self, t_set):
        t_set = np.array(t_set)
        H = 0
        N_i = len(t_set)
        if N_i == 0:
            return 0
        if self.__typeCriterion == 'Entropy':
            for i in range(K):
                N_ik = len(t_set[t_set == i])
                if N_ik == 0:
                    continue
                H += (N_ik / N_i) * np.log(N_ik / N_i)
            return H * (-1)
        elif self.__typeCriterion == 'Gini':
            for i in range(K):
                N_ik = len(t_set[t_set == i])
                H += (N_ik / N_i) ** 2
            return 1 - H
        elif self.__typeCriterion == 'Error':
            MaxN_ik = 0
            for i in range(K):
                N_ik = len(t_set[t_set == i])
                if N_ik > MaxN_ik:
                    MaxN_ik = N_ik
            return 1 - (MaxN_ik / N_i)
    
    # Create Terminal Node
    def __CreateTerminalNode(self, t_set):
        t = np.zeros(K)
        N_i = len(t_set)
        if N_i == 0:
            return {'Confidence': t, 'isTerminal': True}
        for i in range(K):
            t[i] = len(t_set[t_set == i]) / N_i
        return {'Confidence': t, 'isTerminal': True}

    # Create Internal Node
    def __CreateSplitNode(self, params):
        return {'Index_Cord': params['Index'], 'Threshold': params['Threshold'], 'isTerminal': False}
    
    # Get parameters split function
    def __GetParams(self, x_set, t_set):
        threshold = 0
        maxI = -1
        left = list()
        right = list()
        coord_ind = 0
        features = np.arange(x_set.shape[1]) # Features
        if self.__RandNodeOpt:
            np.random.shuffle(features)
            features = np.random.choice(features, size=np.random.randint(1, len(features) + 1 ), replace=False)
        
        for j in features:
          for t in np.arange(self.__threshold[0], self.__threshold[1], self.__stepThreshold):
                left.clear()
                right.clear()

                for i in range(len(x_set)):
                    if x_set[i][j] <= t:
                        left.append(t_set[i])
                    else:
                        right.append(t_set[i])

                I = self.__Criterion(t_set) - ((len(left) / len(t_set)) * self.__Criterion(left) +  (len(right) / len(t_set)) * self.__Criterion(right))
                if I > maxI: # Best parametrs
                    threshold = t
                    coord_ind = j
                    maxI = I

        return {'Index': coord_ind, 'Threshold': threshold}

    # Split function
    def __SplitData(self, x_set, params):
        Index = params['Index']
        t = params['Threshold']
        left = list()
        right = list()
        for i in range(len(x_set)):
            if x_set[i][Index] <= t:
                left.append(i)
            else:
                right.append(i)
        return left, right
    
    # Создание дерева
    def __CreateTree(self, x_set, t_set, deep):
        I = self.__Criterion(t_set)
        if deep > self.__maxDeep or len(x_set) < self.__minData or I < self.__minCriterion:
            node = self.__CreateTerminalNode(t_set)
            return node

        params = self.__GetParams(x_set, t_set)
        left_data, right_data = self.__SplitData(x_set, params)
        node = self.__CreateSplitNode(params)
        deep += 1
        node['left'] = self.__CreateTree(x_set[left_data], t_set[left_data], deep)
        node['right'] = self.__CreateTree(x_set[right_data], t_set[right_data], deep)

        return node

    # Fit
    def fit(self, x_train, t_train):
        maxDeep = self.__maxDeep
        minData = self.__minData
        minCriterion = self.__minCriterion
        Criterions = self.__typeCriterion

        for i in range(self.__countTrees):
            if self.__Bagging:
                idx = np.random.choice(np.arange(len(x_train)), size=np.random.randint(1, len(x_train) + 1 ), replace=False)
                x_set = x_train[idx]
                t_set = t_train[idx]
                
            self.__maxDeep = np.random.randint(1, maxDeep)
            self.__minData = np.random.randint(5, minData)
            self.__minCriterion = np.random.randint(1, int(minCriterion * 100)) / 100
            self.__typeCriterion = np.random.choice(Criterions)

            Tree = self.__CreateTree(x_set, t_set, 0)
            self.__nodes.append(Tree)
            self.__paramsTrees.append({'Max Deep': self.__maxDeep, 'Min Data': self.__minData, 'Min Criterion': self.__minCriterion, 'Type Criterion': self.__typeCriterion})
                      
        self.__maxDeep = maxDeep
        self.__minData = minData
        self.__minCriterion = minCriterion
        self.__typeCriterion = Criterions
    
    # Predict
    def predict(self, X):        
        if len(X.shape) == 1:
            conf = []
            for node in self.__nodes:
                curNode = node
                while not curNode['isTerminal']:
                    if X[curNode['Index_Cord']] <= curNode['Threshold']:
                        curNode = curNode['left']
                    else:
                        curNode = curNode['right']
                conf.append(curNode['Confidence'])
            return np.mean(conf, axis=0)
        else:
            pred = list()
            for i in range(X.shape[0]):
                conf = []
                for node in self.__nodes:
                    curNode = node
                    while not curNode['isTerminal']:
                        if X[i][curNode['Index_Cord']] <= curNode['Threshold']:
                            curNode = curNode['left']
                        else:
                            curNode = curNode['right']
                    conf.append(curNode['Confidence'])
                pred.append(np.mean(conf, axis=0))
            return pred

# Functions
# Build Confusion Matrix
def ConfusionMatrix(t_set, t_predict):
    SizeSet = len(t_set)
    Matrix = np.zeros((K, K))
    for i in range(SizeSet):
        Matrix[t_set[i]][int(t_predict[i])] += 1
    return Matrix

# Calculate Accuracy
def Accuracy(Matrix):
    Acc = 0
    for i in range(len(Matrix)):
        Acc += Matrix[i][i]
    return Acc / np.sum(Matrix)


# Data Set Digits
digits = load_digits()
N = len(digits.data)
D = len(digits.data[0])
K = len(digits.target_names)
x = np.copy(digits.data)
Target = digits.target
Images = digits.images

# Split Data
index_data = np.arange(N)
np.random.shuffle(index_data)
index_train = index_data[:int(N * 0.8)]
index_test = index_data[int(N * 0.8):]
x_train = x[index_train]
t_train = Target[index_train]
x_test = x[index_test]
t_test = Target[index_test]

# Decision Tree params
MaxDeep = 14 # Max Deep
MinData = 50 # Min data in node
MinCrit = 0.1 # Min value criterion
criterion = ['Entropy', 'Gini', 'Error'] # Type criterion: Entropy, Gini, Error
# Split data parametrs
MaxThr = 16 # Max threshold
StepThr = 0.5 # Step threshold
countTrees = 8 # Trees count
isBagging=True # Bagging method
isRandNodeOpt=True # Random Node Optimization method

# Fit model
model = RandomForest(maxDeep=MaxDeep, minData=MinData, minCriterion=MinCrit, typeCriterion=criterion, threshold=[0, MaxThr], stepThreshold=StepThr, countTrees=countTrees, Bagging=isBagging, RandNodeOpt=isRandNodeOpt)
model.fit(x_train, t_train)
print(model.get())

# Train and Test classification
predTrain = model.predict(x_train)
predTest = model.predict(x_test)

predTrain = np.argmax(predTrain, axis=1)
predTest = np.argmax(predTest, axis=1)

# Metrics
print('(Train Set) Confusion Matrix:')
ConfMatrixTrain = ConfusionMatrix(t_train, predTrain)
print(ConfMatrixTrain)
print('(Train Set) Accuracy: {}'.format(Accuracy(ConfMatrixTrain)))

print('(Test Set) Confusion Matrix:')
ConfMatrixTest = ConfusionMatrix(t_test, predTest)
print(ConfMatrixTest)
print('(Test Set) Accuracy: {}'.format(Accuracy(ConfMatrixTest)))

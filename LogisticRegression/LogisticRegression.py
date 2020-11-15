import numpy as np
import matplotlib.pyplot as plt
import pickle
import sklearn as sk
from sklearn.datasets import load_digits

class LogisticRegression:
    def __init__(self, X_Train, T_Train, X_Validation, T_Validation):
        self.X_Train = X_Train
        self.T_Train = T_Train
        self.X_Validation = X_Validation
        self.T_Validation = T_Validation
        self.W_Model = np.random.randn(K, D) * 0.05
        self.B_Model = np.random.randn(K) * 0.05


    # Initialization normal distribution
    def InitNormal(self, sigma):
        self.W_Model = np.random.randn(K, D) * sigma
        self.B_Model = np.random.randn(K) * sigma

    # Initialization uniform distribution
    def InitUniform(self, eps):
        self.W_Model = np.random.uniform(-eps, eps, (K, D))
        self.B_Model = np.random.uniform(-eps, eps, K)

    # Initialization Xavier
    def InitXavier(self):
        self.W_Model = np.random.randn(K, D) * np.sqrt(1 / K)
        self.B_Model = np.random.randn(K) * np.sqrt(1 / K)

    # Initialization He
    def InitHe(self):
        self.W_Model = np.random.randn(K, D) * np.sqrt(2 / K)
        self.B_Model = np.random.randn(K) * np.sqrt(2 / K)

    # Softmax
    def Softmax(self, vec):
        vec -= np.max(vec)
        factor = np.sum(np.exp(vec))
        res = np.float64(np.exp(vec) / factor)
        return res

    # Matrix consisting of softmax vectors
    def GetY(self, x_set, W, b):
        Y = np.zeros((len(x_set), K))
        for i in range(len(x_set)):
            Y[i] = self.Softmax(W @ x_set[i] + b)
        return Y

    # Cross Entropy loss
    def Error(self, t_set, Y):
        E = 0
        for i in range(len(Y)):
            for k in range(K):
                E += t_set[i][k] * np.log(Y[i][k])
        return E * (-1)

    # Calculating Accuracy
    def Accuracy(self, t_set, Y):
        Predict = 0
        size_set = len(Y)
        for i in range(size_set):
            if np.argmax(Y[i]) == np.argmax(t_set[i]):
                Predict += 1
        return Predict / size_set

    # Create the confusion matrix
    def ConfusionMatrix(self, x_set, t_set):
        Matrix = np.zeros((K, K))
        for i in range(len(x_set)):
            Matrix[np.argmax(t_set[i])][np.argmax(self.Softmax(self.W_Model @ x_set[i] + self.B_Model))] += 1
        return Matrix

    def Fit(self, Iters=100, LR=0.005, Eps=pow(10, -6), Lambda=0.005):
        iters_count = Iters
        learning_rate = LR
        eps_norm = Eps

        # Settings for charts
        Train_Err_iters = list()
        Train_Accuracy_iters = list()
        Val_Err_iters = list()
        Val_Accuracy_iters = list()
        ItersList = list()
        fig, axes = plt.subplots(2, 2, num='Training')
        axes[0][0].set_title('Train Set\nError / iter')
        axes[1][0].set_title('Accuracy / iter')
        axes[0][1].set_title('Validation Set\nError / iter')
        axes[1][1].set_title('Accuracy / iter')
        fig.set_figwidth(8)
        fig.set_figheight(7)
        plt.ion()

        print('(Gradient descent): Training...')
        print('Training parameters:\nIterations - ' + str(
            iters_count) + '\nLearning Rate - ' + str(
            learning_rate) + '\nEpsilon for norm of the gradient - ' + str(
            eps_norm) + '\nRegularization - ' + str(Lambda))

        iter = 0
        best_Accuracy = 0
        best_Error = pow(10, 10)
        W_k = self.W_Model
        B_k = self.B_Model
        while iter < iters_count:
            y = self.GetY(self.X_Train, W_k, B_k)
            nabla_W_E = (y - self.T_Train).T @ self.X_Train + Lambda * W_k
            nabla_B_E = (y - self.T_Train).T @ np.ones(len(self.X_Train))
            W_k_1 = W_k - learning_rate * nabla_W_E
            B_k_1 = B_k - learning_rate * nabla_B_E

            y = self.GetY(self.X_Train, W_k_1, B_k_1)
            cur_Accuracy = self.Accuracy(self.T_Train, y)
            cur_Error = self.Error(self.T_Train, y)

            y_Val = self.GetY(self.X_Validation, W_k_1, B_k_1)
            val_Accuracy = self.Accuracy(self.T_Validation, y_Val)
            val_Error = self.Error(self.T_Validation, y_Val)

            # Stopping criterion
            if np.linalg.norm(W_k - W_k_1) < eps_norm or val_Error > best_Error or val_Accuracy < best_Accuracy:
                break
            if iter % 2 == 0:
                print('(Iter ' + str(iter) + ') Train Set: Accuracy - ' + str(
                    float('{:.5f}'.format(cur_Accuracy))) + ', Error - ' + str(
                    float('{:.5f}'.format(cur_Error))) + ' | Validation Set: Accuracy - ' + str(
                    float('{:.5f}'.format(val_Accuracy))) + ', Error - ' + str(float('{:.5f}'.format(val_Error))))

            W_k = W_k_1
            B_k = B_k_1

            ItersList.append(iter)
            Val_Err_iters.append(val_Error)
            Val_Accuracy_iters.append(val_Accuracy)
            Train_Err_iters.append(cur_Error)
            Train_Accuracy_iters.append(cur_Accuracy)
            best_Accuracy = val_Accuracy
            best_Error = val_Error

            axes[0][0].plot(ItersList, Train_Err_iters, 'r.-')
            axes[1][0].plot(ItersList, Train_Accuracy_iters, 'b.-')
            axes[0][1].plot(ItersList, Val_Err_iters, 'r.-')
            axes[1][1].plot(ItersList, Val_Accuracy_iters, 'b.-')

            plt.pause(0.1)
            iter += 1

        plt.ioff()
        self.W_Model = W_k
        self.B_Model = B_k

        print('(Gradient descent): Done!')

    def Predict(self, x_set):
        Pred = np.zeros(len(x_set))
        for i in range(len(x_set)):
            Pred[i] = np.argmax(self.Softmax(self.W_Model @ x_set[i] + self.B_Model))
        return Pred


    def SaveModel(self):
        f = open(r'SaveModel.pickle', 'wb')
        obj = [self.W_Model, self.B_Model]
        pickle.dump(obj, f)
        print('The model is saved to file - SaveModel.pickle!')
        f.close()

    def LoadModel(self):
        f = open(r'SaveModel.pickle', 'rb')
        obj = pickle.load(f)
        self.W_Model = obj[0]
        self.B_Model = obj[1]
        f.close()

def Accuracy(Matrix):
    Acc = 0
    for i in range(len(Matrix)):
        Acc += Matrix[i][i]
    Acc /= np.sum(Matrix)
    return Acc

# Data Set Digits
digits = load_digits()
N = len(digits.data)
D = len(digits.data[0])
K = len(digits.target_names)
x = np.copy(digits.data)
Target = digits.target
Data = digits.data
Images = digits.images


# Startsize data and convert the One-Hot Encoding vectors
mu = np.mean(x, axis=0)
sigma = np.std(x, axis=0)
x -= mu
sigma[sigma == 0] = 1
x /= sigma
t = np.zeros((N, K))
for i in range(N):
    t[i][Target[i]] = 1


# Split data
index_data = np.arange(N)
np.random.shuffle(index_data)
index_train = index_data[:int(N * 0.8)]
index_validation = index_data[int(N * 0.8):]
x_train = x[index_train]
t_train = t[index_train]
x_validation = x[index_validation]
t_validation = t[index_validation]

# Main

# Create model
model = LogisticRegression(x_train, t_train, x_validation, t_validation)
# Inti
model.InitNormal(0.05)
#model.InitUniform(0.05)
#model.InitXavier()
#model.InitHe()
# Training
model.Fit()
# Predict
model.Predict(x_validation)
# Result Accuracy and Confusion Matrix
ConfusionMatrix = model.ConfusionMatrix(x_validation, t_validation)
print('Accuracy on Validation Set: ' + str(Accuracy(ConfusionMatrix)))
print(ConfusionMatrix)

# Save model
print('Save the model?(y/n): ', end='')
if input() == 'y':
    model.SaveModel()
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
Data = np.load('dataset.npy')
Target = np.load('target.npy')
N = Data.shape[0] // 2

# Standardization of data
mu = np.mean(Data, axis=0)
sigma = np.std(Data, axis=0)
Data -= mu
Data /= sigma

# Split data
SizeData = N + N
index_data = np.arange(SizeData)
np.random.shuffle(index_data)
index_train = index_data[:int(SizeData * 0.8)]
index_validation = index_data[int(SizeData * 0.8):]
x_train = Data[index_train]
t_train = Target[index_train]
x_validation = Data[index_validation]
t_validation = Target[index_validation]


# Функции
# Distributes the powers of polynomials over the components of the vector for each basis function
def GetSteps(Step):
    MaxStep = len(Step)
    StepsMatrix = np.zeros((MaxStep, 2))
    for i in range(MaxStep):
        StepsMatrix[i][0] = Step[i]
    StepsMatrix[1][0] = 0
    StepsMatrix[1][1] = 1
    return StepsMatrix


# Standardization of data
def Standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    data -= mu
    data /= sigma
    return data


# Calculation error
def Error(Y, t_set):
    E = 0
    for i in range(len(Y)):
        E += t_set[i] * np.log(Y[i]) + (1 - t_set[i]) * np.log(1.0 - Y[i])
    return E * (-1)

# Получить матрицу плана
def GetPlanMatrix(x_set, Steps):
    Plan_Matrix = np.ones((len(x_set), len(Steps)))
    for i in range(len(Steps)):
        Plan_Matrix[:, i] = (x_set[:, 0] ** Steps[i][0]) * (x_set[:, 1] ** Steps[i][1])
    return Plan_Matrix


# Получить матрица состоящая из выходных векторов для выборки
def GetY(x_set, W, Steps):
    Y = np.zeros(len(x_set))
    Phi_x = np.ones(len(Steps))
    for i in range(len(x_set)):  # dataset
         for j in range(len(Steps)): # degrees
            Phi_x[j] = (x_set[i][0] ** Steps[j][0]) * (x_set[i][1] ** Steps[j][1])

         a = np.float64(W.T @ Phi_x)
         a = np.clip(a, -709, 20)

         Y[i] = 1.0 / (1.0 + np.exp(a * (-1)))
    return Y


# Parameter initialization
# Steps = np.array([0, 1, 7, 1, 12, 8, 5, 11, 4, 10, 9, 6, 3, 2])
# Steps = np.array([0, 1, 2, 11, 6, 12, 5, 9, 1, 8, 3, 10, 4, 7])
# Steps = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
Steps = np.array([0, 1, 1, 2, 3, 4, 5]) # the degree of the polynomials
StepsMatrix = GetSteps(Steps)
Lambda = pow(10, -10)
best_Error = pow(10, 100)
IterCount = 200
init_sigma = 0.01  # 0.01 - 0.1

learning_rate = 0.00011  # 0.001 - 0.01
eps_norm = pow(10, -3)  # 10^-6   10^-3
M = len(Steps)
W_k = np.random.randn(M) * init_sigma

# The process of Gradient descent
iter = 0
Train_Err_iters = list() 
Val_Err_iters = list()
PlanMatrix = GetPlanMatrix(x_train, StepsMatrix)

print('(Gradient descent): Training...')
print('Training parameters:\nParameter initialization - ' + str(init_sigma) + '\niterations - ' + str(
    IterCount) + '\nLearning Rate - ' + str(learning_rate) + '\nEpsilon for norm gradient - ' + str(eps_norm))
while iter < IterCount:
    Y = GetY(x_train, W_k, StepsMatrix)
    nabla_W_E = (Y - t_train).T @ PlanMatrix + Lambda * W_k.T
    W_k_1 = W_k - learning_rate * nabla_W_E

    Y_Train = GetY(x_train, W_k_1, StepsMatrix)
    Train_Error = Error(Y_Train, t_train)

    Y_Val = GetY(x_validation, W_k_1, StepsMatrix)
    Val_Error = Error(Y_Val, t_validation)

    # Exit conditions
    if np.linalg.norm(W_k - W_k_1) < eps_norm or Val_Error > best_Error:
        break
    if iter % 5 == 0:
        print('(Iteration ' + str(iter) + ') Train Set: Error - ' + str(
            float('{:.5f}'.format(Train_Error))) + ' | Validation Set: Error - ' + str(
            float('{:.5f}'.format(Val_Error))))

    W_k = W_k_1
    Val_Err_iters.append(Val_Error)
    Train_Err_iters.append(Train_Error)
    best_Error = Val_Error

    iter += 1
print('(Gradient descent): Done!')

# Calculation metrics
Y = GetY(x_validation, W_k, StepsMatrix)
TP = 0
TN = 0
FN = 0
FP = 0
SizeSet = len(x_validation)
for i in range(SizeSet):
    t_Predict = 0
    if Y[i] > 0.5:
        t_Predict = 1
    if t_Predict == t_validation[i] == 1:
        TP += 1
    elif t_Predict == t_validation[i] == 0:
        TN += 1
    elif t_Predict == 0 and t_validation[i] == 1:
        FN += 1
    elif t_Predict == 1 and t_validation[i] == 0:
        FP += 1
Accuracy = (TP + TN) / SizeSet
ErrorAlpha = FP / (TN + FP)
ErrorBeta = FN / (TP + FN)
if TP + FP == 0:
    Precision = 1
else:
    Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2 * (Precision * Recall) / (Precision + Recall)
print('True Positive = ' + str(TP))
print('False Positive = ' + str(FP))
print('True Negative = ' + str(TN))
print('False Negative = ' + str(FN))
print('Accuracy = ' + str(Accuracy))
print('Precision = ' + str(Precision))
print('Recall = ' + str(Recall))
print('F1 Score = ' + str(F1_Score))
print('Alpha error = ' + str(ErrorAlpha))
print('Beta error = ' + str(ErrorBeta))

ItersCount = np.array(range(iter))
fig, axes = plt.subplots(1, 2, num='Training')
axes[0].set_title('Train Set\nError / iter')
axes[0].plot(ItersCount, Train_Err_iters, 'r.-')
axes[1].set_title('Validation Set\nError / iter')
axes[1].plot(ItersCount, Val_Err_iters, 'r.-')
fig.set_figwidth(8)
fig.set_figheight(4)
plt.show()

# Building a model that separates classes
X = np.linspace(0, 1, N)
Y = np.linspace(min(np.min(y1), np.min(y2)), max(np.max(y1), np.max(y2)), N)
X_Points = np.linspace(0, 1, N)
Y_Points = np.linspace(min(np.min(y1), np.min(y2)), max(np.max(y1), np.max(y2)), N)
X_Points -= mu[0]
X_Points /= sigma[0]
Y_Points -= mu[1]
Y_Points /= sigma[1]

X_index = list()
Y_index = list()

print('Finding the border...')
Phi_x = np.ones(M)
for i in range(N):
    for j in range(N):
         for k in range(len(StepsMatrix)):
              Phi_x[k] = (X_Points[i] ** StepsMatrix[k][0]) * (Y_Points[j] ** StepsMatrix[k][1])

         a = W_k.T @ Phi_x
         if abs(a) < 0.01:
             X_index.append(i)
             Y_index.append(j)

# Plot
plt.figure('Logistic regression')
plt.plot(x, y1, 'b.', label='С0')
plt.plot(x, y2, 'r.', label='С1')
plt.plot(X[X_index], Y[Y_index], 'k.', label='The boundary classifier')

plt.legend()
plt.show()
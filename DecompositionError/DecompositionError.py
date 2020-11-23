import numpy as np
import matplotlib.pyplot as plt

x = np.load('dataset.npy')
z = np.load('realmodel.npy')

N = x.shape[0]

index_data = np.arange(N)
np.random.shuffle(index_data)
index_train = index_data[:int(N * 0.8)]
index_test = index_data[int(N * 0.8):]
x_train = x[index_train]
z_train = z[index_train]
x_test = x[index_test]
z_test = z[index_test]


# Design matrix
def DesignMatrix(x_set, M):
    Plan_Matrix = np.zeros((x_set.shape[0], M))
    for i in range(M):
        Plan_Matrix[:, i] = x_set ** i
    return Plan_Matrix
# Train model regression
def TrainW(Plan_Matrix, target, RL):
    w = np.linalg.inv(Plan_Matrix.T @ Plan_Matrix + RL * np.eye(Plan_Matrix.shape[1])) @ Plan_Matrix.T @ target
    return w

# Parameters
begin = 7
end = 13
rounds = 20
sigma = 5

# Analysis of the decomposition
x_line = np.array(range(begin, end + 1))
global_error_list = []
bias_list = []
var_list = []
for i in range(begin, end + 1):
    pred_list = []
    error_list = []
    error_l_list = []
    for r in range(rounds):
        error = sigma * np.random.randn(int(N * 0.8))
        t_train = z_train + error

        error = sigma * np.random.randn(int(N * 0.2))
        t_test = z_test + error

        # Train model
        train_MP = DesignMatrix(x_train, i)
        w = TrainW(train_MP, t_train, 0)

        # Predict test set
        test_MP = DesignMatrix(x_test, i)
        pred = test_MP @ w

        # Error
        error_list.append(np.mean((t_test - pred) ** 2))

        # Bias and Variance
        pred_list.append(pred)

    bias_list.append(np.mean((z_test - np.mean(pred_list, axis=0)) ** 2))
    var_list.append(np.mean(np.var(pred_list, axis=0)))
    global_error_list.append(np.mean(error_list))


# Plot
plt.figure('Decomposition Error', figsize=(7, 7))
plt.plot(x_line, global_error_list, label='Error')
plt.plot(x_line, bias_list, label='Bias')
plt.plot(x_line, var_list, label='Variance')
plt.legend()
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.show()
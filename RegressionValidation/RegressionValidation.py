import numpy as np
import matplotlib.pyplot as plt

x = np.load('dataset.npy')
t = np.load('target.npy')
z = np.load('realmodel.npy')

N = x.shape[0]

# Functions:
# Get random non-repeating basis functions
def Get_Basis_Fun(Fi):
    return np.random.choice(Fi, size=np.random.randint(1, len(Fi) + 1 ), replace=False)
# To get a random lambda number
def Get_Lambda(L):
    return np.random.choice(L)
# To create the design matrix
def Get_Plan_Matrix(x_set, fi):
    Plan_Matrix = np.ones((len(x_set), len(fi) + 1))
    for i in range(0, len(fi)):
        Plan_Matrix[:, i + 1] = fi[i](x_set)
    return Plan_Matrix
# To get a regression model
def Get_W(x_set, t_set, fi, lam):
    Plan_Matrix = Get_Plan_Matrix(x_set, fi)
    Mura_Penrose = np.linalg.inv(Plan_Matrix.T @ Plan_Matrix + lam * np.eye(len(fi) + 1)) @ Plan_Matrix.T
    w = Mura_Penrose @ t_set
    return w
# To calculate the error
def Get_Error(x_set, t_set, w_val, fi):
    Plan_Matrix = Get_Plan_Matrix(x_set, fi)
    y = Plan_Matrix @ w_val
    return np.sum((y - t_set) ** 2) / 2
# Polynomials
def Polynom1(x): return x
def Polynom2(x): return x ** 2
def Polynom3(x): return x ** 3
def Polynom4(x): return x ** 4
def Polynom5(x): return x ** 5
def Polynom6(x): return x ** 6
def Polynom7(x): return x ** 7
def Polynom8(x): return x ** 8
def Polynom9(x): return x ** 9
def Polynom10(x): return x ** 10
def Polynom11(x): return x ** 11
def Polynom12(x): return x ** 12

# Train|Val|Test (80:10:10)
index_data = np.arange(N)
np.random.shuffle(index_data)
index_train = index_data[:int(N * 0.8)]
index_validation = index_data[int(N * 0.8):int(N * 0.9)]
index_test = index_data[int(N * 0.9):]
x_train = x[index_train]
t_train = t[index_train]
x_validation = x[index_validation]
t_validation = t[index_validation]
x_test = x[index_test]
t_test = t[index_test]

# Starting value
print('Iterations:', end=' ')
iter_count = int(input())
Lambda = np.zeros(31)
for i in range(1, 16):
    Lambda[i] = pow(10, -2 * i)
    Lambda[i + 15] = pow(10, 2 * i)
Fi_functions = np.array([Polynom1, Polynom2, Polynom3, Polynom4, Polynom5, Polynom6, Polynom7, Polynom8, Polynom9, np.sin, np.cos, np.sqrt])# Polynom10, Polynom11, Polynom12])

# The best options
Lambda_best = 0
Fi_basis_best = Fi_functions
E_min = pow(10, 30)
W_best = np.zeros(1)

# Iterative search for parameters
for i in range(iter_count):
    Cur_Fi = Get_Basis_Fun(Fi_functions)
    Cur_Lambda = Get_Lambda(Lambda)
    w_val = Get_W(x_train, t_train, Cur_Fi, Cur_Lambda)
    Cur_E = Get_Error(x_validation, t_validation, w_val, Cur_Fi)
    if Cur_E < E_min:
        E_min = Cur_E
        Fi_basis_best = Cur_Fi
        Lambda_best = Cur_Lambda
        W_best = w_val

# We calculate the error on the test sample, build a model, and output the best parameters
E_result = Get_Error(x_test, t_test, W_best, Fi_basis_best)
y = Get_Plan_Matrix(x, Fi_basis_best) @ W_best
print('Best regularization: ' + str(Lambda_best))
fi_str = ''
for i in range(len(Fi_basis_best)):
    fi_str += Fi_basis_best[i].__name__ + ' '
print('Best set basic functions: ' + fi_str)
print('Test error: ' + str(E_result))

# Show best train model
plt.figure('Regression')
plt.plot(x_train, t_train, 'b.', label='Train Set')
plt.plot(x_validation, t_validation, 'r.', label='Validation Set')
plt.plot(x_test, t_test, 'y.', label='Test Set')
plt.plot(x, z, 'k-', label='Real Model')
plt.plot(x, y, 'r-', label='Best Model')
plt.legend()
plt.show()
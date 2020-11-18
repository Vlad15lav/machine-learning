import numpy as np
import matplotlib.pyplot as plt

N = 1000
T = 300
Step_T = 1
isCustom = False

if isCustom:
    # Custom dataset
    print('Average height of a football player - ', end='')
    mu_0 = int(input())
    print('Standard deviation of a football players height - ', end='')
    sigma_0 = int(input())
    print('Average height of a basketball player - ', end='')
    mu_1 = int(input())
    print('Standard deviation of a basketball players height - ', end='')
    sigma_1 = int(input())

    # Dataset
    Soccer_Height = mu_0 + np.random.randn(N) * sigma_0
    Basketball_Height = mu_1 + np.random.randn(N) * sigma_1
    t_RealSoccer = np.zeros(N)
    t_RealBasketball = np.ones(N)
else:
    # Load dataset
    dataset = np.load("SportDataset.npy")
    target = np.load("SportTarget.npy")

    Soccer_Height = dataset[:1000]
    Basketball_Height = dataset[1000:]
    t_RealSoccer = target[:1000]
    t_RealBasketball = target[1000:]

# Functions
# Classification function
def Classification(t):
    t_ClassificationSoccer = np.zeros(N)
    t_ClassificationBasketball = np.zeros(N)

    t_ClassificationSoccer[Soccer_Height >= t] = 1
    t_ClassificationBasketball[Basketball_Height >= t] = 1
    return t_ClassificationSoccer, t_ClassificationBasketball

# Calculate TP FN TN FP
def Quality(t_ClassificationSoccer, t_ClassificationBasketball):
    TP = np.sum(t_ClassificationSoccer == t_RealSoccer)
    TN = np.sum(t_ClassificationBasketball == t_RealBasketball)
    
    FN = N - TP
    FP = N - TN
    return TP, FN, TN, FP
# Calculate Accuracy
def Accuracy(TP, TN):
    return (TP + TN) / (2 * N)
# Calculate Error_Alpha
def Error_Alpha(FP, TN): # Вычисление ошибки 1-го рода
    return FP / (TN + FP)
# Calculate Error_Beta
def Error_Beta(TP, FN): # Вычисление ошибки 2-го рода
    return FN / (TP + FN)
# Calculate Precision
def Precision(TP, FP):
    if TP + FP != 0:
        return TP / (TP + FP)
    else:
        return 1
# Calculate Recall
def Recall(TP, FN):
    return TP / (TP + FN)
# Calculate F1 Score
def F1_Score(Precision, Recall):
    return 2 * (Precision * Recall) / (Precision + Recall)

# Calculation of all metrics
def Metrics(TP, FN, TN, FP):
    R_Accuracy = Accuracy(TP, TN)
    R_Precision = Precision(TP, FP)
    R_Recall = Recall(TP, FN)
    R_F1_Score = F1_Score(R_Precision, R_Recall)
    R_Error_Alpha = Error_Alpha(FP, TN)
    R_Error_Beta = Error_Beta(TP, FN)
    print('Accuracy = ' + str(R_Accuracy))
    print('Precision = ' + str(R_Precision))
    print('Recall = ' + str(R_Recall))
    print('F1 Score = ' + str(R_F1_Score))
    print('Alpha error = ' + str(R_Error_Alpha))
    print('Beta error = ' + str(R_Error_Beta))

# Calculating Precision, Recall, Alpha error, Beta error for different thresholds and finding the best one
def ROC():
    Alpha_Roc = np.zeros(T)
    Rec_Roc = np.zeros(T)
    Precision_PR = np.zeros(T)
    Recall_PR = np.zeros(T)
    Best_Accuracy = 0
    t_best = 0
    for i in range(0, T, Step_T):
        t_ClassificationSoccer, t_ClassificationBasketball = Classification(i)
        TP, FN, TN, FP = Quality(t_ClassificationSoccer, t_ClassificationBasketball)
        Cur_Accuracy = Accuracy(TP, TN)
        if Best_Accuracy < Cur_Accuracy:
            t_best = i
            Best_Accuracy = Cur_Accuracy
        Alpha_Roc[i] = Error_Alpha(FP, TN)
        Rec_Roc[i] = 1 - Error_Beta(TP, FN)
        Precision_PR[i] = Precision(TP, FP)
        Recall_PR[i] = Recall(TP, FN)
    return Alpha_Roc, Rec_Roc, Precision_PR, Recall_PR, t_best

# Calculating the area
def AUC(Alpha_Roc, Rec_Roc):
    Square = 0
    for i in range(T - 1):
        Square += (Rec_Roc[i] + Rec_Roc[i + 1]) * (Alpha_Roc[i + 1] - Alpha_Roc[i]) / 2
    return Square

# Calculating metrics for the specified threshold
print('Enter the threshold: ', end='')
t = int(input())
t_ClassificationSoccer, t_ClassificationBasketball = Classification(t)
TP, FN, TN, FP = Quality(t_ClassificationSoccer, t_ClassificationBasketball)
Metrics(TP, FN, TN, FP)

# The values for curves
Alpha_Roc, Rec_Roc, Precision_PR, Recall_PR, t_best = ROC()

# Metric for the best threshold
print('Metrics for the best threshold (t = ' + str(t_best) + ')')
t_ClassificationSoccer, t_ClassificationBasketball = Classification(t_best)
TP, FN, TN, FP = Quality(t_ClassificationSoccer, t_ClassificationBasketball)
Metrics(TP, FN, TN, FP)

# Calculation AUC for ROC Curve
Area_Under_Curve = AUC(Alpha_Roc, Rec_Roc)
print('AUC for Receiver Operating Characteristic Curve = ' + str(Area_Under_Curve))

# Build Receiver Operating Characteristic Curve
plt.figure("ROC Curve")
plt.title("AUC = " + str(Area_Under_Curve))
Rand_Line = np.linspace(0, 1, N)
plt.plot(Alpha_Roc, Rec_Roc, 'y.-', label='Our classifier')
plt.plot(Rand_Line, Rand_Line, 'b--', label='Random guessing')
plt.legend()
plt.show()

# Calculation AUC for Precision-Recall Curve
Area_Under_Curve = AUC(Recall_PR, Precision_PR)
print('AUC for Precision-Recall Curve = ' + str(Area_Under_Curve))

# Build Precision-Recall Curve
plt.figure("Precision-Recall Curve")
plt.title("AUC = " + str(Area_Under_Curve))
Rand_Line = np.linspace(0, 1, N)
plt.plot(Recall_PR, Precision_PR, 'y.-', label='Our classifier')
plt.plot(Rand_Line, 1 - Rand_Line, 'b--', label='Random guessing')
plt.legend()
plt.show()

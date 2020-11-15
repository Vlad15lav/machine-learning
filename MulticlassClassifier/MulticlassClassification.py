import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import load_digits

# Data Set Digits
digits = load_digits()
N = len(digits.images)
Img_h, Img_w = digits.images[0].shape
Images = digits.images
t = digits.target

# Split data
index_data = np.arange(N)
np.random.shuffle(index_data)
index_train = index_data[:int(N * 0.8)]
index_test = index_data[int(N * 0.8):]
x_train = Images[index_train]
t_train = t[index_train]
x_test = Images[index_test]
t_test = t[index_test]

# Creating masks for all digits of images
Masks_IMG = np.zeros((10, 8, 8))
for digit in range(10):
    Count_IMG = 0
    for i in range(len(x_train)):
        if t_train[i] == digit:
            Count_IMG += 1
            Masks_IMG[digit] += x_train[i]
    Masks_IMG[digit] /= Count_IMG
    Masks_IMG[digit] -= np.sum(Masks_IMG[digit]) / (Img_h * Img_w)

# Functions
# Softmax
def Softmax(vector):
    vec = vector - max(vector)
    sum = np.sum(np.exp(vec))
    vec = np.exp(vec) / sum
    return vec

# Classifies the image
def Classifier(img_digit):
    Conviction_Vec = np.zeros(10)
    for digit in range(10):
        Conviction_Vec[digit] = np.sum(img_digit * Masks_IMG[digit])
    Conviction_Vec = Softmax(Conviction_Vec)
    return Conviction_Vec

# Classifying a DataSet from images
def Classification_Set(x_set):
    classification_set = np.zeros(len(x_set))
    for i in range(len(x_set)):
        classification_set[i] = np.argmax(Classifier(x_set[i]))
    return classification_set

# Build Confusion matrix
def Confusion_Matrix(t_set, classification_set):
    C_Matrix = np.zeros((10, 10))
    for i in range(len(t_set)):
        C_Matrix[int(t_set[i])][int(classification_set[i])] += 1
    return C_Matrix

# Calculate Accuracy
def Accuracy(conf_matrix):
    sum = 0
    for i in range(len(conf_matrix)):
        sum += conf_matrix[i][i]
    return sum / np.sum(conf_matrix)


# Classify DataSets
train_clasification = Classification_Set(x_train)
test_clasification = Classification_Set(x_test)

# Result Confusion Matrix for Train and Test
ConfusionMatrixTrain = Confusion_Matrix(t_train, train_clasification)
ConfusionMatrixTest = Confusion_Matrix(t_test, test_clasification)
print('Confusion Matrix (Train Set):')
print(ConfusionMatrixTrain)
print('Accuracy Train (Train Set): ' + str(Accuracy(ConfusionMatrixTrain)))
print('Confusion Matrix (Test Set):')
print(ConfusionMatrixTest)
print('Accuracy (Test Set): ' + str(Accuracy(ConfusionMatrixTest)))

# Show best and worse predict
Confidence_Imgs = list() # List (max confidence, index img, real class, predict class)
for i in range(len(Images)):
    Confidence_Imgs.append((max(Classifier(Images[i])), i, t[i], np.argmax(Classifier(Images[i]))))
Confidence_Imgs.sort(key=lambda mass: mass[0]) # Sort by key: confidence value

fig, axes = plt.subplots(3, 2, num='The predictions of the classifier')
axes[0][0].set_title('Best\nReal - ' + str(Confidence_Imgs[N - 1][2]) + ' Predict - ' + str(Confidence_Imgs[N - 1][3]))
axes[0][0].imshow(Images[int(Confidence_Imgs[N - 1][1])], cmap='gray')
axes[0][0].set_axis_off()

axes[1][0].set_title('Real - ' + str(Confidence_Imgs[N - 2][2]) + ' Predict - ' + str(Confidence_Imgs[N - 2][3]))
axes[1][0].imshow(Images[int(Confidence_Imgs[N - 2][1])], cmap='gray')
axes[1][0].set_axis_off()

axes[2][0].set_title('Real - ' + str(Confidence_Imgs[N - 3][2]) + ' Predict - ' + str(Confidence_Imgs[N - 3][3]))
axes[2][0].imshow(Images[int(Confidence_Imgs[N - 3][1])], cmap='gray')
axes[2][0].set_axis_off()

axes[0][1].set_title('Worse\nReal - ' + str(Confidence_Imgs[0][2]) + ' Predict - ' + str(Confidence_Imgs[0][3]))
axes[0][1].imshow(Images[int(Confidence_Imgs[0][1])], cmap='gray')
axes[0][1].set_axis_off()

axes[1][1].set_title('Real - ' + str(Confidence_Imgs[1][2]) + ' Predict - ' + str(Confidence_Imgs[1][3]))
axes[1][1].imshow(Images[int(Confidence_Imgs[1][1])], cmap='gray')
axes[1][1].set_axis_off()

axes[2][1].set_title('Real - ' + str(Confidence_Imgs[2][2]) + ' Predict - ' + str(Confidence_Imgs[2][3]))
axes[2][1].imshow(Images[int(Confidence_Imgs[2][1])], cmap='gray')
axes[2][1].set_axis_off()

fig.set_figwidth(6)
fig.set_figheight(6)
plt.show()
# Machine Learning
Topics and tasks for the classical machine learning

1. [Linear regression](https://github.com/Vlad15lav/Machine-Learning/tree/main/LinearRegression)
2. [Multy target regression](https://github.com/Vlad15lav/Machine-Learning/tree/main/MultyLinearRegression)
3. [The validation parameters](https://github.com/Vlad15lav/Machine-Learning/tree/main/RegressionValidation)
4. [Simple classifiers and metrics](https://github.com/Vlad15lav/Machine-Learning/tree/main/SimpleClassifier)
5. [Multiclass classifier](https://github.com/Vlad15lav/Machine-Learning/tree/main/MulticlassClassifier)
6. [Logistic regression](https://github.com/Vlad15lav/Machine-Learning/tree/main/LogisticRegression)
7. [Border classifier](https://github.com/Vlad15lav/Machine-Learning/tree/main/BorderClassifier)
8. [Decision Tree](https://github.com/Vlad15lav/Machine-Learning/tree/main/DecisionTree)
9. [Random Forest](https://github.com/Vlad15lav/Machine-Learning/tree/main/RandomForest)
10. [AdaBoost](https://github.com/Vlad15lav/Machine-Learning/tree/main/AdaBoost)
11. [K-means](https://github.com/Vlad15lav/Machine-Learning/tree/main/K-means)

`TODO: SVM, Bias-Variance decomposition, The Boston Housing Dataset with sklearn tools.`
# Requirements
`pip install -U -r requirements.txt`

## Linear regression
Train a simple regression model using the Moore-Penrose matrix. Polynomial regression.</br>
![](/LinearRegression/dataset.png)
# Multy target regression
Solar Flare Data Set. Decomposition QR, SVD.
![](/MultyTargetRegression/solarflare.jpg)
## The validation parameters
Validation of hyperparameters. Regularization. Overfitting and Underfitting.</br>
![](/RegressionValidation/training.png)
## Simple classifiers and metrics
Simple classifier of football and basketball players.</br>
Creating a custom dataset using normal distribution.</br>
Metrics: TP, TN, FP, FN, Error Alpha, Error Beta, Accuracy, Precision, Recall, F1-score, ROC, PRC, AUC
## Multiclass classifier
Dataset digits sklearn. Classification of 10 digits. </br>
Classifier - the scalar product of the masked figures.</br>
![](/MulticlassClassifier/digits.png)
## Logistic regression
Logistic regression model. Softmax. Gradient descent. Initialization Xavier, He.</br>
![](/LogisticRegression/GradientDescent.png)</br>
![](/LogisticRegression/train.gif)
## Border classifier
Binary classification. Sigmoid. Building a border that divides the space.</br>
![](/BorderClassifier/classifier.png)
## Decision Tree
Binary decision tree. Root node, internal node (weak classifier), terminal node.</br>
Split function(hyperplanes parallel to the coordinate axes). Feature selection function.</br>
Criterions - Entropy (Information Gain), Gini, Error.</br>
Criterions for stopping tree growth - max deep, min sample, min criterion.</br>
![](/DecisionTree/ExampleTree.png)
## Random Forest
Ensemble of classifiers. Different trees. Bagging. Random Node Optimization.
![](/RandomForest/ExampleRandomForest.png)
## AdaBoost
Training on a weighted sample. Strong classifier - a set of weighted weak classifiers.</br>
Calculating the weight for a weak classifier and updating the sample weights.</br>
## K-means
Clusterings. The update of centroids. Elbow method.</br>
![](/K-means/eblowmethod.png)</br>
![](/K-means/training.png)

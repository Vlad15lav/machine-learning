# Machine Learning
Topics and tasks for the classical Machine Learning

1. [Linear regression](https://github.com/Vlad15lav/Machine-Learning/tree/main/LinearRegression)
2. [Multy target regression](https://github.com/Vlad15lav/Machine-Learning/tree/main/MultyTargetRegression)
3. [Decomposition Error](https://github.com/Vlad15lav/Machine-Learning/tree/main/DecompositionError)
4. [Validation parameters](https://github.com/Vlad15lav/Machine-Learning/tree/main/RegressionValidation)
5. [Simple classifiers and metrics](https://github.com/Vlad15lav/Machine-Learning/tree/main/SimpleClassifier)
6. [Multiclass classifier](https://github.com/Vlad15lav/Machine-Learning/tree/main/MulticlassClassifier)
7. [Logistic regression](https://github.com/Vlad15lav/Machine-Learning/tree/main/LogisticRegression)
8. [Border classifier](https://github.com/Vlad15lav/Machine-Learning/tree/main/BorderClassifier)
9. [Decision Tree](https://github.com/Vlad15lav/Machine-Learning/tree/main/DecisionTree)
10. [Random Forest](https://github.com/Vlad15lav/Machine-Learning/tree/main/RandomForest)
11. [AdaBoost](https://github.com/Vlad15lav/Machine-Learning/tree/main/AdaBoost)
12. [K-means](https://github.com/Vlad15lav/Machine-Learning/tree/main/K-means)
13. [KNN](https://github.com/Vlad15lav/Machine-Learning/tree/main/KNN)
14. [Naive Bayes](https://github.com/Vlad15lav/Machine-Learning/tree/main/NaiveBayes)
15. [SVM](https://github.com/Vlad15lav/Machine-Learning/tree/main/SVM)
16. [GradientBoost](https://github.com/Vlad15lav/Machine-Learning/tree/main/GradientBoost)
17. [Sklearn](https://github.com/Vlad15lav/Machine-Learning/tree/main/Sklearn)

# Requirements
```
pip install -U -r requirements.txt
```

## Linear regression
Train a simple regression model using the Moore-Penrose matrix. Polynomial regression.</br>
![](/LinearRegression/dataset.png)

## Multy target regression
Solar Flare Data Set. Decomposition QR, SVD.</br>
![](/MultyTargetRegression/solarflare.jpg)

## Decomposition Error
Decomposition error on bias and variance. Underfitting and Overfitting.</br>
<img src="/DecompositionError/decomp.png" alt="drawing" width="450"/>

## Validation parameters
Validation of hyperparameters. Regularization.</br>
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
<img src="/LogisticRegression/GradientDescent.png" alt="drawing" width="550"/>
<img src="/LogisticRegression/train.gif" alt="drawing" width="550"/>

## Border classifier
Binary classification. Sigmoid. Building a border that divides the space.</br>
![](/BorderClassifier/classifier.png)

## Decision Tree
Binary decision tree. Root node, internal node (weak classifier), terminal node.</br>
Split function(hyperplanes parallel to the coordinate axes). Feature selection function.</br>
Criterions - Entropy (Information Gain), Gini, Error.</br>
Criterions for stopping tree growth - max deep, min sample, min criterion.</br>
<img src="/DecisionTree/ExampleTree.png" alt="drawing" width="550"/>

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

## K-nearest neighbor classifier
L1 distance.</br>
<img src="/KNN/knn.png" alt="drawing" width="550"/>

## Naive Bayes
Bayes' Theorem. Naive assumption of feature independence. Classifying Email as Spam or Non-Spam.</br>
<img src="/NaiveBayes/naive-bayes.png" alt="drawing" width="550"/>

## Support Vector Machine
<img src="/SVM/svm.png" alt="drawing" width="550"/>

## Gradient Boost
Decision Stump ensemble. Boston Housing Dataset.

## Sklearn
Use Sklearn tools for Titanic Dataset by Kaggle. Accuracy - 0.77990.

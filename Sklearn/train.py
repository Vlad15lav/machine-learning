import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Read CSV
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Feature Enginering
def FeatureEngine(data, train=True):
    if train:
        set_y = data['Survived']
        set_x = data.drop(columns=['PassengerId','Survived', 'Name',
                                  'Ticket', 'Cabin'])
    else:
        set_x = data.drop(columns=['PassengerId', 'Name',
                                  'Ticket', 'Cabin'])
    
    set_x = set_x.fillna(0) 

    set_x['Sex'][set_x['Sex'] == 'male'] = 0
    set_x['Sex'][set_x['Sex'] == 'female'] = 1

    set_x['Embarked'][set_x['Embarked'] == 'S'] = 1
    set_x['Embarked'][set_x['Embarked'] == 'C'] = 2
    set_x['Embarked'][set_x['Embarked'] == 'Q'] = 3
    if train:
        return set_x, set_y
    else:
        return set_x

# Train/Val 85:15
train_X, train_y = FeatureEngine(train, train=True)
X_train, X_val, Y_train, Y_val = train_test_split(np.array(train_X), np.array(train_y), test_size=0.15, random_state=5)

# Test
X_test, Y_test = FeatureEngine(train)

# Experiments
# Logit
model = LogisticRegression(random_state=0)
model.fit(X_train, Y_train)
result = model.score(X_val, Y_val)
print('Logit| Val score - {}'.format(result * 100))

# CVM
model = SVC(kernel='linear')
model.fit(X_train, Y_train)
result = model.score(X_val, Y_val)
print('SVM| Val score - {}'.format(result * 100))

# XGBoost
model = XGBClassifier(max_depth=12, learning_rate=0.5, n_estimators=100, reg_lambda=0.1)
model.fit(X_train, Y_train)
result = model.score(X_val, Y_val)
print('XGBoost| Val score - {}'.format(result * 100))

# Naive Bayes
model = GaussianNB()
model.fit(X_train, Y_train)
result = model.score(X_val, Y_val)
print('Naive Bayes| Val score - {}'.format(result * 100))

# AdaBoost
model = AdaBoostClassifier(n_estimators=12, random_state=0)
model.fit(X_train, Y_train)
result = model.score(X_val, Y_val)
print('AdaBoost| Val score - {}'.format(result * 100))

# Random Forest
forrest_params = dict(     
    max_depth = [n for n in range(9, 18)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(80, 160, 10)],
)

model = RandomForestClassifier()
model_cv = GridSearchCV(estimator=model, param_grid=forrest_params, cv=5) 
model_cv.fit(X_train, Y_train)
result = model_cv.score(X_val, Y_val)
print('RF| Val score - {}'.format(result * 100))

# Create Kaggle Submission
pred_test = model_cv.predict(X_test)
df = pd.DataFrame({"PassengerId": test['PassengerId'], "Survived": pred_test})
df.to_csv('gender_kaggle.csv',index = False)
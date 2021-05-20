import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import decomposition
from sklearn.pipeline import Pipeline


# Load dataset

path = "C:/Wz Python Projects/Wonder_tk_MSc/DPES Rhoda Final/Specific Objective 2.xlsx"


dataset = pd.read_excel(path)


# Define X, Y

array = dataset.values

X = array[:, 1:12]

Y = array[:, 0]


# Validation Size

test_size = 0.20

seed = 14


# Train, Test Splitting for further Stratified Splitting

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed, stratify=Y)


# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Using PCA

pca = decomposition.PCA()


"""
Fitting LR to training set
"""

logistic_Reg = LogisticRegression()


"""
Using Pipeline for GridSearchCV
"""

pipe = Pipeline(steps=[('sc', sc),
                       ('pca', pca),
                       ('logistic_Reg', logistic_Reg)])

n_components = list(range(1, X.shape[1]+1, 1))

C = np.logspace(-4, 4, 50)

penalty = ['l1', 'l2']

parameters = dict(pca__n_components=n_components,
                  logistic_Reg__C=C,
                  logistic_Reg__penalty=penalty)


clf_GS = GridSearchCV(pipe, parameters)

clf_GS.fit(X, Y)


print('Best Penalty:', clf_GS.best_estimator_.get_params()['logistic_Reg__penalty'])
print('Best C:', clf_GS.best_estimator_.get_params()['logistic_Reg__C'])
print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
print(clf_GS.best_estimator_.get_params()['logistic_Reg'])

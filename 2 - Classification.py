import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# Load dataset

path = "C:/Wz Python Projects/Wonder_tk_MSc/DPES Rhoda Final/Specific Objective 2.xlsx"


dataset = pd.read_excel(path)


# Define X, Y

array = dataset.values

X = array[:, 1:12]

Y = array[:, 0]


# Validation Size

test_size = 0.20


# Train, Test Splitting for further Stratified Splitting

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)


# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
Spot Check Algorithm
"""

models = []

models.append(('LR', LogisticRegression(C=0.0001, penalty='l2')))


"""
Evaluate each model in turn
"""

results = []

names = []

for name, model in models:

    skf = StratifiedKFold(n_splits=10)

    cv_results = cross_val_score(model, X_train, Y_train, cv=skf, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s:%f(%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)


"""
Make predictions on test dataset using the logistic regression model
"""

lr = LogisticRegression()

lr.fit(X_train, Y_train)

predictions = lr.predict(X_test)

print(accuracy_score(Y_test, predictions))

print(confusion_matrix(Y_test, predictions))

print(classification_report(Y_test, predictions))



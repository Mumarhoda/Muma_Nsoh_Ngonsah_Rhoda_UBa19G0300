import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
sns.set(font_scale=1.5)

data_path = "C:/Wz Python Projects/Wonder_tk_MSc/DPES Rhoda Final/Specific Objective 2 - header.xlsx"
df = pd.read_excel(data_path)


X = df[['Q6', 'Q9']]
y = df['Gender']
# change the labels to numbers
y = pd.factorize(y, sort=True)[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, shuffle=True
)


X_array = np.asarray(X)
X_train_array = np.asarray(X_train)
X_test_array = np.asarray(X_test)


"""
"""


def add_labels(standardized=False):
    plt.title('Gender (Female[0], Male[1]) Dataset Visualized')

    plt.xlabel('Q6')
    plt.ylabel('Q9')
    plt.tight_layout()
    # plt.show()


y_str = y.astype(np.str)
y_str[y_str == '0'] = 'red'
y_str[y_str == '1'] = 'blue'
y_str[y_str == '2'] = 'green'


plt.scatter(X['Q6'], X['Q9'], c="blue")   # use y_str
plt.xlim(0, 7.9)
plt.ylim(-0.9, 3.5)
add_labels()


scale = StandardScaler()
scale.fit(X_train)
X_std = scale.transform(X)
X_train_std = scale.transform(X_train)
X_test_std = scale.transform(X_test)


lgr = LogisticRegression(solver="lbfgs", multi_class="auto")
lgr.fit(X_train_std, y_train)
print(
    "{:.1%} of the test set was correct.".format(
        metrics.accuracy_score(y_test, lgr.predict(X_test_std))
    )
)


plot_decision_regions(
    X_std, y, clf=lgr, X_highlight=X_test_std, colors='red,blue,green'
)
add_labels(standardized=True)
plt.show()

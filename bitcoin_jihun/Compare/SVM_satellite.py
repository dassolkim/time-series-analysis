from scipy.io import loadmat
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
import timeit

data = loadmat("C:/Users/tony9/OneDrive/문서/KNU/DKE_assignment/Time series/time-series-analysis/bitcoin_jihun/Compare/satellite.mat")
df = {k:v for k, v in data.items() if k[0] != '_'}

X = np.array(df['X'])
y = np.array(df['y']).reshape(-1,)
# data = np.concatenate((X,y), axis=1)
print(X)
print(y)
C_range =  np.logspace(-2,5,8)
gamma_range = np.logspace(-4, 3, 8)
# param_grid = dict(gamma=gamma_range, C=C_range)

# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid, cv=cv)

# grid.fit(X, y)

# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

C = []
temp = [0, 0, 0]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

for this_gamma in C_range:
    for this_C in gamma_range:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("linear_svc", SVC(kernel='rbf', C=this_C, gamma=this_gamma))
            ])
        start = timeit.default_timer()
        clf.fit(X_train,y_train)
        scoretrain = clf.score(X_train,y_train)
        scoretest  = clf.score(X_test,y_test)
        stop = timeit.default_timer()
        # C.append(scoretest)
        # gamma.append(scoretest)
        print("SVM for Non Linear \n Gamma: {} C:{} Training Score : {:2f} Test Score : {:2f}\n".format(this_gamma,this_C,scoretrain,scoretest))
        print(stop - start)
        if scoretest > temp[-1]:
            temp = [this_gamma, this_C, scoretest]
        

# C = C_range[C.index(max(C))]
# gamma = gamma_range[gamma.index(max(gamma))]
print("C: {} gamma: {}".format(temp[0], temp[1]))

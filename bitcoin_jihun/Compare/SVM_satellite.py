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

data = loadmat("C:/Users/tony9/OneDrive/문서/KNU/DKE_assignment/Time series/time-series-analysis/bitcoin_jihun/Compare/satellite.mat")
df = {k:v for k, v in data.items() if k[0] != '_'}

X = np.array(df['X'])
y = np.array(df['y'])
data = np.concatenate((X,y), axis=1)
print(X)
print(y)
y=y.reshape(-1)
C_range =  np.logspace(-2,10,13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict( gamma=gamma_range, C=C_range)
param_grid_ = dict(C=C_range)

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid_ = GridSearchCV(SVC(kernel="linear"),param_grid=param_grid_, cv=cv)

grid_.fit(X, y)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f" % (grid_.best_params_, grid_.best_score_))
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

C = []
gamma = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

for this_gamma in [.01,.5,.10,.25,.50,1]:
    for this_C in [1,5,7,10,15,25,50]:
        clf = SVC(kernel='rbf',C=this_C,gamma=this_gamma).fit(X_train,y_train)
        scoretrain = clf.score(X_train,y_train)
        scoretest  = clf.score(X_test,y_test)
        C.append(scoretest)
        gamma.append(scoretest)
        print("SVM for Non Linear \n Gamma: {} C:{} Training Score : {:2f} Test Score : {:2f}\n".format(this_gamma,this_C,scoretrain,scoretest))
        
C = [1,5,7,10,15,25,50][C.index(max(C))]
gamma = [.01,.5,.10,.25,.50,1][gamma.index(max(gamma))]
print("C: {} gamma: {}".format(C, gamma))
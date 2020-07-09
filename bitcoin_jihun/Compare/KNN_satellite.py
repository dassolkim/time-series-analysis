import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
import timeit

data = loadmat("C:/Users/tony9/OneDrive/문서/KNU/DKE_assignment/Time series/time-series-analysis/bitcoin_jihun/Compare/satellite.mat")
df = {k:v for k, v in data.items() if k[0] != '_'}
n_neighbors = 15
h = 0.02
X = np.array(df['X'])
y = np.array(df['y'])
data = np.concatenate((X, y), axis=1)
print(X)
print(y)
print(data)

X_train, X_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.2, random_state=42)


for weights in ['uniform', 'distance']:
    Scores = []
    t = [0,0,0]
    for n_neighbors in range(1,16):
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        start = timeit.default_timer()
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        stop = timeit.default_timer()
        Scores.append([n_neighbors, weights, score])
        print("{} neighbors, {} weights, {} TestScore".format(n_neighbors, weights, score))
        print(stop-start)
        if t[-1] < score:
            t = [n_neighbors, weights, score]
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
    print("\n{}\n".format(t))
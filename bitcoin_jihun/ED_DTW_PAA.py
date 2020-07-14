import numpy as np
import pandas as pd
import math
from fastdtw import fastdtw
from dtw import dtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PAA.PAA_dev import *
from PAA.PAA_review import *

# #####################################################################################################
# arraysA = []                                                                                        #
# arraysB = []                                                                                        #
# for line in open('Datasets/lightCurveA.txt'):                                                       #
#     arraysA.append(np.array([float(val) for val in line.rstrip('\n').split(' ') if val != '']))     #
# for line in open('Datasets/lightCurveB.txt'):                                                       #
#     arraysB.append(np.array([float(val) for val in line.rstrip('\n').split(' ') if val != '']))     #
# arraysA = np.array(arraysA).reshape(-1)                                                             #
# arraysB = np.array(arraysB).reshape(-1)                                                             #
# #####################################################################################################
data_path = "bitcoin_jihun/DTW/dataset/"
file_list = os.listdir(data_path)
file_list_csv = [file for file in file_list if file.endswith(".csv")]

# print(file_list_csv)
arraysA = pd.read_csv("bitcoin_jihun/DTW/dataset/{}".format(file_list_csv[6]))
arraysB = pd.read_csv("bitcoin_jihun/DTW/dataset/{}".format('HymanMinsky.csv'))

arraysA['Date'] = arraysA['Date'].map(pd.to_datetime)
arraysB.set_index(arraysB['x'])

arraysA = np.array(arraysA['Close']).reshape(-1)
arraysB = np.array(arraysB['y']).reshape(-1)

arraysA_ = np.array(paa(arraysA, 64)).reshape(-1)
# arraysB_ = np.array(PAA(arraysB, 381)).reshape(-1)
# arraysA_ = arraysA
arraysB_ = arraysB

distance, cost_matrix, acc_cost_matrix, path = dtw(arraysA, arraysB, dist=euclidean)
distance_, path_ = fastdtw(arraysA_, arraysB_, dist=euclidean)

# distance = np.float(distance)
print("distance: ",distance)
print("PAA_distance: ", distance_)

fig, ax = plt.subplots(2,1,figsize=(10, 10))

ax[0].plot(arraysA,color='r',label='arraysA',alpha=0.75)
ax[1].plot(arraysA_,color='r',label='arraysA_',alpha=0.75)
ax[0].plot(arraysB,color='b',label='arraysB',alpha=0.75)
ax[1].plot(arraysB_,color='b',label='arraysB_',alpha=0.75)


# arraysA = pd.DataFrame(data=arraysA)
# arraysB = pd.DataFrame(data=arraysB)
# arraysA_ = pd.DataFrame(data=arraysA_)
# arraysB_ = pd.DataFrame(data=arraysB_)

# ax[0].plot([arraysA.iloc[v] for v in [p[0] for p in path]], color='b', label='arraysA', alpha=0.75)
# ax[1].plot([arraysA_.iloc[v] for v in [p[0] for p in path_]], color='g', label='arraysA', alpha=0.75)
# ax[0].plot([arraysB.iloc[v] for v in [p[1] for p in path]], color='r', label='arraysB', alpha=0.75)
# ax[1].plot([arraysB_.iloc[v] for v in [p[1] for p in path_]], color='y', label='arraysB', alpha=0.75)

ax[0].legend()
ax[1].legend()
ax[0].set_title("A vs. B {}| distance: {}".format(file_list_csv[10], round(distance, 3)), fontsize=15)
ax[0].set_xlabel("time steps")
ax[0].set_ylabel("normalized price")
ax[1].set_title("A vs. B {}| distance: {}".format(file_list_csv[10],round(distance_, 3)), fontsize=15)
ax[1].set_xlabel("time steps")
ax[1].set_ylabel("normalized price")

print("Corrcoef: \n",np.corrcoef(arraysA_, arraysB_[:64]))
arraysB_dtw = [arraysB[p] for p in path[1]]
print("Corrcoef: \n",np.corrcoef(arraysA, arraysB_dtw[:380]))


plt.show()
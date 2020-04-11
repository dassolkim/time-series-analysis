import numpy as np
import pandas as pd
import math
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PAA.PAA_dev import *

# def paa(arr, sections):
#     try:
#         assert arr.shape[0] != sections
#     except AssertionError as e:
#         return np.copy(arr)
#     else:
#         if arr.shape[0] % sections == 0:
#             sectionarr = np.array_split(arr, sections)
#             res = np.array([item.mean() for item in sectionarr])
#             res = np.array(
#                 [sample for item in res for sample in 
#                 [item.mean()] * item.shape[0]]
#             )
#         else:
#             sectionarr = np.zeros(sections)
#             space_size = np.arange(0, arr.shape[0] * sections - 1)
#             outputIndex = space_size // arr.shape[0]
#             inputIndex = space_size // sections
#             uniques, nUniques = np.unique(outputIndex, return_counts=True)
            
#             res = [arr[indices].sum() / arr.shape[0] for indices in
#                    np.split(inputIndex, nUniques.cumsum())[:-1]]
#             indices = ([row.mean() for row in np.split(inputIndex, nUniques.cumsum())[:-1]])
#     return res
def PAA(data, segment):
    
    len_data = len(data)
    div_len_data = len(data)/segment

    data=np.array(data)

    if len_data % segment == 0:
        sectionedArr = np.array_split(data, segment)
        res = np.array([item.mean() for item in sectionedArr])

    else:
        value_space= np.arange(0, len_data * segment)   #[0,...,길이*분할 수]
        output_index = value_space // len_data          #[0,...,길이*분할 수]/길이
        input_index = value_space // segment            #[0,...,길이*분할 수]/분할 수
        uniques, nUniques = np.unique(output_index, return_counts=True)#배열의 고유한 요소를 찾는다. uniques: 고유한 값을 제공하는 입력 배열의 인덱스, nUniques: 입력 배열에 각 고유값이 나타나는 횟수
        # print(uniques,"\n",nUniques)
        # print(input_index)
        # print(output_index)
        # input
        res = [data[indices].sum() / data.shape[0] for indices in np.split(input_index, nUniques.cumsum())[:-1]]
        # print(np.split(input_index, nUniques.cumsum())[:-1])
    #print(res)
    return res
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
arraysA = pd.read_csv("bitcoin_jihun/DTW/dataset/{}".format(file_list_csv[10]))
arraysB = pd.read_csv("bitcoin_jihun/DTW/dataset/{}".format('HymanMinsky.csv'))

arraysA['Date'] = arraysA['Date'].map(pd.to_datetime)
arraysB.set_index(arraysB['x'])

arraysA = np.array(arraysA['Close']).reshape(-1)
arraysB = np.array(arraysB['y']).reshape(-1)

arraysA_ = np.array(paa(arraysA, 100)).reshape(-1)
# arraysB_ = np.array(PAA(arraysB, 381)).reshape(-1)
# arraysA_ = arraysA
arraysB_ = arraysB

distance, path = fastdtw(arraysA, arraysB, dist=euclidean)
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

plt.show()
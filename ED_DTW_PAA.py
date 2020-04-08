import numpy as np
import pandas as pd
import math
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
def PAA(data, segment):
    
    len_data = len(data)
    div_len_data = len(data)/segment

    data=np.array(data)

    if len_data % segment == 0:
        sectionedArr = np.array_split(data, segment)
        res = np.array([item.mean() for item in sectionedArr])

    else:
        value_space= np.arange(0, len_data * segment)#[0,...,길이*분할 수]
        output_index = value_space // len_data#[0,...,길이*분할 수]/길이
        input_index = value_space // segment#[0,...,길이*분할 수]/분할 수
        uniques, nUniques = np.unique(output_index, return_counts=True)#배열의 고유한 요소를 찾는다. uniques: 고유한 값을 제공하는 입력 배열의 인덱스, nUniques: 입력 배열에 각 고유값이 나타나는 횟수
        # print(uniques,"\n",nUniques)
        # print(input_index)
        # print(output_index)
        # input
        res = [data[indices].sum() / data.shape[0] for indices in np.split(input_index, nUniques.cumsum())[:-1]]
        # print(np.split(input_index, nUniques.cumsum())[:-1])
        print(res)
arraysA = []
arraysB = []
for line in open('Datasets/lightCurveA.txt'):
    arraysA.append(np.array([float(val) for val in line.rstrip('\n').split(' ') if val != '']))
for line in open('Datasets/lightCurveB.txt'):
    arraysB.append(np.array([float(val) for val in line.rstrip('\n').split(' ') if val != '']))
arraysA = np.array(arraysA).reshape(-1)
arraysB = np.array(arraysB).reshape(-1)

distance, path = fastdtw(arraysA, arraysB, dist=euclidean)

# arraysA = np.array(PAA(arraysA, 30)).reshape(-1)
# arraysB = np.array(PAA(arraysB, 30)).reshape(-1)
# distance, path = fastdtw(arraysA, arraysB, dist=euclidean)
# distance = np.float(distance)
print("distance: ",distance)
# print("PAA_distance: ", distance)

fig, ax = plt.subplots(figsize=(20, 10))

arraysA_ = pd.DataFrame(data=arraysA)
arraysB_ = pd.DataFrame(data=arraysB)

# ax.plot([arraysA.iloc[v] for v in [p[0] for p in path]], color='b', label='arraysA', alpha=0.75)
ax.plot([arraysA_.iloc[v] for v in [p[0] for p in path]], color='g', label='arraysA', alpha=0.75)
# ax.plot([arraysB.iloc[v] for v in [p[1] for p in path]], color='r', label='arraysB', alpha=0.75)
ax.plot([arraysB_.iloc[v] for v in [p[1] for p in path]], color='y', label='arraysB', alpha=0.75)

ax.legend()
ax.set_title("A vs. B | distance: {}".format(round(distance, 3)), fontsize=15)
ax.set_xlabel("time steps")
ax.set_ylabel("normalized price")

plt.show()
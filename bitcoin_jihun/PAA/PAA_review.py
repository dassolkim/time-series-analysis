# #PAA(Piecewise Aggregation Approximation): 부분 집계 근사법
# 저차원 변환 방법의 하나로서, 고차원 시퀀스를 여러 구간으로 나누고,
# 각 구간의 평균을 해당 시퀀스의 특성(feature) 값으로 사용한다.
# 장점:   기존 저차원 변환 방법에 비해 계산 과정이 매우 간단하고 성능이 우수하다.
#         임의의 길이에 대한 질의도 처리 가능하다.
# 단점:   이상치탐지에 부적합하다.
import numpy as np

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
        

# data = np.random.random_integers(low=0, high= 15, size=15)
# segment = 4
# PAA(data, segment)
# data = [1,2,3,4,5,6,7,8,9,10,10,10,10,10,10]
# segment = 4
# PAA(data, segment)
arrays = []
for line in open('Datasets/lightCurveA.txt'):
    arrays.append(np.array([float(val) for val in line.rstrip('\n').split(' ') if val != '']))
arrays = np.array(arrays).reshape(1,-1)
print(len(arrays))
print(arrays.shape)
segment = 36
PAA(arrays, segment)
import math
import numpy as np
import pandas as pd

def PAA(data, seg):
    data = np.array(data)
    len_data = len(data)
    len_seg = len(data)/seg
    tmp = 0
    
    
    avg = [0 for i in range(0, seg)]
    for i in range(0, seg):
        count = 0
        get_sum = 0
        for j in range(math.floor(tmp), math.floor(tmp+len_seg)):
            get_sum += data[j]
            count+=1
        # print("get_sum: ",get_sum,"\n")
        # print("count: ",count,"\n")
        if(count!=0):
            avg[i] = get_sum/count
        tmp = tmp+len_seg

    print("avg: ",avg,"\n")
    
arrays = []
for line in open('Datasets/lightCurveA.txt'):
    arrays.append(np.array([float(val) for val in line.rstrip('\n').split(' ') if val != '']))
arrays = np.array(arrays).reshape(-1)
print(len(arrays))
# print(arrays.shape)
segment = 36
PAA(arrays, segment)
import math
import numpy as np

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
        print(get_sum)
        print(count)
        avg[i] = get_sum/count
        tmp = tmp+len_seg

    print(avg)
    
data = [1,2,3,4,5,6,7,8,9,10,10,10,10,10,10]
segment = 4
PAA(data, segment)
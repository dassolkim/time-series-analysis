import numpy as np

def paa(arr, sections):
    try:
        assert arr.shape[0] != sections
    except AssertionError as e:
        return np.copy(arr)
    else:
        if arr.shape[0] % sections == 0:
            sectionarr = np.array_split(arr, sections)
            res = np.array([item.mean() for item in sectionarr])
            # res = np.array(
            #     (sample for item in res for sample in 
            #     [item.mean()] * item.shape[0])
            # )
            # 동작 잘 안함..
        else:
            sectionarr = np.zeros(sections) # section 만큼의 0으로 초기화된 배열 생성
            space_size = np.arange(0, arr.shape[0] * sections - 1) # 0에서 arr의 개수에 section의 곱에 1을 뺀만큼 나열
            print("space_size: ",space_size)
            outputIndex = space_size // arr.shape[0]
            print("outputIndex: ",outputIndex) 
            inputIndex = space_size // sections
            print("inputIndex: ",inputIndex)
            uniques, nUniques = np.unique(outputIndex, return_counts=True)
            print("uniques: ",uniques,"\nnUniques: ",nUniques)
            
            res = [arr[indices].sum() / arr.shape[0] for indices in
                   np.split(inputIndex, nUniques.cumsum())[:-1]]
            indices = ([row.mean() for row in np.split(inputIndex, nUniques.cumsum())[:-1]]) # 왜 존재하는지 이해 불가
            print("indices: ",indices)
    return res

# arrays = []
# for line in open('bitcoin_jihun/Datasets/lightCurveA.txt'):
#     arrays.append(np.array([float(val) for val in line.rstrip('\n').split(' ') if val != '']))
# arrays = np.array(arrays).reshape(-1)
# print(len(arrays))
# # print(arrays.shape)
# segment = 36
# print(paa(arrays, segment))
# t1 = np.random.random(size=15)
# print(paa(t1, 4))
data = np.array([1,2,3,4,5,6,7,8,9,10,10,10,10,10,10])
segment = 4
print(paa(data, segment))
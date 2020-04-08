import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw

ts1 = [1, 5, 3, 4, 7, 6]
ts2 = [0, 2, 6, 3, 5, 6, 8, 5]

x = np.array(ts1).reshape(-1,1)
y = np.array(ts2).reshape(-1,1)

euclidean_norm = lambda x, y: np.abs(x-y)

d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)

plt.imshow(acc_cost_matrix.T, origin='lower',cmap='gray',interpolation='nearest')
plt.plot(path[0],path[1],'w')
plt.show()

ts1_dtw = [ts1[p] for p in path[0]]

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.title('Time series 1')
plt.plot(ts1)
plt.grid(True)
plt.subplot(122)
plt.title('Comparison : ts1_dtw vs. ts2')
plt.plot(ts1_dtw)
plt.plot(ts2)
plt.legend(['Time series 1 - Warping', 'Time series 2'])
plt.grid(True)
plt.show()

print(np.corrcoef(ts1_dtw, ts2))
#### output
# array([[ 1.        ,  0.92247328],
#        [ 0.92247328,  1.        ]])
print(d,"\n",cost_matrix,"\n",acc_cost_matrix,"\n",path)
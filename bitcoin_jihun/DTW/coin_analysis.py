import coin_crawler
import os
import numpy as np
import pandas as pd
from fastdtw import fastdtw
#import _ucrdtw
# from dtw import dtw
import glob
import re
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

bitcoin_path = 'dataset/bitcoin.csv'
minsky_path = 'HymanMinsky.csv'

bitcoin_df = pd.read_csv(bitcoin_path)
bitcoin_df['Date'] = bitcoin_df['Date'].map(pd.to_datetime)
minsky_df = pd.read_csv(minsky_path)

# euclidean_norm = lambda x, y: np.abs(x, y)

bitcoin_df = np.array(bitcoin_df['Close']).reshape(-1,1)
minsky_df = np.array(minsky_df['y']).reshape(-1,1)

mns = MinMaxScaler()

bitcoin_df = mns.fit_transform(bitcoin_df)
minsky_df = mns.fit_transform(minsky_df)
print(minsky_df)
# d, cost_matrix, acc_cost_matrix, path = dtw(minsky_df.adj_close, bitcoin_df.adj_y, dist=euclidean_norm)
distance, path = fastdtw(bitcoin_df, minsky_df, dist=euclidean)

# plt.imshow(acc_cost_matrix.T, origin='lower', cmap='coolwarm', interpolation='nearest')
# plt.plot(path[0], path[1], 'w')
# plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
bitcoin_df = pd.DataFrame(data=bitcoin_df)
minsky_df = pd.DataFrame(data=minsky_df)

ax.plot([bitcoin_df.iloc[v] for v in [p[0] for p in path]], color='b', label='BitCoin', alpha=0.75)

ax.plot([minsky_df.iloc[v] for v in [p[1] for p in path]], color='r', label='HymanMinsky Model', alpha=0.75)

ax.legend()
distance = np.float(distance)
ax.set_title("BitCoin vs. Hyman Minsky Model | distance: {}".format(round(distance, 3)), fontsize=15)
ax.set_xlabel("time steps")
ax.set_ylabel("normalized price")

plt.show()

# print(minsky_df)
# minsky_dtw = [minsky_df[p] for p in path[0]]

# plt.figure(figsize=(15,5))
# plt.subplot(121)
# plt.title('Minsky')
# plt.plot(minsky_df)
# plt.grid(True)
# plt.subplot(122)
# plt.title('Minsky vs bitcoin')
# plt.plot(minsky_dtw)
# plt.plot(bitcoin_df)
# plt.legend(['minsky','bitcoin'])
# plt.grid(True)
# plt.show()

# np.corrcoef(minsky_dtw, bitcoin_df)
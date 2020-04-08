import os
import numpy as np
import pandas as pd
from fastdtw import fastdtw
# from ucrdtw import _ucrdtw
from functools import reduce
from input_path import *
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt

a_, b_ = input_path('bitcoin')

bitcoin_path = a_
minsky_path = b_

bitcoin_df = pd.read_csv(bitcoin_path)
bitcoin_df['Date'] = bitcoin_df['Date'].map(pd.to_datetime)
minsky_df = pd.read_csv(minsky_path)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(bitcoin_df.Close)
ax[1].plot(minsky_df.y)

ax[0].set_title("BitCoin price", fontsize=15)
ax[1].set_title("Hyman Minsky bubble chart", fontsize=15)

plt.tight_layout()
plt.show()

################################################
#
# Using DTW
#
################################################

mns = MinMaxScaler()
# 0 ~ 1 fitting

bit_close = []
bit_close = bitcoin_df.Close
bit_close_arrary = bit_close.values.reshape(-1, 1)
minsky_y = []
minsky_y = minsky_df.y
minsly_y_array = minsky_y.values.reshape(-1, 1)
bitcoin_df['adj_close'] = mns.fit_transform(bit_close_arrary)
minsky_df['adj_y'] = mns.fit_transform(minsly_y_array)

distance, path = fastdtw(bitcoin_df.adj_close, minsky_df.adj_y, dist=euclidean)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot([bitcoin_df.adj_close.iloc[v] for v in [p[0] for p in path]], color='b', label='BitCoin', alpha=0.75)
ax.plot([minsky_df.adj_y[v] for v in [p[1] for p in path]], color='r', label='HymanMinsky Model', alpha=0.75)

ax.legend()
ax.set_title("BitCoin vs. Hyman Minsky Mocel | distance: {}".format(round(distance, 3)), fontsize=15)
ax.set_xlabel("time steps")
ax.set_ylabel("normalized price")

plt.show()

# check correlation
bitcoin_dtw_path = [p[0] for p in path]

minsky_dtw_path = [p[1] for p in path]

plt.plot(bitcoin_dtw_path, minsky_dtw_path)
plt.xlabel("BitCoin path")
plt.ylabel("Minsky_path")
plt.show()
import numpy as np
import pandas as pd
from input_path import *
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

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
# Using linear interpolation
#
################################################

hm_datapoints = len(minsky_df.y)
bc_datapoints = len(bitcoin_df.Close)

every = bc_datapoints // hm_datapoints
stretched = np.zeros(bc_datapoints)
for hd in range(hm_datapoints):
    stretched[hd * every] = minsky_df.y.iloc[hd]
stretched[-1] = minsky_df.y.iloc[-1]
stretched[np.where(stretched == 0)] = np.nan

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

nans, x = nan_helper(stretched)
stretched[nans] = np.interp(x(nans), x(~nans), stretched[~nans])

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(bitcoin_df.Close)
ax[1].plot(stretched)

ax[0].set_title("BitCoin price", fontsize=15)
ax[1].set_title("Hyman Minsky bubble chart(linear interpolation", fontsize=15)

plt.tight_layout()
plt.show()

# print('this is correlation')
# print(np.corrcoef([bitcoin_df.Close, stretched]))
# print('this is scipy euclidean distance')
# print(euclidean(bitcoin_df.Close, stretched))
# print('this is numpy l2 norm')
# print(np.linalg.norm(bitcoin_df.Close-stretched))
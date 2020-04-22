from dateutil.parser import parse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from math import *
import time
np.random.seed(np.random.randint(10))

# dfA=np.random.randn(128)
# dfB=np.random.randn(128)

dfA = np.arange(0,20)
dfB = np.arange(4,24)
# def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100, color='red'):
    # plt.figure(figsize=(16,5), dpi=dpi)
    # plt.plot(x, y, color=color)
    # plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
def euclidean_distance(x,y):
    return sqrt(sum(pow(a-b,2)for a, b in zip(x,y)))
print(euclidean_distance(dfA,dfB))
# plot_df(dfA, x=range(len(dfA)), y=dfA, title='dfA')
# plot_df(dfB,x=range(len(dfB)), y=dfB, title='dfB', color='blue')
plt.figure(figsize=(16,5),dpi=100)
plt.plot(range(len(dfA)),dfA, color='red')
plt.plot(range(len(dfB)),dfB, color='blue')
plt.title('Euclidean similarity')
plt.legend(['dfA','dfB'])
plt.show()

#range
#scipy.euclidean_distance
#
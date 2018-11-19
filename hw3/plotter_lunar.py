import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import pickle
import numpy as np

rews = []
for i in range(2, 11, 2):
    with open("lander-layers-{}.pkl".format(i), 'rb') as f:
        rews.append(pickle.load(f))

min_len = min([len(r) for r in rews])
rews = [r[0:min_len] for r in rews]
smoothed_rews = [[np.mean(r[i - 100: i]) for i in range(100, len(r))] for r in rews]
max_rews = [np.maximum.accumulate(r) for r in smoothed_rews]
# smoothed_vanilla = [np.mean(vanilla_rews[i - 100: i]) for i in range(100, len(vanilla_rews))]
# smoothed_double = [np.mean(double_rews[i - 100: i]) for i in range(100, len(double_rews))]
# max_vanilla = np.maximum.accumulate(smoothed_vanilla)
# max_double = np.maximum.accumulate(smoothed_double)
sns.set_style("darkgrid")
for i in range(5):
    plt.plot(smoothed_rews[i], label="mean-{}".format(2*(i+1)))

plt.ylabel("reward")
plt.xlabel("episodes")
plt.legend()
plt.show()

for i in range(5):
    plt.plot(max_rews[i], label="max-{}".format(2*(i+1)))

plt.ylabel("reward")
plt.xlabel("episodes")
plt.legend()
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import pickle
import numpy as np

with open("39d8960d-e867-446b-a604-aa0a0b0efcd8.pkl", 'rb') as f:
    vanilla_rews = pickle.load(f)[:2400]
with open("391035c3-01bb-47b1-a314-fa85c449becf.pkl", 'rb') as f:
    double_rews = pickle.load(f)[:2400]

# def plot_data(data, value="AverageReturn"):
#     if isinstance(data, list):
#         data = pd.concat(data, ignore_index=True)

#     sns.set(style="darkgrid", font_scale=1.5)
#     sns.tsplot(data=data, time="Iteration", value=value, unit="Unit", condition="Condition")
#     plt.legend(loc='best').draggable()
#     plt.show()

# plot_data(vanilla_rews)
smoothed_vanilla = [np.mean(vanilla_rews[i - 100: i]) for i in range(100, len(vanilla_rews))]
smoothed_double = [np.mean(double_rews[i - 100: i]) for i in range(100, len(double_rews))]
max_vanilla = np.maximum.accumulate(smoothed_vanilla)
max_double = np.maximum.accumulate(smoothed_double)

sns.set_style("darkgrid")
plt.plot(smoothed_vanilla, label='vanilla')
plt.plot(smoothed_double, label='double')
plt.plot(max_vanilla, label='max vanilla')
plt.plot(max_double, label='max double')
plt.ylabel("reward")
plt.xlabel("episodes")
plt.legend()
plt.show()

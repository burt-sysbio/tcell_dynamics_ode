#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:28:49 2020

@author: burt
"""
import numpy as np
from scipy.stats import gamma
from matplotlib import pyplot as plt

import seaborn as sns
sns.set(style = "ticks", context = "poster")

# plot the distributions
k_values = [10, 2, 1]
theta_values = [1/10, 0.5, 1]
x = np.linspace(0, 3, 1000)

# plot the distributions
fig, ax = plt.subplots(figsize=(5, 4))

linestyles = ["-", "--", ":"]
labels = [r"SD $\Psi=0.1$", r"SD $\Psi=0.5$", r"SD $\Psi=1.0$"]

for k, t, ls, label in zip(k_values, theta_values, linestyles, labels):
    dist = gamma(k, 0, t)
    plt.plot(x, dist.pdf(x), ls=ls, c='black',
             label=label)

ax.set_xlim(0, x[-1])
ax.set_ylim(0, 1.5)
ax.set_yticks([0,0.5,1,1.5])

ax.set_xlabel('time')
ax.set_ylabel("density")

ax.legend(loc=0)
plt.tight_layout()
plt.show()

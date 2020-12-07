#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:28:49 2020

@author: burt
"""

import numpy as np
from scipy.stats import gamma
from matplotlib import pyplot as plt
from scipy.stats import lognorm
import seaborn as sns
sns.set(style = "ticks", context = "talk", rc = {"lines.linewidth" : 4})
#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.

#------------------------------------------------------------
# plot the distributions
k_values = [10, 2, 1]
theta_values = [1/10, 0.5, 1]
x = np.linspace(0, 3, 1000)


#------------------------------------------------------------
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
#fig.savefig("gamma_dist.svg")
#fig.savefig("gamma_dist.pdf")

# =============================================================================
# plot lognorm instead 
# =============================================================================
x = np.linspace(2,20,200)

loc = 0

def lognorm_params(mode, stddev):
    """
    Given the mode and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    p = np.poly1d([1, -1, 0, 0, -(stddev/mode)**2])
    r = p.roots
    sol = r[(r.imag == 0) & (r.real > 0)].real
    shape = np.sqrt(np.log(sol))
    scale = mode * sol
    return shape, scale

shape1, scale1 = lognorm_params(10, 1)
shape2, scale2 = lognorm_params(10, 3)


fig, ax = plt.subplots()
ax.plot(x, lognorm.pdf(x, s = shape1, scale = scale1, loc = loc), "k", label = "low heterogeneity")
ax.plot(x, lognorm.pdf(x, s = shape2,  scale = scale2, loc = loc), "k", ls = "--", label = "high heterogeneity")
ax.legend()
ax.set_xlabel(r"$\beta_p$")
ax.set_ylabel("density")
fig.savefig("lognorm_distribution.svg")
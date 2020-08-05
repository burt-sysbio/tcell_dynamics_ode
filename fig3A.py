#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:49:43 2020

@author: burt

create figure with 2 pos feedback and different init probs 
then vary feedback strength
do this for alpha = 1 and alpha = 10
"""
from prec_model import Simulation, prec_model

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(context = "poster", style = "ticks")

celltypes = ["naive", "prec", "th1", "tfh"]


d = {
     "alpha_naive" : 10,
     "alpha_prec" : 10,
     "alpha_th1" : 10,
     "alpha_tfh": 10,
     "beta_naive" : 10,
     "beta_prec" : 10,
     "beta_p_th1" : 20,
     "beta_p_tfh" : 20,
     "n_div_eff" : 1,
     "n_div_prec" : 1.,
     "death_th1" : 2,
     "death_tfh" : 2,
     "p_th1" : 0.45,
     "p_tfh" : 0.35,
     "p_prec" : 0.2,
     "deg_myc" : 0.1,
     "EC50_myc" : 0.5,
     "fb_ifng_prob_th1" : 1,
     "fb_il10_prob_th1" : 1.0,
     "fb_il21_prob_tfh" : 1,
     "EC50_ifng_prob_th1" : 0.2,
     "EC50_il10_prob_th1" : 0.2,
     "EC50_il21_prob_tfh" : 0.2,
     "r_ifng" : 1,
     "r_il21" : 1,
     "r_il10" : 1,
     "deg_ifng" : 1,
     "deg_il21" : 1,
     "deg_il10" : 1,
     }

d_rate = dict(d)
params = ["alpha_naive", "alpha_prec", "alpha_th1", "alpha_tfh", "beta_naive",
          "beta_prec"]

for p in params:
    d_rate[p] = 1
d_rate["beta_p_th1"] = d["beta_p_th1"] / 10
d_rate["beta_p_tfh"] = d["beta_p_tfh"] / 10

time = np.arange(0,20,0.01)
sim_rate = Simulation("alpha1", prec_model, d_rate, celltypes, time)
sim_rtm = Simulation("alpha10", prec_model, d, celltypes, time)

sim_rate.run_timecourse()
sim_rtm.run_timecourse()

df = pd.concat([sim_rate.state_tidy, sim_rtm.state_tidy])
g = sns.relplot(data = df, x = "time", y = "cells", hue = "cell_type", kind = "line",
                row = "sim_name")

arr_dict = {"fb_ifng_prob_th1" : np.geomspace(1,10,50), 
            "fb_il21_prob_th1" : np.geomspace(1,10,50)}

# vary parameters
df = sim_rate.vary_param(arr_dict)
df2 = sim_rtm.vary_param(arr_dict)

# get relative readouts
df3 = sim_rate.get_relative_readouts(df)
df4 = sim_rtm.get_relative_readouts(df2)

# normalize
df5 = sim_rate.normalize_readout_df(df3, norm_idx = 0)
df6 = sim_rtm.normalize_readout_df(df4, norm_idx = 0)

# combine
df7 = pd.concat([df5, df6])
df7 = df7.melt(id_vars = ["param_val", "sim_name", "param_name"], value_name = "effect size", var_name = "readout")

# plot
g = sns.relplot(data = df7, x = "param_val", y = "effect size", col = "sim_name", 
                hue = "readout", kind = "line")
g.set(xscale = "log", xlabel = "feedback fold-change")

g = sns.relplot(data = df7, x = "param_val", y = "effect size", col = "readout", 
                hue = "sim_name", kind = "line", col_wrap=(2))
g.set(xscale = "log", xlabel = "feedback fold-change")


# plot time course and relative cells with feedback variation
df8 = sim_rtm.run_timecourses(arr_dict)
sim_rtm.plot_timecourses(df8, log = True, cbar_label = "feedback fold-change")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:49:43 2020

@author: burt

effectors proliferate
create figure with 2 pos feedback and different init probs 
then vary feedback strength
do this for alpha = 1 and alpha = 10
"""
from prec_model import Simulation, prec_model

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context = "poster", style = "ticks")

celltypes = ["Naive", "Prec", "Th1", "Tfh"]

def filter_cells(cells, names):
    cells = cells[cells.cell_type.isin(names)]
    return cells

d = {
     "alpha_naive" : 10,
     "alpha_prec" : 10,
     "alpha_th1" : 10,
     "alpha_tfh": 10,
     "beta_naive" : 10,
     "beta_prec" : 10,
     "beta_p_th1" : 10,
     "beta_p_tfh" : 10,
     "beta_m_th1" : 0,
     "beta_m_tfh" : 0,
     "n_div_eff" : 1,
     "n_div_prec" : 2.5,
     "death_th1" : 2.1,
     "death_tfh" : 2.1,
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

# set parameters specific to this condition
d["p_prec"] = 0
d["n_div_prec"] = 0
d["beta_p_th1"] = 10
d["beta_p_tfh"] = 10

# normalize probabililties since prec prob is not there and to compare to fig3a
norm = d["p_th1"] + d["p_tfh"]
d["p_th1"] = d["p_th1"] / norm
d["p_tfh"] = d["p_tfh"] / norm

d_rate = dict(d)
params = ["alpha_naive", "alpha_prec", "alpha_th1", "alpha_tfh", "beta_naive",
          "beta_prec"]

for p in params:
    d_rate[p] = 1
d_rate["beta_p_th1"] = d["beta_p_th1"] / 10
d_rate["beta_p_tfh"] = d["beta_p_tfh"] / 10

d_fb = dict(d)
d_fb["fb_il10_prob_th1"] = 10

time = np.arange(0,15,0.01)

# set up simulations and run timecourse
sim_rate = Simulation("alpha=1", prec_model, d_rate, celltypes, time)
sim_rtm = Simulation("alpha=10", prec_model, d, celltypes, time)
sim_fb = Simulation("fb on", prec_model, d_fb, celltypes, time)
sim_rate.run_timecourse()
sim_rtm.run_timecourse()
sim_fb.run_timecourse()

# compare feedbakc on off timecourse
df = pd.concat([sim_rtm.state_tidy, sim_fb.state_tidy])
df = filter_cells(df, names = ["Th1", "Tfh", "Prec", "Total"])
g = sns.relplot(
    data = df, x = "time", y = "cells", 
    hue = "cell_type", kind = "line",
    style = "sim_name", legend = False, aspect = 1.2)
plt.show()
#g.savefig("fig3B_1.svg")
arr_dict = {"fb_ifng_prob_th1" : np.geomspace(1,100,50), 
            "fb_il21_prob_th1" : np.geomspace(1,100,50)}

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
g.set_titles("{col_name}")
plt.show()
g.savefig("plot_fig3B_readouts.svg")

# plot time course and relative cells with feedback variation
df8 = sim_rtm.run_timecourses(arr_dict)
g = sim_rtm.plot_timecourses(df8, log = True, cbar_label = "feedback fold-change")
plt.show()
g.savefig("plot_fig3B_timecourse.pdf")

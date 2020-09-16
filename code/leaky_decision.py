#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:49:43 2020

@author: burt

compare fate decision for proliferating vs non proliferating cells
i.e. does it make a difference if differentiation happens in a context with fast proliferators vs slow/no proliferators
my hypothesis: in the context of proliferation control over the differentiation via different probabilities is leaky
should work better if nothing proliferates afterwards
"""
from prec_model import Simulation, prec_model

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context = "poster", style = "ticks")

celltypes = ["naive", "Prec", "Th1", "Tfh"]

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
     "beta_p_th1" : 100,
     "beta_p_tfh" :100,
     "beta_m_th1" : 0,
     "beta_m_tfh" : 0,
     "n_div_eff" : 1,
     "n_div_prec" : 0,
     "death_th1" : 2,
     "death_tfh" : 2,
     "p_th1" : 0.25,
     "p_tfh" : 0.55,
     "p_prec" : 0,
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

d2 = dict(d)
# set parameters specific to this condition
d2["p_prec"] = 0
d2["n_div_prec"] = 0
d2["beta_p_th1"] = 0
d2["beta_p_tfh"] = 0
#d2["death_th1"] = 0
#d2["death_tfh"] = 0

d_list = [d, d2]
modes = [r"$\beta_p > \beta_d$", r"$\beta_p < \beta_d$"]
df_list1 = []
df_list2 = []

time = np.arange(0,4,0.01)

# run pipeline for eff prolif and prec prolif models
for dic, mode in zip(d_list, modes):

    d_rate = dict(dic)
    params = ["alpha_naive", "alpha_prec", "alpha_th1", "alpha_tfh", "beta_naive",
              "beta_prec"]

    # adjust rate parameters
    for p in params:
        d_rate[p] = 1
    d_rate["beta_p_th1"] = d["beta_p_th1"] / 10
    d_rate["beta_p_tfh"] = d["beta_p_tfh"] / 10

    # set up simulations
    sim_rate = Simulation("alpha=1", prec_model, d_rate, celltypes, time)
    sim_rtm = Simulation("alpha=10", prec_model, dic, celltypes, time)
    sim_rate.run_timecourse()
    sim_rtm.run_timecourse()

    res = 50
    arr_dict = {"fb_ifng_prob_th1" : np.geomspace(1,100, res),
                "fb_il21_prob_tfh" : np.geomspace(1,100, res)}

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

    # run timecourse (only for rtm)
    df8 = sim_rtm.run_timecourses(arr_dict)

    df7["mode"] = mode
    df8["mode"] = mode

    df_list1.append(df7)
    df_list2.append(df8)

df1 = pd.concat(df_list1)
df2 = pd.concat(df_list2)
# plot
g = sns.relplot(data = df1, x = "param_val", y = "effect size", hue = "mode", col = "sim_name",
                row = "readout", kind = "line", facet_kws= {"margin_titles" : True})
g.set(xscale = "log")
g.set_titles(col_template = "{col_name}",
             row_template = "{row_name}")
plt.show()
g.set(xscale = "log", xlabel = "feedback fold-change")
plt.show()
#g.savefig("../figures/fig3/fig3A_readouts.pdf")
# plot time course and relative cells with feedback variation


g = sim_rtm.plot_timecourses(df2,
                             log = True,
                             cbar_label = "feedback fold-change",
                             ylabel = "cells X (% of total)",
                             col = "mode")
plt.show()
g.savefig("../figures/fig3/fig3A_leaky_timecourse.pdf")
g.savefig("../figures/fig3/fig3A_leaky_timecourse.svg")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:56:27 2020

@author: burt
"""
from prec_model import Simulation, prec_model

import numpy as np
import pandas as pd
import seaborn as sns

celltypes = ["Naive", "Prec", "Th1", "Tfh"]


d = {
     "alpha_naive" : 10,
     "alpha_prec" : 10,
     "alpha_th1" : 10,
     "alpha_tfh": 10,
     "beta_naive" : 10,
     "beta_prec" : 10,
     "beta_p_th1" : 0,
     "beta_p_tfh" : 0,
     "n_div_eff" : 1,
     "n_div_prec" : 1.,
     "death_th1" : 1,
     "death_tfh" : 1,
     "beta_m_th1" : 0,
     "beta_m_tfh" : 0,
     "p_th1" : 0.35,
     "p_tfh" : 0.45,
     "p_prec" : 0.2,
     "deg_myc" : 0.1,
     "EC50_myc" : 0.5,
     "fb_ifng_prob_th1" : 1.,
     "fb_il10_prob_th1" : 1.0,
     "fb_il21_prob_tfh" : 1.,
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
d2["p_prec"] = 0
d2["beta_p_th1"] = 10
d2["beta_p_tfh"] = 10
d2["p_th1"] = 0.44
d2["p_tfh"] = 0.56


d3 = dict(d)
d3["fb_ifng_prob_th1"] = 100
d3["fb_il21_prob_tfh"] = 100

d4 = dict(d2)
d4["fb_ifng_prob_th1"] = 100
d4["fb_il21_prob_tfh"] = 100


time = np.arange(0,12,0.01)

sim_names = ["prec_prolif", "eff prolif", "prec prolif fb", "eff prolif fb"]

dicts = [d,d2,d3,d4]
simlist = []
statelist = []
for name, dic in zip(sim_names, dicts):
    
    sim = Simulation(name, prec_model, dic, celltypes, time)
    sim.run_timecourse()
    df = sim.state_tidy 
    statelist.append(df)
    

df = pd.concat(statelist)
df = df[(df.cell_type == "Th1") | (df.cell_type == "Tfh")]
g = sns.relplot(data = df, x = "time", y = "cells", hue = "cell_type", row = "sim_name", kind = "line")

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

celltypes = ["naive", "prec", "th1", "tfh"]


d = {
     "alpha_naive" : 10,
     "alpha_prec" : 10,
     "alpha_th1" : 10,
     "alpha_tfh": 10,
     "beta_naive" : 10,
     "beta_prec" : 10,
     "beta_p_th1" : 40,
     "beta_p_tfh" : 40,
     "n_div_eff" : 1,
     "n_div_prec" : 2.0,
     "death_th1" : 2,
     "death_tfh" : 2,
     "p_th1" : 0.6,
     "p_tfh" : 0.2,
     "p_prec" : 0.2,
     "deg_myc" : 0.1,
     "EC50_myc" : 0.5,
     "fb_ifng_prob_th1" : 1,
     "fb_il10_prob_th1" : 0.1,
     "fb_il21_prob_tfh" : 1,
     "EC50_ifng_prob_th1" : 1.,
     "EC50_il10_prob_th1" : 0.1,
     "EC50_il21_prob_tfh" : 1.,
     "r_ifng" : 1,
     "r_il21" : 1,
     "r_il10" : 1,
     "deg_ifng" : 1,
     "deg_il21" : 1,
     "deg_il10" : 1,
     }


time = np.arange(0,40,0.01)
sim = Simulation("test", prec_model, d, celltypes, time)
sim.run_timecourse()


df = sim.state_tidy
g = sns.relplot(data = df, x = "time", y = "cells", hue = "cell_type", kind = "line")

df2 = sim.molecules_tidy
g = sns.relplot(data = df2, x = "time", y = "conc.", hue = "molecule", kind = "line")
#arr_dict = {"fb_ifng_prob_th1" : np.geomspace(0.1,10,50)}
#df = sim.vary_param(arr_dict)
#df2 = sim.get_relative_readouts(df)

#df3 = sim.normalize_readout_df(df, norm_idx = 49)
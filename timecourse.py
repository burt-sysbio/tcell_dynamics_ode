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
     "alpha_prec" : 2,
     "alpha_th1" : 2,
     "alpha_tfh": 2,
     "beta_naive" : 10,
     "beta_prec" : 2,
     "beta_p_th1" : 2,
     "beta_p_tfh" : 2,
     "n_div_eff" : 1,
     "n_div_prec" : 1.,
     "death_th1" : 2,
     "death_tfh" : 2,
     "p_th1" : 0.5,
     "p_tfh" : 0.3,
     "p_prec" : 0.2,
     "deg_myc" : 0.1,
     "EC50_myc" : 0.5,
     "fb_ifng_prob_th1" : 1,
     "fb_il21_prob_tfh" : 1,
     "EC50_ifng_prob_th1" : 1,
     "EC50_il21_prob_tfh" : 1,
     "r_ifng" : 1,
     "r_il21" : 1,
     }


time = np.arange(0,10,0.01)
sim = Simulation("test", prec_model, d, celltypes, time)
sim.run_timecourse()


df = sim.state_tidy
g = sns.relplot(data = df, x = "time", y = "cells", hue = "cell_type", kind = "line")

arr_dict = {"fb_ifng_prob_th1" : np.geomspace(0.1,10,50)}
df = sim.vary_param(arr_dict)

df2 = sim.get_relative_readouts(df)
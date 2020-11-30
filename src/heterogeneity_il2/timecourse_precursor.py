# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:01:00 2020

@author: Philipp
"""
# load tcell package
from tcell_model.prec_model import Simulation, prec_model
from scipy.integrate import odeint
# load other libraries
import numpy as np
import seaborn as sns
sns.set(context = "poster", style = "ticks", rc = {"lines.linewidth": 4})
import matplotlib.pyplot as plt
import pandas as pd

d = {
    "alpha_naive" : 2,
    "alpha_prec" : 2,
    "alpha_th1" : 2,
    "alpha_tfh" : 2,
    "beta_naive" : 2,
    "beta_prec" : 2,
    "beta_p_th1" : 2,
    "beta_p_tfh" : 2,
    "death_th1" : 1,
    "death_tfh" : 1,
    "n_div_prec" : 2,
    "n_div_eff" : 2,
    "p_th1" : 0.4,
    "p_tfh" : 0.3,
    "p_prec" : 0.3,
    "lifetime_myc" : 1.36,
    }

d_il2 = dict(d)
# =============================================================================
# run time course
# =============================================================================
time = np.arange(0, 10, 0.01) 
cell_types = ["naive", "prec", "th1", "tfh"]      
sim = Simulation(name = "test", model = prec_model, parameters = d, 
                 cell_types = cell_types, time = time)

df = sim.run_timecourse()

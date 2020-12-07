#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:02:12 2020

@author: burt
"""

# add package to python path
import sys
sys.path.append("/home/burt/Documents/projects/20209/test_code/tcell_model/")

# load tcell package
from tcell_model.exp import Simulation
import tcell_model.models as model

# load other libraries
import numpy as np
import seaborn as sns
sns.set(context = "poster", style = "ticks", rc = {"lines.linewidth": 4})
import matplotlib.pyplot as plt
import pandas as pd

d = {
    "b" : 0,
    "initial_cells" : 1.0,
    "alpha" : 10,
    "beta" : 10.,
    "alpha_p" : 10,
    "beta_p" : 30.,
    "lifetime_eff" : 1.0,
    "d_prec" : 0,
    "d_naive" : 0,
    "n_div" : 2,        
    "rate_il2" : 9900.26,
    "rate_C" : 0.47,
    "lifetime_myc" : 1.36,
    "K_il2": 0.1,
    "K_C" : 0.1,
    "K_myc" : 0.1,
    "hill" : 3,
    "crit_myc" : 0.1,
    "crit_C" : 0.1,
    "crit_il2" : 0.1,
    "crit" : False,
    "t0" : None,
    "c_il2_ex" : 700,
    "time_il2_perturb" : 0
    }

d_il2 = dict(d)
# =============================================================================
# run time course
# =============================================================================
time = np.arange(0, 10, 0.01) 

#d["rate_il2"] = 9000
#d["lifetime_myc"] = 1.0
# set up simulations for different models        
sim1 = Simulation(name = "IL2", mode = model.il2_menten_prolif, parameters = d, 
                  time = time, core = model.diff_effector)

sim2 = Simulation(name = "Timer+IL2", mode = model.timer_il2, parameters = d_il2, 
                  time = time, core = model.diff_effector)

sim3 = Simulation(name = "Carrying capacity", mode = model.C_thres_prolif, parameters = d, 
                  time = time, core = model.diff_effector)

# get model readouts
#print(sim1.get_readouts())

# plot time courses for all models
df1 = sim1.run_timecourse()
df2 = sim2.run_timecourse()

df = pd.concat([df1, df2])

g = sns.relplot(data = df, x = "time", y = "cells", hue = "name", kind = "line")
il2, il2_ex = sim1.get_il2_max()

print(np.trapz(il2_ex, time)/np.trapz(il2, time))

il2, il2_ex = sim2.get_il2_max()

print(np.trapz(il2_ex, time)/np.trapz(il2, time))




#fig, ax = plt.subplots()

#ax.plot(time, il2)
#ax.plot(time, il2_ex)
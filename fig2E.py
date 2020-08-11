#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:02:12 2020

@author: burt
"""

from tcell_model.exp_fig_2e import Simulation, SimList, make_sim_list, change_param2
import tcell_model.models_fig_2e as model

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

d= {
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
        "rate_il2" : 1.0,
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
        "c_il2_ex" : 0,
        "up_il2" : 0.1,
        }


sns.set(context = "poster", style = "ticks", rc = {"lines.linewidth": 4})

# =============================================================================
# make simulation for a model for beta p
# =============================================================================
time = np.arange(0, 30, 0.1)   

# =============================================================================
# set up perturbations for IL2 and IL2+timer model with external il2      
# =============================================================================
model1 = model.il2_menten_prolif
model2 = model.timer_il2


sim1 = Simulation(name="IL2", mode=model1, parameters=d, 
                  time=time, core=model.diff_effector)

sim2 = Simulation(name="IL2+Timer", mode=model2, parameters=d, 
                  time=time, core=model.diff_effector)


df1 = sim1.run_timecourse()
df2 = sim2.run_timecourse()

sum(sim1.get_il2_ex())
sum(sim2.get_il2_ex())
df = pd.concat([df1, df2])
g = sns.relplot(data = df, x = "time", y = "cells", hue = "name", kind = "line")

# =============================================================================
# run time course simulation
# =============================================================================
# make simulation lists
simlist = [sim1, sim2]

res = 30
lo = 1
hi = 10000
il2_arr = np.geomspace(lo, hi, res)
arr_name = "IL2 ext. (a.u.)"
name = "c_il2_ex"

simlist2 = [make_sim_list(sim, n = res) for sim in simlist]
simlist3 = [change_param2(simlist, name, il2_arr) for simlist in simlist2]
# make simlist3 flat
flat_list = [item for sublist in simlist3 for item in sublist]

exp = SimList(flat_list)
g, data = exp.plot_timecourses(il2_arr, arr_name, log = True, log_scale = True)
g.set(title = "", ylim = (1,150), xlim = (0,12))


# add additional lineplot
for ax, sim in zip(g.axes.flat, simlist):
    df = sim.run_timecourse()
    sns.lineplot(x = "time", y = "cells", color = "crimson",
                 data = df, ax = ax)


g.savefig("../figures/fig2/fig2E_timecourse.svg")
# =============================================================================
# without heterogeneity, vary peaktime in both models
# =============================================================================

arr = np.geomspace(lo, hi, 50)
pname = "c_il2_ex"
df1 = sim1.vary_param(pname, arr)
df2 = sim2.vary_param(pname, arr)

df = pd.concat([df1,df2])
df = df[df.readout != "Decay"]
g = sns.relplot(data = df, x = "p_val", y = "log2FC", col = "name", kind = "line",
                hue = "readout", facet_kws = {"despine" : False, "legend_out" : False})
g.set(xscale = "log")
g.set(xlabel = "IL2 ext. (a.u.)", xlim = (lo, hi), ylim = (None, 2.0), ylabel = "effect size")

g.set_titles("{col_name}")
g.savefig("../figures/fig2/fig2E_readouts.svg")

il2_arr = sim1.get_il2_max()
il2_arr2 = sim2.get_il2_max()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:02:12 2020

@author: burt
"""

from tcell_model.exp import Simulation, SimList, make_sim_list, change_param
import tcell_model.models as model
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(context = "poster", style = "ticks", rc = {"lines.linewidth": 4})

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
    "c_il2_ex" : 0
    }


def sample_prolif(sim, cv_array, pname, n):
    """
    for each val in cv_arr draw n samples from lognorm dist. with sd = val*mean
    compute and return readouts
    """
    std_array = cv_array*sim.parameters[pname]
    peak_mean = []
    peaktime_mean = [] 
    sd_peak_mean = []
    sd_peak_time = []

    for std in std_array:
        sample = sim.gen_lognorm_params(pname, std, n)
        simlist = make_sim_list(sim, n = n)
        simlist = change_param(simlist, pname, sample)
        exp = SimList(simlist)
        df = exp.run_timecourses()
    
        # get std peakmaxima and peaktimes first sim
        index = df.groupby("name").cells.max()
        mean = np.mean(index.values)
        #cv = np.std(index.values)/mean
        peaktimes = df[df.cells.isin(index.values)].time
        sd_peaktimes = peaktimes.std()
        peaktimes_mean = peaktimes.mean()
        
        peak_mean.append(mean)
        sd_peak_mean.append(np.std(index.values))
        peaktime_mean.append(peaktimes_mean)
        sd_peak_time.append(sd_peaktimes)        
        
    return peak_mean, sd_peak_mean, peaktime_mean, sd_peak_time
# =============================================================================
# make simulation for a model for beta p
# =============================================================================
time = np.arange(0, 10, 0.01)   

# =============================================================================
# set up two perturbations for IL2 and IL2+timer model with external il2      
# =============================================================================
model1 = model.il2_menten_prolif
model2 = model.timer_menten_prolif

sim1 = Simulation(name = "IL2", mode = model1, parameters = d, 
                  time = time, core = model.diff_effector)

sim2 = Simulation(name = "Timer", mode = model2, parameters = d, 
                  time = time, core = model.diff_effector)

#draw from lognorm dist and sort
pname = "beta_p"
std = 15.0
n = 50
sample = sim1.gen_lognorm_params(pname, std, n)
sample.sort()
# get top x% of array (top proliferators)
x = 5
top = int(n/100*10)
top_prolif = sample[-top:]
normal_prolif = sample[:-top]

# for timer and il2 model and for top and regular prolif generate simlist
df_list = []
sims = [sim1, sim2]
samples = [normal_prolif, top_prolif]
labels = ["normal_prolif", "top_prolif"]

for sim in sims:
    for betap_dist, label in zip(samples, labels):
        simlist = make_sim_list(sim, n = len(betap_dist))
        simlist = change_param2(simlist, pname, betap_dist)
        exp = SimList(simlist)
        df = exp.run_timecourses()
        df["prolif_type"] = label
        df_list.append(df)

df = pd.concat(df_list)

palette = ["k", "tab:grey"]
sns.set_palette(palette)
g = sns.relplot(data = df, x = "time", y = "cells", hue = "prolif_type", col = "model_name", 
                kind = "line", legend = False, 
                facet_kws = {"despine" : False, "legend_out" : False},
                aspect = 1.2)

g.set(ylim = (1, None), xlim = (0, 10))
g.set(yscale = "log")


# only use this if I want to draw additional lines
#g.set(ylim = (0.1, 1e4))
#g = sns.relplot(data = df2, x = "time", y = "cells", kind = "line")
# add curves with no heterogeneity
df3 = sim1.run_timecourse()
df4 = sim2.run_timecourse()
g.axes[0][0].plot(df3.time, df3.cells, c = "crimson")
g.axes[0][1].plot(df4.time, df4.cells, c = "crimson")
g.set_titles("")
#g.savefig("../figures/fig2/fig2E_heterogeneity.svg")
#plt.subplots_adjust(wspace = 0.2)

#g.savefig("figures/heterogeneity_timecourse_prolif.pdf")
#g.savefig("figures/heterogeneity_timecourse_prolif.svg")

cv_array = np.geomspace(0.1, 1, num = 20)
n = 50
pname = "beta_p"
#std_array = cv_array*d[pname]
#std_peakmax_il2 = []
#std_peakmax_timer = []
#std_peaktime_il2 = []
#std_peaktime_timer = []

#readouts_il2 = sample_prolif(sim1, cv_array, pname, n)
#readouts_timer = sample_prolif(sim2, cv_array, pname, n)
# =============================================================================
# for std in std_array:
#     # for different STDS get maxima and time of peak
#     sample = sim1.gen_lognorm_params(pname, std, n)
#     simlist = make_sim_list(sim1, n = n)
#     simlist = change_param2(simlist, pname, sample)
#     exp = SimList(simlist)
#     df = exp.run_timecourses()
# 
#     # get std peakmaxima and peaktimes first sim
#     index1 = df.groupby("name").cells.max()
#     max1 = np.std(index1.values)/np.mean(index1.values)
#     std1 = df[df.cells.isin(index1.values)].time.std()
#     std_peakmax_il2.append(max1)
#     std_peaktime_il2.append(std1)
#     
#     sample = sim2.gen_lognorm_params(pname, std, n)
#     simlist2 = make_sim_list(sim2, n = n)
#     simlist2 = change_param2(simlist2, pname, sample)
#     exp2 = SimList(simlist2)
#     df2 = exp2.run_timecourses()
# 
#     # get std peakmaxima and peaktimes first sim
#     index2 = df2.groupby("name").cells.max()
#     #print(np.max(index2.values))
#     max2 = np.std(index2.values) /np.mean(index2.values)   
#     std2 = df2[df2.cells.isin(index2.values)].time.std()
#     std_peakmax_timer.append(max2)
#     std_peaktime_timer.append(std2)
# =============================================================================

labels = ["Resp. peak (avg.)", "Resp. peak (sd)", "Peak time (avg.)", "Peak time (sd)"]
names = ["resp_peak_avg", "resp_peak_sd", "peak_time_avg", "peak_time_sd"]
for read1, read2, label, name in zip(readouts_il2, readouts_timer, labels, names):
    #fig, ax = plt.subplots(figsize = (5.5,4.5))
    ax.scatter(cv_array, read2, c = "tab:red")
    ax.scatter(cv_array, read1, c = "0.2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("heterogeneity")
    ax.set_ylabel(label)
    #ax.set_ylim(1e3, 1e15)
    #ax.set_xlim(0.1,2)
    #ax.set_ylim([0,10000])
    plt.tight_layout()
    #fig.savefig("../figures/fig2/fig2D_"+name+".svg")
#fig.savefig("figures/prolif_heterogeneity.pdf")

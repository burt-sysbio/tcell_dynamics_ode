"""
draw parameter values from lognorm dist with constant mean and look at systems behavior
first look at time courses for large and small CV
then vary CV systematically and quantify
"""

from exp_fig_2e import Simulation, SimList, make_sim_list, change_param
import models_fig_2e as model

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def sample_prolif(sim, cv_array, pname, n):
    """
    for each val in cv_arr draw n samples from lognorm dist. with sd = val*mean
    compute and return readouts
    """
    std_array = cv_array * sim.parameters[pname]
    peak_mean = []
    peaktime_mean = []
    sd_peak_mean = []
    sd_peak_time = []

    for std in std_array:
        sample = sim.gen_lognorm_params(pname, std, n)

        
        simlist = make_sim_list(sim, n=n)
        simlist = change_param(simlist, pname, sample)
        exp = SimList(simlist)
        df = exp.run_timecourses()

        # get std peakmaxima and peaktimes first sim
        index = df.groupby("name").cells.max()
        mean = np.mean(index.values)
        # cv = np.std(index.values)/mean
        peaktimes = df[df.cells.isin(index.values)].time
        sd_peaktimes = peaktimes.std()
        peaktimes_mean = peaktimes.mean()

        peak_mean.append(mean)
        sd_peak_mean.append(np.std(index.values))
        peaktime_mean.append(peaktimes_mean)
        sd_peak_time.append(sd_peaktimes)

    df = pd.DataFrame({"peak mean": peak_mean,
                       "peak sd": sd_peak_mean,
                       "peaktime mean": peaktime_mean,
                       "peaktime sd": sd_peak_time})
    return df


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
        "rate_il2" : 0.5,
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
time = np.arange(0, 12, 0.01)

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
plt.show()

n = 200 # number of samples to draw for each val of cv_arr
pname2 = "rate_il2"
pname = "up_il2"

# show timecourse once for samples drawn small dist. with small cv and once with large cv
sd_arr = [0.1, 0.5, 1.0]
sims = [sim1, sim2]
df_list = []
labels = ["CV=0.1", "CV=0.5", "CV=1.0"]

# loop over both simulations (high cv and low cv) and over both models, then draw params from lognorm dist
for sim in sims:
    for sd, label in zip(sd_arr, labels):
        # draw samples from lognorm dist for 2 parameters
        sample = sim.gen_lognorm_params(pname, sd, n)
        fig, ax = plt.subplots()
        sns.distplot(sample, ax = ax)
        #sample2 = sim.gen_lognorm_params(pname2, sd, n)
        # for each val in sample arr make new simulation
        # make a simulation list as deepcopy from original list
        simlist = make_sim_list(sim, n = len(sample))
        # then change parameters in each list
        simlist = change_param(simlist, pname, sample)
        #simlist = change_param(simlist, pname2, sample2)
        exp = SimList(simlist)
        df = exp.run_timecourses()
        df["sd"] = label
        df_list.append(df)

df = pd.concat(df_list)


g = sns.relplot(data = df, x = "time", y = "cells", hue = "model_name", col = "sd", kind = "line",
                ci = "sd")

g.set_titles("{col_name}")
plt.show()
# vary rate_il2 by using default mean and varying sd, then drawing from sd and compute means
n_samples = 100
n_cv_arr = 50
cv_arr = np.geomspace(0.1, 1, num = n_cv_arr)
pname = "up_il2"

readouts_il2 = sample_prolif(sim1, cv_arr, pname, n_samples)
readouts_timer = sample_prolif(sim2, cv_arr, pname, n_samples)

readouts_il2.to_csv("readouts_il2.csv")
readouts_timer.to_csv("readouts_timer.csv")

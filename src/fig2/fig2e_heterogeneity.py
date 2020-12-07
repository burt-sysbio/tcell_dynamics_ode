"""
analyse effect of heterogeneous IL2 secretion on IL2 and IL2+timer model
draw secretion rates from lognorm dist with constant mean and look at systems behavior
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
sns.set(context = "poster", style = "ticks", rc = {"lines.linewidth": 4})

def dummy(sim, sample, pname, n):
    simlist = make_sim_list(sim, n)
    simlist = change_param(simlist, pname, sample)
    exp = SimList(simlist)
    return exp


def dummy2(exp):

    df = exp.run_timecourses()
    # get std peakmaxima and peaktimes first sim
    index = df.groupby("name").cells.max()
    max_values = index.values
    mean_max = np.mean(max_values)
    sd_max = np.std(max_values)
    peaktimes = df[df.cells.isin(max_values)].time
    sd_peaktimes = peaktimes.std()
    mean_peaktimes = peaktimes.mean()

    return np.array([mean_max, sd_max, mean_peaktimes, sd_peaktimes])


def dummy3(exp):
    """
    get readouts from experiments and compute means and standard deviation for each readout
    return as flat array for stacking
    """
    reads = exp.get_readouts()
    reads = pd.concat(reads)
    means = reads.groupby("readout")["read_val"].mean()
    sds = reads.groupby("readout")["read_val"].std()

    readouts = ["Peak Height", "Peak Time", "Response Size"]
    means = means[readouts]
    sds = sds[readouts]
    arr = np.concatenate((means, sds))
    return arr


def sample_prolif(sim1, sim2, cv_arr, pname, n_samples):
    """
    for each val in cv_arr draw n samples from lognorm dist. with sd = val*mean
    compute and return readouts
    """
    sd_array = cv_arr * sim1.parameters[pname]
    reads1_list = []
    reads2_list = []

    for sd in sd_array:
        sample = sim1.gen_lognorm_params(pname, sd, n_samples)
        exp1 = dummy(sim1, sample, pname, len(sample))
        exp2 = dummy(sim2, sample, pname, len(sample))
        reads1 = dummy3(exp1)
        reads2 = dummy3(exp2)

        #dummy3(exp1)
        reads1_list.append(reads1)
        reads2_list.append(reads2)

    # format output into one single data frame
    arr1 = np.stack(reads1_list)
    arr2 = np.stack(reads2_list)
    colnames = ["Mean Peak Height", "Mean Peak Time", "Mean Response Size",
                "SD Peak Height", "SD Peak Time", "SD Response Size"]

    df1 = pd.DataFrame(arr1, columns = colnames)
    df2 = pd.DataFrame(arr2, columns = colnames)
    df1["CV"] = cv_arr
    df2["CV"] = cv_arr
    df1["name"] = sim1.name
    df2["name"] = sim2.name
    df = pd.concat([df1, df2])

    return df


def lognorm_vary(sim1, sim2, cv_arr, pname, n_samples, n_repeats):
    """
    for each val in cv_arr generate params from lognorm dist and run timecourses
    return df for timecourse data and for samples
    """
    df_list = []
    samples_list = []

    for cv in cv_arr:
        sd = cv*sim1.parameters[pname]
        sample_arr = np.array([])
        for i in range(n_repeats):
            # draw samples from lognorm dist for 2 parameters
            sample = sim1.gen_lognorm_params(pname, sd, n_samples)
            sample_arr = np.append(sample_arr, sample)
            exp1 = dummy(sim1, sample, pname, len(sample))
            exp2 = dummy(sim2, sample, pname, len(sample))
            df1 = exp1.run_timecourses()
            df2 = exp2.run_timecourses()
            # same for sim2
            df = pd.concat([df1, df2])
            df["rep"] = i
            df["CV"] = cv
            df_list.append(df)

        samples_list.append(sample_arr)

    df = pd.concat(df_list)
    df2 = pd.DataFrame(samples_list)
    df2 = df2.T
    colnames = [str(cv) for cv in cv_arr]
    df2.columns = colnames

    return df, df2


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
        "up_il2" : 0.087,
        }

# =============================================================================
# make simulation for a model for beta p
# =============================================================================

# =============================================================================
# set up perturbations for IL2 and IL2+timer model with external il2
# =============================================================================
time = np.arange(0, 12, 0.05)

model1 = model.il2_menten_prolif
model2 = model.timer_il2
sim1 = Simulation(name="IL2", mode=model1, parameters=d,
                  time=time, core=model.diff_effector)
sim2 = Simulation(name="IL2+Timer", mode=model2, parameters=d,
                  time=time, core=model.diff_effector)

# plot time course
df1 = sim1.run_timecourse()
df2 = sim2.run_timecourse()
df = pd.concat([df1, df2])
g = sns.relplot(data = df, x = "time", y = "cells",
                hue = "name", kind = "line",
                aspect=0.8)
plt.show()
#
# loop over both simulations (high cv and low cv) and over both models, then draw params from lognorm dist
# plot time course for diff uptake rates IL2 as heterogeneity from lognorm dist
pnames = ["rate_il2"]
cv_arr = [0.5, 1.0, 2.0]
n_samples = 200
rep = 1
res_cv_arr = 30
cv_reads = np.geomspace(1, 10, num = res_cv_arr)

# do  analysis for uptake and secretion rate of IL2
# decrease time increment

for pname in pnames:
    # show timecourse once for samples drawn small dist. with small cv and once with large cv
    sims = [sim1, sim2]
    df, df_samples = lognorm_vary(sim1, sim2, cv_arr, pname, n_samples, rep)

    df_samples.to_csv("data_fig2e_lognorm_samples_"+pname+".csv", index=False)
    df.to_csv("data_fig2e_timecourse_"+pname+".csv", index=False)

    # vary rate_il2 by using default mean and varying sd, then drawing from sd and compute means
    #df_readouts = sample_prolif(sim1, sim2, cv_reads, pname, n_samples)
    #df_readouts.to_csv("data_fig2e_readouts_"+pname+".csv", index=False)

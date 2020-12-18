import numpy as np
from src.modules.exp import Sim, Simlist
from src.analysis.antigen_eff.parameters import d_no_ag, d_ag, d_hi_ag
from src.modules.models import vir_model_gamma, vir_model_ode
import src.modules.proc as proc
import src.modules.pl as pl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# init model
name = "il2"
vir_model = vir_model_gamma
time = np.arange(0,50,0.01)

# create parameter list
d_list = [d_no_ag, d_ag, d_hi_ag]

# create simulation list
sim_list = [Sim(name = name, params = dic, time = time, virus_model = vir_model) for dic in d_list]

# add ids
names = ["wide", "mid", "sharp"]

# run model
exp2 = Simlist(sim_list, names)
cells, molecules = exp2.run_sim()

# plot output
pl.plot_timecourse(cells, hue = "id")
plt.show()

pl.plot_timecourse(molecules, hue = "id")
plt.show()

# run parameter scan
sim_il2 = Sim(name = "il2", params = d_ag, time = time, virus_model= vir_model)
sim_timer = Sim(name = "timer", params = d_ag, time = time, virus_model= vir_model)

if vir_model == vir_model_gamma:
    pname1 = "vir_alpha"
    pname2 = "vir_beta"
    SD = False
else:
    pname1 = "vir_growth"
    pname2 = "vir_death"
    SD = False

arr = np.geomspace(0.1,10,30)
res1 = proc.pscan(sim_il2, arr, "vir_beta")
res2 = proc.pscan(sim_timer, arr, "vir_beta")
res3 = proc.pscan2d(sim_il2, pname1, pname2, prange1=(2,10), prange2=(0.1,100), res = 30, SD = SD)
res4 = proc.pscan2d(sim_timer, pname1, pname2, prange1=(2,10), prange2=(0.1,100), res = 30, SD = SD)

for r in [res1,res2]:
    g = pl.plot_pscan(r, "val_norm")
    g.set(xlabel = "lifetime virus")
    plt.show()

readouts = ["Area", "Peak", "Peaktime"]
for r in readouts:
    g = pl.plot_heatmap(res3, "val_norm", r, vmin = -1, vmax = 1)
    plt.show()
    g = pl.plot_heatmap(res4, "val_norm", r, vmin = -1, vmax = 1)
    plt.show()
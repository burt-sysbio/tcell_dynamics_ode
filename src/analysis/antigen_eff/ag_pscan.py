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
vir_model = vir_model_gamma
time = np.arange(0, 50, 0.01)

# init models
sim_il2 = Sim(name = "il2", params = d_ag, time = time, virus_model= vir_model)
sim_timer = Sim(name = "timer", params = d_ag, time = time, virus_model= vir_model)

# run parameter scans
if vir_model == vir_model_gamma:
    pname1 = "vir_alpha"
    pname2 = "vir_beta"
else:
    pname1 = "vir_growth"
    pname2 = "vir_death"

res = 50
up = 10
down = 0.1
arr = np.geomspace(down, up, res)
res1 = proc.pscan(sim_il2, arr, "vir_beta")
res2 = proc.pscan(sim_timer, arr, "vir_beta")
res3 = proc.pscan2d(sim_il2, pname1, pname2, prange1=(1,100), prange2=(0.1,100), res = res)
res4 = proc.pscan2d(sim_timer, pname1, pname2, prange1=(1,100), prange2=(0.1,100), res = res)

res3.to_csv("ag_pscan2d_gamma_il2.csv")
res4.to_csv("ag_pscan2d_gamma_timer.csv")

# plot parameter scans
for r in [res1, res2]:
    g = pl.plot_pscan(r, "val_norm")
    g.set(xlabel = "lifetime virus")
    plt.show()


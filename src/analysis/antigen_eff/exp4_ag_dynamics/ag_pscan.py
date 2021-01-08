import numpy as np
from src.modules.exp import Sim
from src.analysis.antigen_eff.exp1_const_ag.parameters import d_ag
from src.modules.models import vir_model_gamma
import src.modules.proc as proc
import src.modules.pl as pl
import matplotlib.pyplot as plt

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

# plot parameter scans
for r in [res1, res2]:
    g = pl.plot_pscan(r, "val_norm")
    g.set(xlabel = "lifetime virus")
    plt.show()


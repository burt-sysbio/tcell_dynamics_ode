# test different antigen dynamics with and without chronic cells
import numpy as np
from src.modules.exp import Sim
from src.analysis.antigen_eff.exp4_ag_dynamics.parameters import d
from src.modules.models import vir_model_gamma
import src.modules.proc as proc
import src.modules.pl as pl
import matplotlib.pyplot as plt
import pandas as pd

# init model
vir_model = vir_model_gamma
time = np.arange(0, 50, 0.01)

# init models
name = "il2"

d1 = dict(d)
d1["r_chronic"] = 0
sim = Sim(name = name, params = d, time = time, virus_model= vir_model)
sim2 = Sim(name = name, params = d1, time = time, virus_model= vir_model)
# run parameter scans
pname = "vir_beta"

res = 50
up = 1
down = 0.1
arr = np.linspace(down, up, res)
res1 = proc.pscan(sim, arr, pname)
res2 = proc.pscan(sim2, arr, pname)
res1["chronic"] = "on"
res2["chronic"] = "off"
res = pd.concat([res1,res2])
res.param_value = 1/res.param_value
# plot parameter scans
g = pl.plot_pscan(res, value_col= "val_min", cells = ["teff"], hue = "chronic",
                  palette= ["grey", "k"])
g.set(xlabel = "avg virus lifetime")
plt.show()


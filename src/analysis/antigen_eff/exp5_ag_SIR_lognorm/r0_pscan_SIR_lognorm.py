#use SIR model with lognorm parameterization, vary r0 and check output
import numpy as np
from src.modules.exp import Sim
from src.analysis.antigen_eff.exp5_ag_SIR_lognorm.parameters import d
from src.modules.models import vir_model_SIR
import src.modules.proc as proc
import src.modules.pl as pl
import matplotlib.pyplot as plt
import pandas as pd


# init model
vir_model = vir_model_SIR
time = np.arange(0, 50, 0.01)

# init models
name = "il2"
d["r_chronic"] = 1
d1 = dict(d)
d1["r_chronic"] = 0
sim = Sim(name = name, params = d, time = time, virus_model= vir_model)
sim2 = Sim(name = name, params = d1, time = time, virus_model= vir_model)

# run parameter scans
pname = "SIR_r0"

res = 50
down = 1
up = 3

arr = np.linspace(down, up, res)
res1 = proc.pscan(sim, arr, pname)
res2 = proc.pscan(sim2, arr, pname)
res1["chronic"] = "on"
res2["chronic"] = "off"
res = pd.concat([res1,res2])

# plot parameter scans
g = pl.plot_pscan(res, value_col= "val_min", cells = ["teff"], hue = "chronic",
                  palette= ["grey", "k"])
plt.show()
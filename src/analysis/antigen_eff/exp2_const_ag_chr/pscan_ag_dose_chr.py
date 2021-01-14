import numpy as np
from src.modules.exp import Sim
from src.analysis.antigen_eff.exp2_const_ag_chr.parameters import d
from src.modules.models import vir_model_const
from src.modules.proc import pscan
from src.modules.pl import plot_pscan
import matplotlib.pyplot as plt
import pandas as pd

# init model
name = "il2"
vir_model = vir_model_const
time = np.arange(0, 50, 0.01)

d1 = dict(d)
d1["r_chronic"] = 0
d2 = dict(d)
d2["r_chronic"] = 10.0
# create simulation
sim1 = Sim(name = name, params = d1, time = time, virus_model = vir_model)
sim2 = Sim(name = name, params = d2, time = time, virus_model = vir_model)


res = 50
pname = "vir_load"
arr = np.linspace(0,1.0, res)

pscan1 = pscan(sim1, arr, pname)
pscan2 = pscan(sim2, arr, pname)

pscan1["chronic"] = "on"
pscan2["chronic"] = "off"

pscan = pd.concat([pscan1, pscan2])
pscan = pscan.loc[pscan.cell == "teff"]

g = plot_pscan(pscan, row = "chronic", value_col= "val_min", palette= ["grey"], hue = "cell")
#g.set(ylim(0,20))
plt.show()
import numpy as np
from src.modules.exp import Sim
from src.analysis.antigen_eff.exp3_chr_fb.parameters import d
from src.modules.models import vir_model_const
import pandas as pd
from src.modules.proc import pscan
from src.modules.pl import plot_pscan
import matplotlib.pyplot as plt

# init model
name = "il2"
vir_model = vir_model_const
time = np.arange(0, 50, 0.01)

d["r_chronic"] = 10.0
# compare feedback and no fb scenario
d1 = dict(d)
d2 = dict(d)
d2["pos_fb_chr"] = 10.0
# create simulation
sim1 = Sim(name = name, params = d1, time = time, virus_model = vir_model)
sim2 = Sim(name = name, params = d2, time = time, virus_model = vir_model)

res = 100
pname = "vir_load"
arr = np.linspace(0,1.0, res)

pscan1 = pscan(sim1, arr, pname)
pscan2 = pscan(sim2, arr, pname)

pscan1["feedback"] = "off"
pscan2["feedback"] = "on"

pscan = pd.concat([pscan1, pscan2])
pscan = pscan.loc[pscan.cell == "teff"]

g = plot_pscan(pscan, row = "feedback", value_col= "val_min", palette= ["grey"])
#g.set(ylim(0,20))
plt.show()
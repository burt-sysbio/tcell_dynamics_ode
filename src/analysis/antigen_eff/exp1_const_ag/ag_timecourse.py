import numpy as np
from src.modules.exp import Sim, Simlist
from src.analysis.antigen_eff.exp1_const_ag.parameters import d
from src.modules.models import vir_model_const
import src.modules.pl as pl
import matplotlib.pyplot as plt
import seaborn as sns
# init model
name = "il2"
vir_model = vir_model_const
time = np.arange(0, 10, 0.01)

# create parameter list
d1 = dict(d)
d2 = dict(d1)
d3 = dict(d1)
d4 = dict(d1)
d1["vir_load"] = 0
d2["vir_load"] = 0.1
d3["vir_load"] = 0.5
d4["vir_load"] = 5
d_list = [d1, d2, d3, d4]

names = ["none", "lo",  "mid", "hi"]

# create simulation list
sim_list = [Sim(name = name, params = dic, time = time, virus_model = vir_model) for dic in d_list]

# run model
id = "ag conc."
exp2 = Simlist(sim_list, names, idtype= id)
cells, molecules = exp2.run_sim()

# plot output
g = pl.plot_timecourse(cells, hue = id)
g.set(ylabel = "cells")
plt.show()

g =sns.relplot(data=molecules, x="time", hue = id, col="cell", y="value", kind="line",
               facet_kws= {"sharey" : False})
g.set_titles("{col_name}")
plt.show()
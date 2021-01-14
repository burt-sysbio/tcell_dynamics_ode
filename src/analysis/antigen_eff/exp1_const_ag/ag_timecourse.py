import numpy as np
from src.modules.exp import Sim, Simlist
from src.analysis.antigen_eff.exp1_const_ag.parameters import d
from src.modules.models import vir_model_const
import src.modules.pl as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# init model
name = "il2"
vir_model = vir_model_const
time = np.arange(0, 10, 0.01)

# create parameter list
names = ["none", "lo", "mid", "hi"]
vir_loads = [0,0.01,0.1,10]
d_list = [dict(d) for i in range(len(vir_loads))]
for dic,v in zip(d_list,vir_loads):
    dic["vir_load"] = v

# create simulation list
sim_list = [Sim(name = name, params = dic, time = time, virus_model = vir_model) for dic in d_list]

sim = Sim(name = name, params = d, time = time, virus_model = vir_model)
cells, molecules = sim.run_sim()

pl.plot_timecourse(cells, cells = ["teff"])
plt.show()

cell_list = []
molecules_list = []
for sim, n in zip(sim_list, names):
    cells, molecules = sim.run_sim()
    cells["ag_conc"] = n
    molecules["ag_conc"] = n
    cell_list.append(cells)
    molecules_list.append(molecules)

cells = pd.concat(cell_list)
molecules = pd.concat(molecules_list)

g =pl.plot_timecourse(cells, cells = ["teff"], hue = "ag_conc", col="cell", palette= "Reds")
g.set_titles("{col_name}")
plt.show()

g = sns.relplot(data = molecules, x= "time", y = "value",
                hue = "ag_conc", col = "cell", palette= "Reds", facet_kws= {"sharey" : False}, kind = "line")
g.set(yscale = "log", ylabel = "conc.")
g.set_titles("{col_name}")
plt.show()
import numpy as np
from src.modules.exp import Sim
from src.tests.parameters import d
from src.modules.models import vir_model_gamma, vir_model_ode
import src.modules.plotting_module as plots
from src.modules.analysis_module import pscan
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# init model
name = "il2"
params_no_ag = dict(d)
params_no_ag["vir_alpha"] = 1
params_no_ag["vir_beta"] = 0.1

params_ag = dict(d)
params_ag["vir_alpha"] = 1
params_ag["vir_beta"] = 1
params_hi_ag = dict(d)
params_hi_ag["vir_alpha"] = 1
params_hi_ag["vir_beta"] = 2

time = np.arange(0,10,0.01)

# run model
sim_no_ag = Sim(name = name, params = params_no_ag, time = time, virus_model= vir_model_gamma)
sim_ag = Sim(name = name, params = params_ag, time = time, virus_model= vir_model_gamma)
sim_hi_ag = Sim(name = name, params = params_hi_ag, time = time, virus_model= vir_model_gamma)

sim_list = [sim_no_ag, sim_ag, sim_hi_ag]
names = ["wide", "mid", "sharp"]

cell_list = []
mol_list = []

for sim, n in zip(sim_list, names):
    cells, molecules = sim.run_sim()

    # attach output and combine dfs
    cells["antigen"] = n
    molecules["antigen"] = n
    cell_list.append(cells)
    mol_list.append(molecules)

# combine output for ag and no ag simulation
cells = pd.concat(cell_list)
cells = cells[cells["cell"] != "tnaive"]
molecules = pd.concat(mol_list)

# plot output

g = sns.relplot(data = cells, x = "time", y = "value", hue = "antigen",
                kind = "line")
g.set(yscale = "log", ylim= (1e-1, None))
g.set_titles(col_template = '{col_name}')
plt.show()

g = sns.relplot(data = molecules, x = "time", y = "value", hue = "antigen",
                kind = "line", col = "cell", facet_kws= {"sharey":False})
#g.set(yscale = "log", ylim= (1e-1, None))
g.set_titles(col_template = '{col_name}')
plt.show()

#g.savefig("../figures/antigen_effects/vir_cells.png")

# run parameter scan
pname = "vir_beta"
sim = Sim(name = name,
          params = d,
          time = time,
          virus_model= vir_model_gamma)
arr = np.geomspace(0.1,100,30)
out = pscan(sim, arr, pname)

out = out.loc[out.cell == "teff"]

g = sns.relplot(data = out, x = "param_value", y = "value", col = "readout",
                kind = "line", facet_kws={"sharey": False})
g.set(xscale = "log")
plt.show()
import numpy as np
from src.modules.exp import Sim
from src.tests.parameters import d
import src.modules.plotting_module as plots
from src.modules.analysis_module import pscan
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# init model
name = "il2"
params_no_ag = dict(d)
params_no_ag["vir_load"] = 0
params_ag = dict(d)
params_ag["vir_load"] = 2.5
params_hi_ag = dict(d)
params_hi_ag["vir_load"] = 5

time = np.arange(0,10,0.01)

# run model
sim_no_ag = Sim(name, params_no_ag, time)
sim_ag = Sim(name, params_ag, time)
sim_hi_ag = Sim(name, params_hi_ag, time)

sim_list = [sim_no_ag, sim_ag, sim_hi_ag]
names = ["lo", "mid", "hi"]

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
g = sns.relplot(data = cells, x = "time", y = "value", hue = "antigen", col = "cell",
                facet_kws= {"sharey" : False}, kind = "line")
g.set(yscale = "log", ylim= (1e-1,None))
g.set_titles(col_template = '{col_name}')
plt.show()
g.savefig("../figures/antigen_effects/vir_cells.png")


g = sns.relplot(data = molecules, x = "time", y = "value", hue = "antigen", col = "cell",
                facet_kws= {"sharey" : False}, kind = "line")
g.set(yscale = "log", ylim= (1e-1,None))
g.set_titles(col_template = '{col_name}')
plt.show()
g.savefig("../figures/antigen_effects/vir_molecules.png")

# run parameter scan
pname = "vir_load"
arr = np.linspace(0,100,30)
#out = pscan(sim, arr, pname)


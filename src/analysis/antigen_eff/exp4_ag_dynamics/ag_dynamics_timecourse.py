import numpy as np
from src.modules.exp import Sim
from src.analysis.antigen_eff.exp4_ag_dynamics.parameters import d
from src.modules.models import vir_model_gamma
import src.modules.pl as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# init model
name = "il2"
vir_model = vir_model_gamma
time = np.arange(0, 10, 0.01)

# create parameter list
names = ["A", "B", "C"]
alphas = [2,7,20]
betas = [0.2, 1.1, 3]
d_list = [dict(d) for i in range(len(names))]
for dic, alpha, beta in zip(d_list,alphas, betas):
    dic["vir_alpha"] = alpha
    dic["vir_beta"] = beta

# create simulation list
sim_list = [Sim(name = name, params = dic, time = time, virus_model = vir_model) for dic in d_list]

cell_list = []
molecules_list = []
id = "SD(virus)"
for sim, beta, alpha in zip(sim_list, betas, alphas):
    cells, molecules = sim.run_sim()
    SD = round(alpha/(beta**2), 2)
    cells[id] = str(SD)
    molecules[id] = str(SD)
    cell_list.append(cells)
    molecules_list.append(molecules)

cells = pd.concat(cell_list)
molecules = pd.concat(molecules_list)

g =pl.plot_timecourse(cells, hue = id, col="cell")
g.set_titles("{col_name}")
plt.show()

molecules = molecules.loc[molecules["cell"] == "Virus"]
g = sns.relplot(data = molecules, x= "time", y = "value",
                hue = id, facet_kws= {"sharey" : False}, kind = "line",
                legend = False, aspect= 1.1)

g.set_titles("{col_name}")
g.set(ylabel="virus density")
plt.show()
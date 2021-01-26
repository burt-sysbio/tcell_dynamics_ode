#use SIR model with lognorm parameterization, vary r0 and check output
import numpy as np
from src.modules.exp import Sim
from src.analysis.antigen_eff.exp5_ag_SIR_lognorm.parameters import d
from src.modules.virus_models import vir_model_SIR
import src.modules.proc as proc
import src.modules.pl as pl
import matplotlib.pyplot as plt

# init model
vir_model = vir_model_SIR
time = np.arange(0, 20, 0.01)

# init models
name = "il2"

d["r_chronic"] = 0
sim = Sim(name = name, params = d, time = time, virus_model= vir_model)

# run parameter scans
pname = "SIR_r0"

res = 15
down = 1
up = 5

arr = np.linspace(down, up, res)
cells, molecules = proc.run_timecourses(sim, arr, pname)
mols_norm = proc.norm_molecules(molecules, d)

cells = cells[cells.cell == "teff"]

g = pl.plot_timecourses(cells, log = False, log_scale= True, cmap = "coolwarm")
plt.show()

g = pl.plot_timecourses(mols_norm, cells = ["IL2"], log = False, log_scale= True, ylabel = "conc. / EC50")
plt.show()

g = pl.plot_timecourses(mols_norm, cells = ["Virus"], log = False, log_scale= False, ylabel = "conc. / EC50")
plt.show()


# also run parameter scan
res = 50
arr = np.linspace(down, up, res)
res = proc.pscan(sim, arr, pname)

# plot parameter scans
g = pl.plot_pscan(res, value_col= "val_min", cells = ["teff"], palette= ["k"])
plt.show()
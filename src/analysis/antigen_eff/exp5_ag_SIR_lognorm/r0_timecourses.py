#use SIR model with lognorm parameterization, vary r0 and check output
import numpy as np
from src.modules.exp import Sim
from src.analysis.antigen_eff.exp5_ag_SIR_lognorm.parameters import d
from src.modules.virus_models import vir_model_SIR
import src.modules.proc as proc
import src.modules.pl as pl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import NullFormatter

# init model
vir_model = vir_model_SIR
time = np.arange(0, 60, 0.01)

# init models
name = "il2"

d["r_chronic"] = 0
sim = Sim(name = name, params = d, time = time, virus_model= vir_model)

# run parameter scans
pname = "SIR_r0"

# choose specific values to make dynamics clear
arr_timecourse = np.array([1,1.5,2,2.5,5])
cells, molecules = proc.run_timecourses(sim, arr_timecourse, pname)
mols_norm = proc.norm_molecules(molecules, d)

cells = cells[cells.cell == "all_cells"]

cmap = sns.color_palette("flare", len(arr_timecourse))

# plot settings overall
aspect = 1.2
xlim = (0,40)
g = pl.plot_timecourses(cells, pl_cbar= False, hue_log = False, scale = "log", ylabel = "n cells", cmap = cmap,
                        legend = False, ylim = (100,3e5), aspect = aspect)
g.set(xlim= xlim)
plt.show()
g.savefig("../figures/antigen_effects/timecourse_cells.png")

g = pl.plot_timecourses(mols_norm, pl_cbar= False, cells = ["IL2"], hue_log = False, scale = "log", ylabel = "conc./EC50",
                        cmap = cmap, ylim = (0.1,None), aspect = aspect, legend = False)
g.set(xlim= xlim)
plt.show()

g = pl.plot_timecourses(mols_norm, pl_cbar= False, cells = ["Virus"], hue_log = False, ylabel = "conc./EC50", cmap = cmap,
                        ylim = (0,0.5), aspect = aspect, legend = False)
g.set(xlim= xlim)
plt.show()
g.savefig("../figures/antigen_effects/timecourse_virus.png")


# also run parameter scan
lower = 1
upper = 10
res = 100
arr_pscan = np.geomspace(lower, upper, res)
res = proc.pscan(sim, arr_pscan, pname)

## plot parameter scans
g = pl.plot_pscan(res, xscale = "log", yscale = "log", value_col= "value", cells = ["teff"], color = "grey")
plt.show()

# only look at peak for presentation figure
res2 = res.loc[res.readout == "Peak"]
g = pl.plot_pscan(res2, xscale = "log", yscale = "log", value_col= "value", cells = ["teff"], color = "grey",
                  aspect = 1.2)

ax = g.axes.flatten()[0]
for a, c in zip(arr_timecourse, cmap):
    ax.axvline(x = a, color = c, ls = "dashed")

ax.set_xlabel("infection rate r0")
ax.set_title("")
ax.set_ylabel("response peak")
ax.xaxis.set_minor_formatter(NullFormatter())
plt.show()

g.savefig("../figures/antigen_effects/pscan_r0.png")
# check why 1.7 etc does not return good results
#sim.params[pname] = 1.7
from src.modules.proc import get_readouts
from src.modules.readout_module import *

#cells, molecules = sim.run_sim()
#out = check_criteria2(cells)
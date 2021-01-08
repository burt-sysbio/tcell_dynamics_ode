import numpy as np
from src.modules.exp import Sim, Simlist
from src.analysis.antigen_eff.exp1_const_ag.parameters import d
from src.modules.models import vir_model_const
from src.modules.proc import get_readouts
from src.modules.readout_module import check_criteria2
from src.modules.pl import plot_timecourse
from src.modules.proc import pscan
from src.modules.pl import plot_pscan
import matplotlib.pyplot as plt
import seaborn as sns
# init model
name = "il2"
vir_model = vir_model_const
time = np.arange(0, 200, 0.01)

# create simulation
sim = Sim(name = name, params = d, time = time, virus_model = vir_model)
pname = "vir_load"

cells, molecules = sim.run_sim()

plot_timecourse(cells, col = "cell")
plt.show()

reads = get_readouts(cells, check_criteria2)

res = 30
arr = np.linspace(0,0.13, res)
pscan = pscan(sim, arr, pname)

g = plot_pscan(pscan, column= "val_norm")
plt.show()
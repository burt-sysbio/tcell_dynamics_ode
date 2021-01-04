import numpy as np
from src.modules.exp import Sim, Simlist
from src.analysis.antigen_eff.parameters import d_no_ag, d_ag, d_hi_ag
from src.modules.models import vir_model_gamma, vir_model_ode
import src.modules.proc as proc
import src.modules.pl as pl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# init model
name = "il2"
vir_model = vir_model_gamma
time = np.arange(0, 10, 0.01)

# create parameter list
d_list = [d_no_ag]

# create simulation list
sim_list = [Sim(name = name, params = dic, time = time, virus_model = vir_model) for dic in d_list]

# add ids
names = ["wide"]

# run model
exp2 = Simlist(sim_list, names)
cells, molecules = exp2.run_sim()

# plot output
pl.plot_timecourse(cells, hue = "id")
plt.show()

pl.plot_timecourse(molecules, hue = "id")
plt.show()
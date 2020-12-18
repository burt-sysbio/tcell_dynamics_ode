import numpy as np
import matplotlib.pyplot as plt
# my modules
import src.mypkg.exp as exp
import src.mypkg.pl as pl
import src.mypkg.proc as proc
import src.mypkg.models as mdl

# paramters
from src.tests.parameters import d

# init model
params = dict(d)
time = np.arange(0,50,0.01)

# create multiple sims
mysim1 = exp.Sim(name = "il2", params = params, time = time, virus_model= mdl.vir_model_gamma)
mysim2 = exp.Sim(name = "il2", params = params, time = time, virus_model = mdl.vir_model_ode)
mysim3 = exp.Sim(name = "timer", params = params, time = time, virus_model= mdl.vir_model_gamma)
mysim4 = exp.Sim(name = "timer", params = params, time = time, virus_model= mdl.vir_model_ode)

mysims = [mysim1, mysim2, mysim3, mysim4]
for mysim in mysims:
    # run timecourse
    cells, molecules = mysim.run_sim()

    # plot timecourse
    g = pl.plot_timecourse(cells)
    g = pl.plot_timecourse(molecules)

    # get readouts
    reads = proc.get_readouts(cells)

    # run pscan
    pname = "beta"
    arr = np.geomspace(0.1,3,3)
    myscan1 = proc.pscan(mysim, arr, pname)

    # plot pscan
    g = pl.plot_pscan(myscan1)

    # run pscan2d
    myscan2 = proc.pscan2d(mysim, "beta", "n_div", (1,10), (1,10), res = 5)

    # plot heatmap
    g = pl.plot_heatmap(myscan2, readout = "Area")


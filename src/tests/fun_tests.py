import numpy as np
import matplotlib.pyplot as plt
# my modules
import src.modules.exp as exp
import src.modules.pl as pl
import src.modules.proc as proc
import src.modules.models as mdl
import warnings
warnings.simplefilter("error")
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

mysims = [mysim1]
for mysim in mysims:
    # run timecourse
    cells, molecules = mysim.run_sim()

    # plot timecourse
    g = pl.plot_timecourse(cells)
    plt.show()
    g = pl.plot_timecourse(molecules)
    plt.show()

    # get readouts
    reads = proc.get_readouts(cells)

    # run pscan
    pname = "beta"
    arr = np.geomspace(0.1,1,12)
    myscan1 = proc.pscan(mysim, arr, pname)

    # plot pscan
    g = pl.plot_pscan(myscan1)
    plt.show()



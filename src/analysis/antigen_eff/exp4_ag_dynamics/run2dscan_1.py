import numpy as np
from src.modules.exp import Sim
from src.analysis.antigen_eff.exp4_ag_dynamics.parameters import d
from src.modules.models import vir_model_gamma
import src.modules.proc as proc
import src.modules.pl as pl
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing


# init model
vir_model = vir_model_gamma
name = "il2"
time = np.arange(0, 50, 0.01)
pname1 = "vir_alpha"
pname2 = "vir_beta"

# init models
sim = Sim(name = name, params = d, time = time, virus_model= vir_model)

res = 30
rangefun = np.geomspace
inputs = proc.get_2dinputs(prange1=(1,13), prange2=(0.1,10.0), res = res, rangefun = rangefun)

n_cor = multiprocessing.cpu_count()
params = [sim, pname1, pname2]
outputs = Parallel(n_jobs=n_cor)(delayed(proc.get_2doutput)(i, *params) for i in inputs)
out = proc.get_2dscan(outputs)

readouts = ["Area", "Peak", "Peaktime"]
df_list = [out]
for r in readouts:
    for df in df_list:
        pl.plot_heatmap(df, "value", r, cmap = "Reds", log_color = True, log_axes= True)
        plt.show()

out2 = out.loc[out.cell == "teff"]
out2 = out2.loc[out2.readout == "Area"]

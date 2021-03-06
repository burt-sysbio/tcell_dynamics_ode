import numpy as np
from src.modules.exp import Sim, Simlist
from src.analysis.antigen_eff.exp2_const_ag_chr.parameters import d
from src.modules.models import vir_model_gamma, vir_model_ode, vir_model_const
import src.modules.proc as proc
import src.modules.pl as pl
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing


# init model
vir_model = vir_model_const
time = np.arange(0, 50, 0.01)
pname1 = "r_chronic"
pname2 = "vir_load"

# init models
sim = Sim(name = "il2", params = d, time = time, virus_model= vir_model)

res = 30
inputs = proc.get_2dinputs(prange1=(0,50), prange2=(0,1.0), res = res)

#for input in inputs:
#     p1,p2 = input
     #sim.params[pname1] = p1
     #sim.params[pname2] = p2
#     print(p1,p2)
     #cells, molecules = sim.run_sim()
     #pl.plot_timecourse(cells, cells = ["teff"])
     #plt.show()

n_cor = multiprocessing.cpu_count()
params = [sim, pname1, pname2]
outputs = Parallel(n_jobs=n_cor)(delayed(proc.get_2doutput)(i, *params) for i in inputs)
#outputs = [proc.get_2doutput(i, *params) for i in inputs]
out = proc.get_2dscan(outputs)

readouts = ["Area", "Peak", "Peaktime"]
df_list = [out]
for r in readouts:
    for df in df_list:
        pl.plot_heatmap(df, "norm_min", r, cmap = "Reds", log_color = True, log_axes= False)
        plt.show()

out2 = out.loc[out.cell == "teff"]
out2 = out2.loc[out2.readout == "Area"]

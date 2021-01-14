import numpy as np
from src.modules.exp import Sim, Simlist
from src.analysis.antigen_eff.exp3_chr_fb.parameters import d
from src.modules.models import vir_model_gamma, vir_model_ode, vir_model_const
import src.modules.proc as proc
import src.modules.pl as pl
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing


# init model
vir_model = vir_model_const
time = np.arange(0, 50, 0.01)
d["vir_load"] = 0.3
d["r_chronic"] = 2
pname1 = "pos_fb_chr"
pname2 = "neg_fb_chr"

# init models
sim = Sim(name = "il2", params = d, time = time, virus_model= vir_model)
cells, molecules = sim.run_sim()
from src.modules.readout_module import check_criteria2
readouts = proc.get_readouts(cells, check_criteria2)

res = 30
rangefun = np.geomspace
inputs = proc.get_2dinputs(prange1=(1,10), prange2=(0.1,1.0), res = res, rangefun=rangefun)


n_cor = multiprocessing.cpu_count()
params = [sim, pname1, pname2]
outputs = Parallel(n_jobs=n_cor)(delayed(proc.get_2doutput)(i, *params) for i in inputs)
out = proc.get_2dscan(outputs)

readouts = ["Area"]
df_list = [out]
for r in readouts:
    for df in df_list:
        pl.plot_heatmap(df, "norm_min", r, vmin = 0.1, vmax = 2,
                        cmap = "Reds", log_color = True, log_axes= True)
        plt.show()

out2 = out.loc[out.cell == "teff"]
out2 = out2.loc[out2.readout == "Area"]

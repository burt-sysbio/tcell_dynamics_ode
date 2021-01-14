import numpy as np
from src.modules.exp import Sim, Simlist
from src.analysis.antigen_eff.exp3_chr_fb.parameters import d
from src.modules.models import vir_model_const
import src.modules.pl as pl
from src.modules.proc import get_readouts
from src.modules.readout_module import check_criteria2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# init model
name = "il2"
vir_model = vir_model_const
time = np.arange(0, 10, 0.01)

# create parameter list
names = ["lo",  "mid", "hi"]
d_list = [dict(d) for i in range(len(names))]
vir_loads = [0.1, 0.2, 0.3]

assert len(vir_loads) == len(names)

# add different viral loads
for dic, v in zip(d_list, vir_loads):
    dic["vir_load"] = v
    dic["r_chronic"] = 0

# second dict list where r chronic is active
d_list2 = [dict(dic) for dic in d_list]
for dic in d_list2:
    dic["r_chronic"] = 10

# create simulation list
sim_list = [Sim(name = name, params = dic, time = time, virus_model = vir_model) for dic in d_list]
sim_list2 = [Sim(name = name, params = dic, time = time, virus_model = vir_model) for dic in d_list2]

# run model
id = "ag conc."
exp1 = Simlist(sim_list, names, idtype= id)
cells, molecules = exp1.run_sim()
cells["r_chronic"] = "0"

exp2 = Simlist(sim_list2, names, idtype= id)
cells2, molecules2 = exp2.run_sim()
cells2["r_chronic"] = "10"

cells = pd.concat([cells,cells2])

pl.plot_timecourse(cells, cells = ["teff", "tchronic"], style = "r_chronic", palette= ["grey", "grey"],
                   col = "ag conc.", row = "cell", hue= "r_chronic")
plt.show()
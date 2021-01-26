from src.analysis.fig4.parameters import d
from src.modules.exp_data_model import Sim
import seaborn as sns
from src.modules.virus_models import vir_model_SIR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set(context = "poster", style = "ticks")

# load fit result
fit_name = "20210119_fit"
fit_date = "2021-01-19"

time = np.linspace(0,80,300)
sim = Sim(time = time, name = fit_name, params = d, virus_model=vir_model_SIR)
#sim.set_fit_params(fit_date, fit_name)

sim.params["SIR_r0"] = 1.5
sim.name = "r1.5"
cells1, molecules1 = sim.run_sim()

sim.params["SIR_r0"] = 3
sim.name = "r3"
cells2, molecules2 = sim.run_sim()

sim.params["SIR_r0"] = 5
sim.name = "r5"
cells3, molecules3 = sim.run_sim()


cells = pd.concat([cells1, cells2, cells3])
molecules = pd.concat([molecules1, molecules2, molecules3])

cells = cells.loc[cells.cell.isin(["Precursors", "Th1_eff", "Tfh_eff", "Tr1_all", "Tfh_chr"])]
g = sns.relplot(data = cells, x = "time", y = "value", hue = "name",
                 col = "cell", kind = "line", col_wrap= 3)
g.set(yscale = "log", ylim = (1e-1, None))
plt.show()

# make a model that combines virus and cell dynamics
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
sns.set(context = "poster", style = "ticks")

def fb_pos(x, K):
    out = (x ** 3 * K ** 3) / x ** 3
    return out


def fb_neg(x, K):
    out = K ** 3 / (x ** 3 + K ** 3)
    return out

class sir_model():
    def __init__(self,
                 parameters,
                 time = np.linspace(0,30,100),
                 name = None,
                 celltypes = ["vir", "eff", "chr"]):
        self.params = parameters
        self.celltypes = celltypes
        self.time = time
        self.name = name

    def init_model(self):
        y0 = [1,1,0]
        return y0

    def run_model(self):
        y0 = self.init_model()
        d = self.params
        time = self.time
        celltypes = self.celltypes
        state = odeint(vir_ode, y0, time, args = (d,))
        state = pd.DataFrame(state, columns = celltypes)
        state["time"] = time
        state["tot"] = state.chr + state.eff
        if self.name is not None:
            state["name"] = self.name

        return state

    def tidy_output(self):
        state = self.run_model()
        state = pd.melt(state, var_name= "cell", id_vars=["time", "name"])
        return state


def vir_ode(s, time, d):
    vir, cell_eff, cell_chr = s
    ds1 = vir* (d["vir_growth"] - d["vir_death"] * time)
    ds2 = cell_eff * (d["prolif"] -
                  d["r_chronic"] * fb_pos(vir, d["k_chronic"]) -
                  d["r_death"] * fb_neg(vir, d["k_death"]))
    ds3 = cell_eff * d["r_chronic"] * fb_pos(vir, d["k_chronic"])

    return[ds1, ds2, ds3]


p_acute = {
    "vir_growth" : 3,
    "vir_death" : 1,
    "prolif" : 1,
    "r_chronic" : 0,
    "k_chronic" : 1,
    "r_death" : 2,
    "k_death" : 1,
}

p_chronic = dict(p_acute)
p_chronic["r_chronic"] = 0.5

sim1 = sir_model(p_acute, name = "acute")
sim2 = sir_model(p_chronic, name = "chronic")
df1 = sim1.tidy_output()
df2 = sim2.tidy_output()

df_res = pd.concat([df1, df2])

g = sns.relplot(data = df_res, x = "time", y = "value", row = "cell", col = "name", kind = "line")
g.set(yscale = "log", ylim = (1e-1, None))
plt.show()
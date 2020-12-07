# make a model that combines virus and cell dynamics
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def fb_pos(x, K):
    out = (x ** 3 * K ** 3) / x ** 3
    return out


def fb_neg(x, K):
    out = K ** 3 / (x ** 3 + K ** 3)
    return out


def vir_ode(s, time, d):
    vir, cell_eff, cell_chr = s
    ds1 = vir* (d["vir_growth"] - d["vir_death"] * time)
    ds2 = cell_eff * (d["prolif"] -
                  d["r_chronic"] * fb_pos(vir, d["k_chronic"]) -
                  d["r_death"] * fb_neg(vir, d["k_death"]))
    ds3 = cell_eff * d["r_chronic"] * fb_pos(vir, d["k_chronic"])

    return[ds1, ds2, ds3]


params = {
    "vir_growth" : 5,
    "vir_death" :1,
    "prolif" : 1,
    "r_chronic" : 0.5,
    "k_chronic" : 1,
    "r_death" : 2,
    "k_death" : 1,
}

y0 = [1, 1, 0]
time = np.linspace(0,20,100)
res = odeint(vir_ode, y0, time, args = (params,))

df_res = pd.DataFrame(res, columns= ["vir", "eff", "chr"])
df_res["tot"] = df_res.eff + df_res.chr
df_res["time"] = time

df_res = pd.melt(df_res, var_name= "cell", id_vars="time")

g = sns.relplot(data = df_res, x = "time", y = "value", row = "cell",
                facet_kws= {"sharey" : False})
plt.show()
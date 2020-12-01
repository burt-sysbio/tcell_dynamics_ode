# make a model that combines virus and cell dynamics
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from itertools import product
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
                 time,
                 name = None,
                 celltypes = ["vir", "eff", "chr"]):
        self.params = parameters
        self.celltypes = celltypes
        self.time = time
        self.name = name
        self.state = None
        self.state_tidy = None

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


    def get_readouts(self):
        state = self.tidy_output()

        rmax = state.groupby(by=["cell"]).max()
        # get only the max of total cells
        rmax1 = rmax.loc["chr", "value"]
        rmax2 = rmax.loc["eff", "value"]
        rmax3 = rmax.loc["tot", "value"]
        return np.array([rmax1, rmax2, rmax3])


def pscan(res, pname, pmin, pmax):
    arr = np.linspace(pmin, pmax, 50)

    reads = []
    for a in arr:
        res.params[pname] = a
        rmax = res.get_readouts()
        reads.append(rmax)

    df = pd.DataFrame(reads, columns= ["chr", "eff", "tot"])
    df["val"] = arr
    df = pd.melt(df, id_vars = "val", value_name= "peak_height", var_name="cell")
    df["name"] = res.name
    return df


def gridfun(prod, sim, pname1, pname2):
    sim.params[pname1] = prod[0]
    sim.params[pname2] = prod[1]
    rmax = sim.get_readouts()
    rmax = rmax[0]

    return np.array([rmax,rmax])


def pscan2d(sim, pname1, pname2, p1 = (0,1), p2 = (0,1), res = 3):
    arr1 = np.linspace(p1[0], p1[1], res)
    arr2 = np.linspace(p2[0], p2[1], res)

    prod = product(arr1,arr2)
    # get readouts for each cart. prod. of param comb.
    out = [gridfun(p, sim, pname1, pname2) for p in prod]
    out = np.asarray(out)
    max_grid = out[:,0]

    max_grid = max_grid.reshape((res, res))
    return arr1, arr2, max_grid


def vir_ode(s, time, d):
    vir, cell_eff, cell_chr = s
    ds1 = vir * (d["vir_growth"] - d["vir_death"] * time)
    ds2 = cell_eff * (d["prolif"] -
                  d["r_chronic"] * fb_pos(vir, d["k_chronic"]) -
                  d["r_death"] * fb_neg(vir, d["k_death"]))
    ds3 = cell_eff * d["r_chronic"] * fb_pos(vir, d["k_chronic"])

    return[ds1, ds2, ds3]


p_acute = {
    "vir_growth" : 2,
    "vir_death" : 1,
    "prolif" : 1,
    "r_chronic" : 0,
    "k_chronic" : 1,
    "r_death" : 2,
    "k_death" : 1,
}

p_chronic = dict(p_acute)
p_chronic["r_chronic"] = 0.4

# timecourse for both model versions
t_short = np.linspace(0,10,100)
sim1 = sir_model(p_acute, time = t_short, name = "acute")
sim2 = sir_model(p_chronic, time = t_short, name = "chronic")
df1 = sim1.tidy_output()
df2 = sim2.tidy_output()
df_res = pd.concat([df1, df2])

g = sns.relplot(data = df_res, x = "time", y = "value", col = "cell", hue = "name", kind = "line")
g.set(yscale = "log", ylim = (1, None), ylabel = "n cells")
g.set_titles("{col_name}")
plt.show()

g.savefig("../../figures/sir_model/sir_timecourse.pdf")
# parameter scan
t_long = np.linspace(0,20,100)
sim1 = sir_model(p_acute, time = t_long, name = "acute")
sim2 = sir_model(p_chronic, time = t_long, name = "chronic")

reads1 = pscan(sim1, "vir_growth", 0, 3)
reads2 = pscan(sim2, "vir_growth", 0, 3)

df_pscan = pd.concat([reads1, reads2])
df_pscan = df_pscan[df_pscan.cell == "tot"]
g = sns.relplot(data = df_pscan,
                x = "val",
                y = "peak_height",
                hue = "name",
                kind = "line")

g.set(yscale = "log", xlabel = "viral load")
plt.show()

g.savefig("../../figures/sir_model/pscan_viral_load.pdf")
#x,y,z = pscan2d(sim1, "r_chronic", "vir_growth", (0,2), (0,10), res = 3)

#z = z[:-1, :-1]

#fig, ax = plt.subplots()
#c = ax.pcolormesh(x, y, z, cmap='Blues', norm=matplotlib.colors.LogNorm())
#fig.colorbar(c, ax=ax)
#plt.show()
"""
linear chain naive --> eff cells. eff homeostasis regulated by IL2 and Timer
IL2 secreted by naive cells, consumed by eff cells
analyze delays and pos, neg feedback
"""

import numpy as np
from scipy.integrate import odeint
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from readout_module import get_area

sns.set(style="ticks", context="poster")


def diff_chain(state, influx, beta, death, n_div):
    """
    Parameters
    ----------
    state : arr
        arr to intermediate states.
    influx : float
        flux into first state of chain.
    beta : float
        DESCRIPTION.
    death : float
        DESCRIPTION.

    Returns
    -------
    dt_state : array
        DESCRIPTION.

    """
    dt_state = np.zeros_like(state)
    for i in range(len(state)):
        if i == 0:
            dt_state[i] = influx - (beta + death) * state[i] + 2 * n_div * beta * state[-1]
        else:
            dt_state[i] = beta * state[i - 1] - (beta + death) * state[i]

    return dt_state


def prob_fb(x, fc, EC50, hill=3):
    out = (fc * x ** hill + EC50 ** hill) / (x ** hill + EC50 ** hill)
    return out


def pos_fb(x, EC50, hill=3):
    out = x**hill / (x**hill + EC50**hill)
    return out


def simple_chain(state, time, d):
    # split states
    naive = state[:d["alpha_naive"]]
    eff = state[d["alpha_naive"]:-2]

    myc = state[-1]
    il2 = state[-2]

    dt_myc = -d["deg_myc"] * myc
    # compute influx into next chain

    influx_naive = 0
    n_eff = np.sum(eff)
    n_naive = np.sum(naive)
    dt_il2 = d["r_il2"]*n_naive - d["up_il2"]*n_eff*il2

    # algebraic relations timer and IL2
    signal_il2 = pos_fb(il2, d["EC50_il2"]) if il2 >= 0 else 0
    signal_myc = pos_fb(myc, d["EC50_myc"]) if myc >= 0 else 0
    signal_il2_myc = np.sqrt(signal_il2*signal_myc)

    if d["mode"] == "il2" :
        signal = signal_il2
    elif d["mode"] == "timer" :
        signal = signal_myc
    else:
        signal = signal_il2_myc
    # prolif rate is comb of feedback and il2+myc signals
    beta_p = d["beta_p"] * signal*prob_fb(n_eff, d["fb_strength"], d["fb_EC50"])
    # algebraic relations feedback
    beta = d["beta"]
    influx_eff = naive[-1] * beta
    death = d["d_eff"]
    dt_naive = diff_chain(naive, influx_naive, beta, d["d_naive"], d["div_naive"])
    dt_eff = diff_chain(eff, influx_eff, beta_p, death, d["div_eff"])

    dt_state = np.concatenate((dt_naive, dt_eff, [dt_il2], [dt_myc]))

    return dt_state


def init_model(d):
    # +2 for myc and il2
    y0 = np.zeros(d["alpha_naive"] + d["alpha_eff"] +  2)
    y0[0] = 1
    # set myc conc.
    y0[-1] = 1
    # set il2 conc: leave out should be 0
    return y0


def run_model(time, d):
    y0 = init_model(d)
    state = odeint(simple_chain, y0, time, args=(d,))
    return state


def get_cells(state, time, d):
    naive = state[:, :d["alpha_naive"]]
    naive = np.sum(naive, axis=1)

    # -2 because of IL2 and myc
    eff = state[:, d["alpha_naive"]:-2]
    eff = np.sum(eff, axis=1)

    cells = np.stack([naive, eff], axis=-1)
    df = pd.DataFrame(data=cells, columns=["naive", "eff"])
    df["time"] = time
    return df

def get_molecules(state, time, d):
    state = state[:,-2:]
    df = pd.DataFrame(data = state, columns=["IL2", "Myc"])
    df["time"] = time
    df = df.melt(id_vars= ["time"], value_name= "conc. (a.u.)", var_name="Molecule")
    return df

# parameters
mode = "myc"
d1 = {
     "alpha_naive" : 1,
     "beta" : 1,
     "div_naive" : 0,
     "div_eff" : 1,
     "alpha_eff" : 10,
     "beta_p" : 30,
     "d_naive": 0,
     "d_eff" : 1.0,
     "fb_strength" : 1,
     "fb_EC50" : 0.1,
     "EC50_myc" : 0.5,
     "deg_myc" : 0.5,
    "r_il2" : 1,
    "up_il2" : 1,
    "EC50_il2" : 0.3,
    "mode" : mode,
     }


d2 =dict(d1)
d2["alpha_naive"] = 2
d2["beta"] = 2

d3 = dict(d1)
d3["alpha_naive"] = 50
d3["beta"] = 50

fb_pos = 10.0
fb_neg = 0.1

d_fb_pos1 = dict(d1)
d_fb_pos2 = dict(d2)
d_fb_pos3 = dict(d3)
d_fb_neg1 = dict(d1)
d_fb_neg2 = dict(d2)
d_fb_neg3 = dict(d3)

d_fb_off = [d1,d2,d3]
d_fb_pos = [d_fb_pos1, d_fb_pos2, d_fb_pos3]
d_fb_neg = [d_fb_neg1, d_fb_neg2, d_fb_neg3]

for d, f in zip(d_fb_pos, d_fb_neg):
    d["fb_strength"] = fb_pos
    f["fb_strength"] = fb_neg

labels = ["No Delay", "Small Delay", "Strong Delay"]
feedbacks = ["No Feedback", "Neg Feedback", "Pos Feedback"]
dict_list = [d_fb_off, d_fb_neg, d_fb_pos]

time = np.arange(0,7, 0.01)

modes = ["timer", "il2", "timer_il2"]

df_list = []
df2_list = []

# run pipeline for all models
for mode in modes:
    cells_list = []
    molecules_list = []
    # run pipeline for all feedback and all delay types
    for dic, feedback in zip(dict_list, feedbacks):
        for d, label in zip(dic, labels):
            d["mode"] = mode
            state = run_model(time, d)
            cells = get_cells(state, time, d)
            molecules = get_molecules(state, time, d)
            cells = cells[["time", "eff"]]
            cells["name"] = label
            molecules["name"] = label
            cells["feedback"] = feedback
            molecules["feedback"] = feedback
            cells_list.append(cells)
            molecules_list.append(molecules)

    # combine
    cells = pd.concat(cells_list)
    molecules = pd.concat(molecules_list)

    # normalize to no delay maximum for all feedback conditions
    maxima = cells.groupby(["name", "feedback"])["eff"].max()
    maxima = maxima["No Delay"]
    cells = pd.merge(cells, maxima, how = "left", on = "feedback")
    cells["norm"] = cells.eff_x / cells.eff_y
    cells["mode"] = mode
    df_list.append(cells)

    # systematic feedback analysis
    fb_arr = np.geomspace(1.0,10,50)
    fb_1 = []
    fb_2 = []
    fb_3 = []
    fb_list = [fb_1, fb_2, fb_3]

    # for all delay types vary feedback strength
    for d, label, l in zip(dic, labels, fb_list):
        # for each fb value get resposne size
        for fb in fb_arr:
            d["fb_strength"] = fb
            state = run_model(time, d)
            cells = get_cells(state, time, d)
            cells = cells[["time", "eff"]]
            area = get_area(time, cells.eff)

            l.append(area)

    df = pd.DataFrame({"no_delay" : fb_list[0],
                       "small_delay" : fb_list[1],
                       "high_delay" : fb_list[2],
                      "fb_fc" : fb_arr})

    df = df.melt(id_vars= "fb_fc", value_name= "population response", var_name = "delay")
    df["mode"] = mode
    df2_list.append(df)

cells = pd.concat(df_list)
df_fb = pd.concat(df2_list)

g = sns.relplot(data = cells, x = "time", y = "norm",
                row = "feedback",
                col = "mode",
                hue = "name",
                kind = "line", aspect = 0.9, palette= "Blues",
                facet_kws={"sharey":True, "margin_titles" : True})

g.set_titles(col_template="{col_name}", row_template= "{row_name}")
g.set(ylabel = "effector cells (a.u.)",
      xlabel = "time (a.u.)")
plt.show()
g.savefig("../figures/fig3/fb_delay_timecourse.pdf")
g.savefig("../figures/fig3/fb_delay_timecourse.svg")

g = sns.relplot(data = df_fb, x = "fb_fc", y = "population response",
                hue = "delay",
                col = "mode",
                kind = "line",
                palette= "Blues")
g.set(xscale = "log", ylim = (0, 30))
plt.show()

#g.savefig("../figures/fig3/fb_delay_analysis.pdf")
#g.savefig("../figures/fig3/fb_delay_analysis.svg")

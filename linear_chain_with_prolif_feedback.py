#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:38:50 2020

@author: burt
one simple diff x--> y with differentiation
"""

import numpy as np
from scipy.integrate import odeint
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(style = "ticks", context = "poster")

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
            dt_state[i] = influx - (beta+death)*state[i] + 2*n_div*beta*state[-1]
        else:
            dt_state[i] = beta*state[i-1] - (beta+death)*state[i]
    
    return dt_state


def prob_fb(x, fc, EC50, hill = 3):
    out = (fc*x**hill + EC50**hill) / (x**hill + EC50**hill)
    return out


def pos_fb(x, EC50, hill = 3):
    out = x**hill / (x**hill + EC50**hill)
    return out


def simple_chain(state, time, d):
    # split states
    naive = state[:d["alpha_naive"]]
    eff = state[d["alpha_naive"]:-1]
    
    myc = state[-1]
    dt_myc = -d["deg_myc"]*myc   
    # compute influx into next chain


    influx_naive = 0
    n_eff = np.sum(eff)

    # algebraic relations timer
    beta_p = d["beta_p"]*prob_fb(n_eff, d["fb_strength"], d["fb_EC50"])
    # algebraic relations feedback
    beta = d["beta"]
    influx_eff = naive[-1]*beta
    death = d["d_eff"]*time
    dt_naive = diff_chain(naive, influx_naive, beta, d["d_naive"], d["div_naive"])
    dt_eff = diff_chain(eff, influx_eff, beta_p, death, d["div_eff"])

    dt_state = np.concatenate((dt_naive, dt_eff, [dt_myc]))
    
    return dt_state


def init_model(d):
    # +1 for myc
    y0 = np.zeros(d["alpha_naive"]+d["alpha_eff"]+1)
    y0[0] = 1
    # set myc conc.
    y0[-1] = 1
    return y0


def run_model(time, d):
    y0 = init_model(d)
    state = odeint(simple_chain, y0, time, args = (d,))
    return state




def get_cells(state, time, d):

    naive = state[:,:d["alpha_naive"]]
    naive = np.sum(naive, axis = 1)
    
    eff = state[:,d["alpha_naive"]:-1] 
    eff = np.sum(eff, axis = 1)
    
    cells = np.stack([naive, eff], axis = -1)
    df = pd.DataFrame(data = cells, columns= ["naive", "eff"])
    df["time"] = time
    return df

d = {
     "alpha_naive" : 1,
     "beta" : 1,
     "div_naive" : 0,
     "div_eff" : 1,
     "alpha_eff" : 10,
     "beta_p" : 10,
     "d_naive": 0,
     "d_eff" : 1.0,
     "fb_strength" : 1,
     "fb_EC50" : 0.1,
     "EC50_myc" : 0.5,
     "deg_myc" : 0.1,
     }

d2 =dict(d)
d2["alpha_naive"] = 2
d2["beta"] = 2

d3 = dict(d)
d3["alpha_naive"] = 10
d3["beta"] = 10

fb_stren = 2.5


d4 = dict(d)
d5 = dict(d2)
d6 = dict(d3)

for dic in [d4,d5,d6]:
    dic["fb_strength"] = fb_stren

delays = [1, 0.5, 0.1]
delays = 2*delays
feedbacks = 3*["feedback off"]+3*["feedback on"]
dicts = [d,d2,d3,d4,d5,d6]
time = np.arange(0,5, 0.01)

df_list = []
for dic, delay, fb in zip(dicts, delays, feedbacks):
    state = run_model(time, dic)
    cells = get_cells(state, time, dic)
    cells = pd.melt(cells, id_vars = ["time"], value_name= "cells", var_name= "celltype")
    cells = cells[cells.celltype == "eff"]
    cells["delay"] = delay
    cells["feedback"] = fb
    df_list.append(cells)

df = pd.concat(df_list)
g = sns.relplot(data = df, x = "time", y = "cells", col = "feedback", hue = "delay", kind = "line",
                legend = False, aspect = 1.0)


g.set_titles("{col_name}")
g.set(ylabel = "cell dens. norm.", xlabel = "time (a.u.)")
plt.show()
g.savefig("plot_delay_fb_tc.pdf")
g.savefig("plot_delay_fb_tc.svg")
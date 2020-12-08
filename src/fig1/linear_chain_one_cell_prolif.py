#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 16:38:50 2020

@author: burt
simple model only one celltype that proliferates
"""

import numpy as np
from scipy.integrate import odeint
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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

    eff = state[:-1] 
    myc = state[-1]
    dt_myc = -d["deg_myc"]*myc   
    # compute influx into next chain
    n_eff = np.sum(eff)

    # algebraic relations timer
    beta = d["beta"]*prob_fb(n_eff, d["fb_strength"], d["fb_EC50"])
    # algebraic relations feedback
    influx_eff = 0
    death = d["d_eff"]*time
    dt_eff = diff_chain(eff, influx_eff, beta, death, d["div_eff"])

    dt_state = np.concatenate((dt_eff, [dt_myc]))
    
    return dt_state


def init_model(d):
    # +1 for myc
    y0 = np.zeros(d["alpha"]+1)
    y0[0] = 1
    # set myc conc.
    y0[-1] = 1
    return y0


def run_model(time, d):
    y0 = init_model(d)
    state = odeint(simple_chain, y0, time, args = (d,))
    return state




def get_cells(state, time, d):


    eff = state[:,:-1] 
    eff = np.sum(eff, axis = 1)
    
    df = pd.DataFrame(data = eff, columns= ["cells"])
    df["time"] = time
    return df

d = {
     "alpha" : 10,
     "beta" : 10,
     "div_eff" : 1,
     "d_naive": 0,
     "d_eff" : 0,
     "fb_strength" : 1,
     "fb_EC50" : 10.0,
     "EC50_myc" : 2.0,
     "deg_myc" : 0.1,
     }


d2 =dict(d)
d2["alpha"] = 1
d2["beta"] = 1

d3 = dict(d)
d3["fb_strength"] = 1.5
d4 = dict(d2)
d4["fb_strength"] = 1.5

dicts = [d,d2]
names = ["delay no fb", "no delay no fb"]
time = np.arange(0,3, 0.01)

df_list = []
for dic, name in zip(dicts, names):
    state = run_model(time, dic)
    cells = get_cells(state, time, dic)
    cells["norm"] = cells.cells / cells.cells.max()
    cells["cond."] = name
    df_list.append(cells)

df = pd.concat(df_list)

g = sns.relplot(data = df, hue= "cond.", x = "time", y = "cells", kind = "line")
plt.show()
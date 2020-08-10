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
    beta_p = d["beta_p"]*pos_fb(myc, d["EC50_myc"])
    # algebraic relations feedback
    beta = d["beta"]*prob_fb(n_eff, d["fb_strength"], d["fb_EC50"])
    influx_eff = naive[-1]*beta

    dt_naive = diff_chain(naive, influx_naive, beta, d["d_naive"], d["div_naive"])
    dt_eff = diff_chain(eff, influx_eff, beta_p, d["d_eff"], d["div_eff"])

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
     "alpha_naive" : 10,
     "beta" : 10,
     "div_naive" : 0,
     "div_eff" : 1,
     "alpha_eff" : 10,
     "beta_p" : 0,
     "d_naive": 0,
     "d_eff" : 0,
     "fb_strength" : 100,
     "fb_EC50" : 0.5,
     "EC50_myc" : 0.5,
     "deg_myc" : 0.1,
     }

time = np.arange(0,10, 0.01)
state = run_model(time, d)

cells = get_cells(state, time, d)
cells_tidy = pd.melt(cells, id_vars = ["time"])
g = sns.relplot(data = cells_tidy, x = "time", y = "value", hue = "variable", kind = "line")
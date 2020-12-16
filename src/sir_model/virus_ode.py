# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 18:25:59 2020

@author: Philipp
"""


from scipy.integrate import odeint
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def vir_ode(state, t, d):
    x = state[0]
    y = state[1]
    v = state[2]
    
    dx = d["influx"] - d["d_x"]*x - d["b_x"]*v*x
    dy = d["b_x"]*v*x - d["d_y"]*y
    dv = d["b_v"]*y - d["d_v"]*v
    
    return [dx,dy,dv]

def ode2(state, t, d):
    death = d["b"]*t
    dx = d["a"]-death*state
    return dx

params = {
    "influx" : 0, ## influx uninfected
    "d_x" : 0, ## death uninfected
    "d_y" : 1, ## death infected
    "d_v" : 0.5, ## death virus
    "b_x" : 10, ## vir infects uninfected cells
    "b_v" : 100, ## vir production by infected cells
    }


params = {
    "a" : 1000.0,
    "b" : 10.0}

y0 = 0
t = np.linspace(0,10,200)
s = odeint(ode2, y0, t, args = (params,))

plt.plot(t, s)
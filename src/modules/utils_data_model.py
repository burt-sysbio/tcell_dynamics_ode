#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:13:15 2020

@author: burt
"""
import numpy as np
import pandas as pd
from scipy import constants
import src.modules.readout_module as readouts


def get_rel_cells(cells):
    """
    cells needs to be tidy data frame
    add new column to cells df which gives percentage of total cells
    """
    #add total cells and compute relative cell fractions
    tot_cells = cells[cells["celltype"] == "Total_CD4"].rename(columns = {"value" : "total"})
    tot_cells = tot_cells[["time", "total", "Infection"]]
    cells = pd.merge(cells, tot_cells, how = "left", on = ["time", "Infection"])
    cells.total = (cells.value / cells.total)*100
    
    return cells


def get_readouts(time, cells):
    """
    get readouts from state array
    """
    peak = readouts.get_peak_height(time, cells)
    area = readouts.get_area(time, cells)
    tau = readouts.get_peaktime(time, cells)
    decay = readouts.get_duration(time, cells)
    
    reads = [peak, area, tau, decay]
    read_names = ["Peak Height", "Response Size", "Peak Time", "Decay"]
    data = {"readout" : read_names, "read_val" : reads}
    reads_df = pd.DataFrame(data = data)
    
    return reads_df


def get_cytos(df_tidy, keep_cytos = ["IL2", "IL10"]):
    # use again when I use cytos
    cytos = filter_cells(df_tidy, keep_cytos)
    cytos = cytos.rename(columns = {"celltype" : "cytokine"})  
    cytos["conc_pM"] = get_conc(cytos.value)    
    return cytos


def get_conc(cyto):
    # convert cytokine from molecules to picoMolar
    N = constants.N_A
    # assume lymph node volume is one microlitre
    VOL = 1e-6
    PICOMOL = 1e12

    cyto = cyto*PICOMOL / (VOL*N)
    return cyto


def run_pipeline(sim):
    """
    run both armstrong and cl13 simulation with same parameters
    output cells and cytokines separated

    """
    # run simulation with constant ag for no ag and high ag conditions
    sim.params["vir_load"] = 0
    sim.name = "Arm"
    cells_arm, mols_arm = sim.run_sim()
    
    # sum arm and cl13 sim
    sim.params["vir_load"] = 1
    sim.name = "Cl13"
    cells_cl13, mols_cl13 = sim.run_sim()

    cells = pd.concat([cells_arm, cells_cl13])
    molecules = pd.concat([mols_arm, mols_cl13])
    return cells, molecules


def get_residuals(cells, data, timepoints=[9, 30, 60]):
    """
    formerly called transform
    objective function computes difference between data and simulation
    Parameters
    ----------
    sim : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    timepoints : TYPE, optional
        DESCRIPTION. The default is [9, 30, 60].

    Returns
    -------
    resid : arr
        array of data-simulation differences (residuals)

    """
    # get simulation data at these time points
    cells = cells[cells.time.isin(timepoints)]

    # only focus on Tfh and non Tfh
    cells = cells[["Tfh_all", "nonTfh"]]

    # convert to same format as data
    cells = cells.melt()
    resid = (data.value.values - cells.value.values) / data.eps.values
    return resid


def fit_fun(params, sim, data1, data2):
    """
    function to be minimized
    run simulation for same parameter set using data1 and data2
    """
    a = params.valuesdict()

    # adjust parammeters
    for key, val in a.items():
        sim.params[key] = val

    start = 0
    stop = 80
    res = 321
    time = np.linspace(start, stop, res)
    sim.time = time

    # run model arm
    sim.params["vir_load"] = 0
    cells1, molecules1 = sim.run_sim()
    resid1 = get_residuals(cells1, data1)

    # run model cl13
    sim.params["vir_load"] = 1
    cells2, molecules2 = sim.run_sim()
    resid2 = get_residuals(cells2, data2)

    # array of residuals needs to consist of residuals for Arm and Cl13
    resid = np.concatenate((resid1, resid2))

    sim.reset()
    return resid

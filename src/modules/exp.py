# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import src.modules.models as model
import numpy as np
from scipy.integrate import odeint
import pandas as pd
import warnings
from src.modules.models_helper import prolif_wrapper

class Sim:
    """
    model simulation class
    initialize with a name (str), mode (model specific)
    params (dict), time to sim (arr) and core(model specific)
    """
    def __init__(self,
                 name,
                 params,
                 time,
                 virus_model,
                 model = model.th_cell_diff,
                 core = model.diff_core,
                 n_molecules = 2):


        # type of homeostasis model
        self.name = name
        self.prolif_model = prolif_wrapper
        self.params = dict(params)
        self.time = time
        self.core = core
        self.model = model
        # initialize and interpolate virus model for given time and parameters
        self.virus_model = virus_model
        self.n_molecules = n_molecules # myc and il2_ex

    def init_model(self):
        """
        set initial conditions for ODE solver
        """
        y0 = np.zeros(self.params["alpha"]+1*self.params["alpha_p"]+ self.n_molecules)
        y0[0] = self.params["initial_cells"]
        
        # init myc concentration
        y0[-1] = 1.
        y0[-2] = self.params["il2_y0"]
        
        return y0

    def tidy_cells(self, cells):
        """
        summarize effector cells and naive cells
        """
        alpha = self.params["alpha"]
        teff = cells[:, alpha:] # myc and il2_ex are at end of array
        teff = np.sum(teff, axis = 1)
        tnaive = np.sum(cells, axis = 1) - teff
        
        cells = np.stack((tnaive, teff), axis = -1)
        return cells    

    def run_ode(self, hmax = 0.0):
        """
        run time course simulation
        hmax is maximum step size, needed if ODE behavior changes e.g. due to
        timer thresholds t>tcrit 
        normalize : bool (set true if area should be normalized to 1)
        returns data frame
        """
        if hmax != 0.0 : warnings.warn("warning: hmax is set, might take long to integrate")

        # run sim
        y0 = self.init_model()

        # initialize virus model with given parameters
        vir_model = self.virus_model(self.time, self.params)
        p_model = self.prolif_model(self.name, self.params)
        args = (self.params, p_model, self.core, vir_model)
        state = odeint(self.model, y0, self.time, args = args, hmax = hmax)

        # format output
        n = self.n_molecules
        cells = state[:,:-n]
        molecules = state[:,-n:]

        # get virus model and compute virus at all time points
        virus = vir_model(self.time)

        return cells, molecules, virus

    def run_sim(self, hmax = 0.0):

        cells, molecules, virus = self.run_ode(hmax)

        # tidy cell output
        cells = self.tidy_cells(cells)
        cells = pd.DataFrame(cells, columns = ["tnaive", "teff"])
        cells["time"] = self.time
        cells["name"] = self.name
        cells = pd.melt(cells, id_vars=["time", "name"], var_name="cell")

        # tidy molecules and merge virus data
        molecules = pd.DataFrame(molecules, columns= ["IL2", "MYC"])
        molecules["Virus"] = virus
        molecules["time"] = self.time
        molecules["name"] = self.name
        molecules = pd.melt(molecules, var_name= "cell", id_vars= ["time", "name"])

        return cells, molecules

    def set_params(self, pnames, arr):
        for pname, val in zip(pnames, arr):
            self.params[pname] = val




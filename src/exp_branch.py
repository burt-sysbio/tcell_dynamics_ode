#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:31:57 2020

@author: burt
module for branching processes
"""
import numpy as np
from scipy.integrate import odeint
import tcell_model.readout_module as readouts
import tcell_model.models_branch as model
import pandas as pd

class Simulation:
        
    def __init__(self, name, mode, parameters, time, core):
        self.name = name
        self.mode = mode
        self.parameters = dict(parameters)
        self.time = time
        self.core = core
        if self.core.__name__ == "branch_precursor":
            assert "p1_def" in self.parameters
        else:
            assert "p1_def" not in self.parameters
        
        
    def init_model(self):
        # need additional slot for myc ode
        myc = 1
        d = self.parameters
        
        n_states = -1 if self.core.__name__ == "branch_competetive" else 1
        y0 = np.zeros(d["alpha1"]+d["alpha1_p"]+d["alpha2"]+d["alpha2_p"]+n_states+myc)
       
        y0[0] = 1.       
        # init myc concentration
        y0[-1] = 1.
        
        return y0
    
    
    def get_cells(self, state):
        d = self.parameters
        n_states = 0 if self.core.__name__ == "branch_competetive" else 1

        th1 = state[:, 1:(d["alpha1"]+d["alpha1_p"]+n_states)]
        # add -1 because of myc
        th2 = state[:, (d["alpha1"]+d["alpha1_p"]+n_states):-1]  
        
        th1 = th1[:,-d["alpha1_p"]:]
        th2 = th2[:,-d["alpha2_p"]:]      
        
        th1 = np.sum(th1, axis = 1)
        th2 = np.sum(th2, axis = 1)
        cells = np.stack((th1, th2), axis = -1)
            
        return cells    


    def run_timecourse(self):
        #print("running time course simulation..")
        
        y0 = self.init_model()
        mode = self.mode
        params = dict(self.parameters)
        time = self.time
        core = self.core   

        state = odeint(model.th_cell_branch, y0, time, args = (mode, params, core)) 
        
        cells = self.get_cells(state)
        
        colnames = ["th1", "th2"]
        df = pd.DataFrame(cells, columns = colnames)
       	
        df["time"] = self.time
        df["simulation_name"] = self.name

        #make df tidy for both cell types
        df = pd.melt(df, value_vars = colnames, var_name = "cell_type", value_name = "cells", id_vars = ["time", "simulation_name"])
        df["regulation"] = self.mode.__name__
        return df


    def get_readouts(self):
        state = self.run_timecourse()
        df_th1 = state[state.cell_type == "th1"]
        df_th2 = state[state.cell_type == "th2"]

        df_th1 = self.get_readouts_from_df(df_th1, "th1")
        df_th2 = self.get_readouts_from_df(df_th2, "th2")
        
        df = pd.concat([df_th1, df_th2])

        return df
        
    
    def get_readouts_from_df(self, state, name):
        # get readouts
        peak = readouts.get_peak(state.time, state.cells)
        area = readouts.get_area(state.time, state.cells)
        tau = readouts.get_peaktime(state.time, state.cells)
        decay = readouts.get_decay(state.time, state.cells)     
        # convert readouts to dataframe
        reads = [peak, area, tau, decay]

        read_names = ["peak", "area", "tau", "decay"]
        data = {"readout" : read_names, "readout_val" : reads}
        reads_df = pd.DataFrame(data = data)

        # assign cell type to data frame
        reads_df["cell_type"] = name
        
        return reads_df
    

    def vary_param(self, arr_dict):
        """
        vary parameter and compute readouts for both cells
        provide dict with var names and corresponding arrays to be varied at
        the same time. arrays should have same length
        """
        arr_dict = dict(arr_dict)
        old_parameters = dict(self.parameters)
        readout_list = []
        
        values = list(arr_dict.values())
        keys = list(arr_dict.keys())

        # for all parameters in array dict loop over provided arrays and change params 
        for val in zip(*values):
            for key, v in zip(keys, val):
                self.parameters[key] = v
                #print(key, self.parameters[key])
                
            read = self.get_readouts()

            # this is only one parameter value in case I should ever
            # use more than one param array I would need to change this
            read["param_val"] = val[0]
            readout_list.append(read)
        
        df = pd.concat(readout_list)
        df = df.reset_index(drop = True)
        # set the parameter name to the first variable name. in case of multiple
        # variables this is not true
        df["param_name"] = keys[0]
        df["simulation_name"] = self.name
        df["regulation"] = self.mode.__name__
        
        self.parameters = old_parameters
        
        return df
    
    
    def vary_param_norm(self, arr_dict, norm_val, norm_idx):
        """
        vary parameters and then normalize either to beginning or middle and take log
        """
        arr_dict = dict(arr_dict)
        df = self.vary_param(arr_dict)
        
        # normalize
        df["param_val_norm"] = df["param_val"] / norm_val
        
        # get readouts for only for normalized arr value 
        for key in arr_dict:
            val = arr_dict[key][norm_idx]
            arr_dict[key] = [val]
        df_norm = self.vary_param(arr_dict)
        
        # merge these readouts to original data frame
        df_norm = df_norm[["readout", "readout_val", "cell_type"]]
        df_norm = df_norm.rename(columns = {"readout_val" : "norm_readout"})
        df = df.merge(df_norm, on = ["readout", "cell_type"], how = "left")
        
        # get effect size
        logseries = df["readout_val"]/df["norm_readout"]
        logseries = logseries.astype(float)
        df["effect_size"] = np.log2(logseries)
        #df["simulation_name"] = self.name
        
        return df


    def get_relative_readouts(self, df):

        # need to get copy otherwise view is returned if I index like this
        df_th1 = df[df.cell_type == "th1"].copy()
        df_th2 = df[df.cell_type == "th2"].copy()
        
        # get total area
        # compute the relative readouts (division)
        rel_readout = df_th1.readout_val.values / df_th2.readout_val.values
        df_th1["rel_readout"] = rel_readout
        df_th1 = df_th1.drop(columns = ["cell_type", "readout_val", "norm_readout"])
        
        return df_th1
    
    
    def vary_param_rel(self, arr_dict, norm_val, norm_idx):
        df = self.vary_param_norm(arr_dict, norm_val, norm_idx)
        
        # get readouts of norm condition
        df = df.drop(columns = ["effect_size"])
        df_norm = df.loc[df["norm_readout"] == df["readout_val"]]
        
        # get relative readouts of norm condition
        rel_readouts = self.get_relative_readouts(df_norm)

        # rename to later merge to original df
        rel_readouts = rel_readouts.rename(columns = {"rel_readout" : "rel_readout_norm"})

        #compute relative readouts for whole data frame
        df = self.get_relative_readouts(df)

        # only keep two columns to facilitate merge then merge with original rel readout df
        rel_readouts = rel_readouts[["readout", "rel_readout_norm"]]
        df = pd.merge(df, rel_readouts, how = "left")
        
        # compute effect size
        logseries = df["rel_readout"]/df["rel_readout_norm"]
        logseries = logseries.astype(float)
        df["effect_size"] = np.log2(logseries)
        
        return df
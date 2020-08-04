# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 09:56:14 2020

@author: Philipp
"""
import numpy as np
from scipy.integrate import odeint
import pandas as pd

def prec_model(state, time, d):
    
    myc = state[-1]
    state = state[:-1]      
    i_naive = d["alpha_naive"]
    i_prec = d["alpha_prec"]
    i_th1 = d["alpha_th1"]
    i_tfh = d["alpha_tfh"]
    assert(len(state) == sum([i_naive,i_prec,i_th1,i_tfh]))
    
    # split state arr into different cell types
    naive_arr = state[:i_naive]
    prec_arr = state[i_naive:(i_naive+i_prec)]
    th1_arr = state[(i_naive+i_prec):(i_naive+i_prec+i_th1)]
    tfh_arr = state[(i_naive+i_prec+i_th1):]
    
    # calculate influx
    influx_naive = 0
    influx_prec = naive_arr[-1]*d["beta_naive"]
    influx_th1 = prec_arr[-1]*d["beta_prec"]*d["p_th1"]
    influx_tfh = prec_arr[-1]*d["beta_prec"]*d["p_tfh"]
    
    dt_naive = diff_chain(naive_arr, influx_naive, d["beta_naive"], 0, 0, 0)
    dt_prec = diff_chain(prec_arr, influx_prec, d["beta_prec"], 0, d["p_prec"], d["n_div_prec"])
    dt_th1 = diff_chain(th1_arr, influx_th1, d["beta_p_th1"], d["death_th1"], 1, d["n_div_eff"])
    dt_tfh = diff_chain(tfh_arr, influx_tfh, d["beta_p_tfh"], d["death_tfh"], 1, d["n_div_eff"])

    d_myc = -(1./d["lifetime_myc"])*myc
    dt_state = np.concatenate((dt_naive, dt_prec, dt_th1, dt_tfh, [d_myc]))
    
    return dt_state


def diff_chain(state, influx, beta, death, prob, n_div):
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
    prob : float
        set this for precursors to respective prec. probability
        for effectors to 1 (probability is in influx term)
        and for naive to 0 (naive cells dont proliferate orthogonal to diff)
    n_div : int
        number of divisions.

    Returns
    -------
    dt_state : array
        DESCRIPTION.

    """
    dt_state = np.zeros_like(state)
    for i in range(len(state)):
        if i == 0:
            dt_state[i] = influx - (beta-death)*state[i] + 2*n_div*beta*prob*state[-1]
        else:
            dt_state[i] = (beta-death) * (state[i-1]-state[i])
    
    return dt_state



class Simulation:
        
    def __init__(self, name, model, parameters, cell_types, time):
        self.name = name
        self.model = model
        self.parameters = dict(parameters)
        self.time = time
        self.state_raw = None
        self.state = None
        self.state_tidy = None
        self.cell_types = cell_types
   
        
    def init_model(self):
        # need additional slot for myc ode
        myc = 1
        d = self.parameters
        
        y0 = np.zeros(d["alpha_naive"]+d["alpha_prec"]+d["alpha_th1"]+d["alpha_tfh"]+myc)      
        y0[0] = 1.       
        # init myc concentration
        y0[-1] = 1.
        
        return y0
    
    
    def get_cells(self):
        state = self.state_raw
        d = self.parameters
        i_naive = d["alpha_naive"]
        i_prec = d["alpha_prec"]
        i_th1 = d["alpha_th1"]
        i_tfh = d["alpha_tfh"]
       
        # split state arr into different cell types
        naive_arr = state[:,:i_naive]
        prec_arr = state[:,i_naive:(i_naive+i_prec)]
        th1_arr = state[:,(i_naive+i_prec):(i_naive+i_prec+i_th1)]
        # add -1 because of myc
        tfh_arr = state[:,(i_naive+i_prec+i_th1):-1]
        
        cells = [naive_arr, prec_arr, th1_arr, tfh_arr]
        cells = [np.sum(cell, axis = 1) for cell in cells]
        cells = np.stack(cells, axis = -1)
        
        self.state = cells
        return cells    


    def run_ode(self):
        #print("running time course simulation..")
        
        y0 = self.init_model()
        d = dict(self.parameters)
        time = self.time

        state = odeint(self.model, y0, time, args = (d,)) 
        self.state_raw = state
        return state
    
    
    def run_timecourse(self):
        state_raw = self.run_ode()
        cells = self.get_cells()
        
        colnames = self.cell_types
        df = pd.DataFrame(cells, columns = colnames)
       	
        df["time"] = self.time
        df["sim_name"] = self.name

        #make df tidy for both cell types
        df = pd.melt(df, value_vars = colnames, var_name = "cell_type", 
                     value_name = "cells", id_vars = ["time", "sim_name"])

        self.state_tidy = df
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
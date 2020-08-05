# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 09:56:14 2020

@author: Philipp
"""
import numpy as np
from scipy.integrate import odeint
import seaborn as sns
import pandas as pd
import readout_module as readouts
import matplotlib

def pos_fb(x, EC50, hill = 3):
    out = x**hill / (x**hill + EC50**hill)
    return out



def prob_fb(x, fc, EC50, hill = 3):
    out = (fc*x**hill + EC50**hill) / (x**hill + EC50**hill)
    return out

def prec_model(state, time, d):
    
    n_molecules = 4
    n_memory = 2
    myc, ifng, il21, il10 = state[-n_molecules:]
    
    # only keep state vector that contains response time arrays
    state = state[:-(n_molecules+n_memory)]    
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
    
    prec, th1, tfh = [np.sum(cell) for cell in [prec_arr, th1_arr, tfh_arr]]
    dt_myc = -d["deg_myc"]*myc
    dt_ifng = d["r_ifng"]*th1 - d["deg_ifng"]*ifng
    dt_il21 = d["r_il21"]*tfh - d["deg_il21"]*il21
    dt_il10 = d["r_il10"]*tfh - d["deg_il10"]*il10
    dt_molecules = [dt_myc, dt_ifng, dt_il21, dt_il10]
 
    
    p_th1 = d["p_th1"]*prob_fb(ifng, d["fb_ifng_prob_th1"], d["EC50_ifng_prob_th1"])*prob_fb(il10, d["fb_il10_prob_th1"], d["EC50_il10_prob_th1"])
    p_tfh = d["p_tfh"]*prob_fb(il21, d["fb_il21_prob_tfh"], d["EC50_il21_prob_tfh"])
    
    p_norm = p_th1+p_tfh
    p_th1 = p_th1/p_norm * (1-d["p_prec"])
    p_tfh = p_tfh/p_norm * (1-d["p_prec"])

    # calculate influx
    influx_naive = 0
    influx_prec = naive_arr[-1]*d["beta_naive"]
    influx_th1 = prec_arr[-1]*d["beta_prec"]*p_th1
    influx_tfh = prec_arr[-1]*d["beta_prec"]*p_tfh

    
    beta_p_th1 = d["beta_p_th1"]*pos_fb(myc, d["EC50_myc"]) # add prolif feedback here
    beta_p_tfh = d["beta_p_tfh"]*pos_fb(myc, d["EC50_myc"]) # add prolif feedback here
    #print(beta_p_th1)
    
    dt_naive = diff_chain(naive_arr, influx_naive, d["beta_naive"], 0, 0, 0, 0)
    dt_prec = diff_chain(prec_arr, influx_prec, d["beta_prec"], 0, 0, d["p_prec"], d["n_div_prec"])
    dt_th1 = diff_chain(th1_arr, influx_th1, beta_p_th1, d["beta_m_th1"], d["death_th1"], 1, d["n_div_eff"])
    dt_tfh = diff_chain(tfh_arr, influx_tfh, beta_p_tfh, d["beta_m_tfh"], d["death_tfh"], 1, d["n_div_eff"])

    # memory from total influx of th1 and tfh cells
    dt_th1_mem = d["beta_m_th1"]*th1
    dt_tfh_mem = d["beta_m_tfh"]*tfh
    
    dt_state = np.concatenate((dt_naive, dt_prec, dt_th1, dt_tfh, [dt_th1_mem], [dt_tfh_mem], dt_molecules))
    
    return dt_state


def diff_chain(state, influx, beta, beta_m, death, prob, n_div):
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
            dt_state[i] = influx - (beta+death+beta_m)*state[i] + 2*n_div*beta*prob*state[-1]
        else:
            dt_state[i] = beta*state[i-1] - (beta+death+beta_m)*state[i]
    
    return dt_state



class Simulation:
        
    def __init__(self, name, model, parameters, celltypes, time):
        self.name = name
        self.model = model
        self.parameters = dict(parameters)
        self.time = time
        self.state_raw = None
        self.state = None
        self.state_tidy = None
        self.celltypes = celltypes
        self.molecules = None
        self.molecules_tidy = None
        self.molecule_names = ["myc", "IFNg", "IL21", "IL10"]
        self.n_molecules = 4 # myc and 3 cytokines
        self.n_memory = 2
        
    def init_model(self):
        # need additional slot for myc ode
        d = self.parameters

        y0 = np.zeros(d["alpha_naive"]+d["alpha_prec"]+d["alpha_th1"]+d["alpha_tfh"]+self.n_molecules+self.n_memory)      
        y0[0] = 1.       
        # init myc concentration
        y0[-4] = 1.
        
        return y0
    
    def get_rel_cells(self):
        """
        cells needs to be tidy data frame
        add new column to cells df which gives percentage of total cells
        """
        df = self.run_timecourse()
        df_th1 = df[df.cell_type == "th1"]
        df_tfh = df[df.cell_type == "tfh"]
        df_th1 = df_th1.reset_index(drop = True)
        df_tfh = df_tfh.reset_index(drop = True)
        df_tfh["total"] = df_th1.cells + df_tfh.cells
        df_tfh["rel_cells"] = (df_tfh.cells / df_tfh.total)*100
        #add total cells and compute relative cell fractions      
        return df_tfh
    
    def get_cells(self):
        state = self.state_raw
        # first keep only state wo molecules
        state = state[:,:-self.n_molecules]
        d = self.parameters
        i_naive = d["alpha_naive"]
        i_prec = d["alpha_prec"]
        i_th1 = d["alpha_th1"]
        n_memory = self.n_memory
        # split state arr into different cell types
        naive_arr = state[:,:i_naive]
        prec_arr = state[:,i_naive:(i_naive+i_prec)]
        th1_arr = state[:,(i_naive+i_prec):(i_naive+i_prec+i_th1)]
        tfh_arr = state[:,(i_naive+i_prec+i_th1):-n_memory]
        

        cells = [naive_arr, prec_arr, th1_arr, tfh_arr]
        cells = [np.sum(cell, axis = 1) for cell in cells]
        
        # get memory cells
        th1_mem = state[:,-2]
        tfh_mem = state[:,-1]

        cells[2] = cells[2] + th1_mem
        cells[3] = cells[3] + tfh_mem        
        cells = np.stack(cells, axis = -1)
        
        self.state = cells
        return cells    


    def get_molecules(self):
        """
        from raw data get molecules and convert to data frame
        Returns
        -------
        df : data frame
            cytokines and other molecules over time
        """
        state = self.state_raw
        arr = state[:,-self.n_molecules:]
        df = pd.DataFrame(data = arr, columns = self.molecule_names)
        df["time"] = self.time
        df_tidy = pd.melt(df, id_vars = ["time"], value_name = "conc.", var_name = "molecule")
        self.molecules = df
        self.molecules_tidy = df_tidy
        return df
    

    def run_ode(self):
        """
        run ode and store as raw data, incl cells and molecules
        Returns
        -------
        state : data frame
            raw simulation data
        """
        y0 = self.init_model()
        d = dict(self.parameters)
        time = self.time

        state = odeint(self.model, y0, time, args = (d,)) 
        self.state_raw = state
        return state
    
    
    def run_timecourse(self):
        self.run_ode()
        self.get_cells()
        self.get_molecules()
        
        colnames = self.celltypes
        df = pd.DataFrame(self.state, columns = colnames)
       	
        df["time"] = self.time
        df["sim_name"] = self.name

        #make df tidy for both cell types
        df = pd.melt(df, value_vars = colnames, var_name = "cell_type", 
                     value_name = "cells", id_vars = ["time", "sim_name"])

        self.state_tidy = df
        return df


    def get_readouts(self):
        """
        run time course and get readouts for each effector cell
        return dataframe
        df : dataframe
        """
        df = self.run_timecourse()
        df_th1 = df[df.cell_type == "th1"]
        df_tfh = df[df.cell_type == "tfh"]

        df_th1 = self.get_readouts_from_df(df_th1)
        df_th1["name"] = "th1"
        df_tfh = self.get_readouts_from_df(df_tfh)
        df_tfh["name"] = "tfh"
        df = pd.concat([df_th1, df_tfh])

        return df
        
    
    def get_readouts_from_df(self, state):
        """
        run simulation and then get readouts

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.

        Returns
        -------
        reads_df : TYPE
            DESCRIPTION.

        """
        # get readouts
        peak = readouts.get_peak(state.time, state.cells)
        area = readouts.get_area(state.time, state.cells)
        tau = readouts.get_peaktime(state.time, state.cells)
        decay = readouts.get_decay(state.time, state.cells)     
        # convert readouts to dataframe
        reads = [peak, area, tau, decay]

        read_names = ["peak", "area", "tau", "decay"]
        reads_df = pd.DataFrame(data = np.array([reads]), columns = read_names)
     
        return reads_df
    

    def vary_param(self, arr_dict):
        """
        vary parameters and compute readouts for both cells
        provide dict with var names and corresponding arrays to be varied at
        the same time. arrays should have same length
        arr_dict: dictionary
        should contain parameter name and corresponding array as value
        all parameters in dict are varied at the same time
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
        df["sim_name"] = self.name
        
        self.parameters = old_parameters
        
        return df


    def get_relative_readouts(self, df):
        """
        split readout data frame based on cell types, then compute relative readouts
        """
        # need to get copy otherwise view is returned if I index like this
        df_th1 = df[df.name == "th1"].copy()
        df_tfh = df[df.name == "tfh"].copy()
        df_th1 = df_th1.reset_index(drop = True)
        df_tfh = df_tfh.reset_index(drop = True)

        # take only readout values (first four columns) and divide
        df_rel = df_th1.iloc[:,:4] / df_tfh.iloc[:,:4]
        # add other columns (same for th1 and tfh df)
        df_rel["param_val"] = df_th1.param_val
        df_rel["param_name"] = df_th1.param_name
        df_rel["sim_name"] = self.name
        # get total area
        # compute the
        return df_rel
    
    
    def normalize_readout_df(self, df, norm_idx):
        """
        take either relative or normal readout data frame and readouts and param value
        by row corresponding to norm_idx
        """
        cols = ["peak", "area", "tau", "decay", "param_val"]
        
        if "name" in df.columns:
            df_th1 = df[df.name == "th1"].copy()
            df_tfh = df[df.name == "tfh"].copy()
            df_th1 = df_th1.reset_index(drop = True)
            df_tfh = df_tfh.reset_index(drop = True)
        
        # divide each readout column and parameter val column by row that corresponds to norm idx
            df_norm = df_th1.loc[:,cols] / df_th1.loc[norm_idx,cols]
            df_norm["name"] = "th1"
            df_norm2 = df_tfh.loc[:,cols] / df_tfh.loc[norm_idx,cols]        
            df_norm2["name"] = "tfh"
            out = pd.concat([df_norm, df_norm2])
        
        else:
            out = df.loc[:,cols] / df.loc[norm_idx, cols]
            out["sim_name"] = self.name
            out["param_name"] = df.param_name[0]

        return out
    

    def run_timecourses(self, arr_dict):
        """
        run multiple time courses, vary parameter and save relative cells (tfh%total incl parameter value)

        Parameters
        ----------
        arr_dict

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """
        old_parameters = dict(self.parameters)
        df_list = []
        values = list(arr_dict.values())
        keys = list(arr_dict.keys())
        for val in zip(*values):
            for key, v in zip(keys, val):
                self.parameters[key] = v

            df = self.get_rel_cells()
            df["pval"] = val[0]
            df_list.append(df)
        
        df = pd.concat(df_list)
        
        self.parameters = old_parameters
        return df
    

    def plot_timecourses(self, df, log = False, cbar_label = "feedback strength"):
        """

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # parameter for scaling of color palette in sns plot
        arr = df.loc[:,["pval"]]
        vmin = np.min(arr)
        vmax = np.max(arr)
        if log == True:

            norm = matplotlib.colors.LogNorm(vmin = vmin, vmax = vmax)
            # if log then also in sns plot
        else:
            norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)  
        
        # make mappable for colorbar
        cmap = "Blues"
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])        
        
        # set hue to parameter name 
        g = sns.relplot(x = "time", y = "rel_cells", kind = "line", data = df, hue = "pval", 
                        hue_norm = norm, palette = cmap, legend = False)


        g.set(ylim = (0,100), ylabel = "Tfh % of total")
        cbar = g.fig.colorbar(sm)
        # add colorbar       
        cbar.set_label(cbar_label)
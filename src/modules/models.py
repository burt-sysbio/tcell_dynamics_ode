# -*- coding: utf-8 -*-
"""
keep all ode models here
"""
import numpy as np
from src.modules.models_helper import *
from scipy.stats import gamma
# =============================================================================
# linear models
# ============================================================================
def virus_model(d):
    alpha = d["vir_alpha"]
    beta = d["vir_beta"]
    scale = 1/beta
    mygamma = gamma(a = alpha, scale = scale)
    return mygamma


def th_cell_diff(state, time, d, prolif_model, core_model, virus_model):
    """
    takes state vector to differentiate effector cells as linear chain
    needs alpha and beta(r) of response time distribution, probability
    and number of precursor cells
    """
    # get proliferation and death based on proliferation model (il2/timer)
    beta_p = prolif_model(state, d)

    # divide array into cell states
    myc = state[-1]
    il2 = state[-2]
    n_molecules = 2
    th_state = state[:-n_molecules]
    tnaive, teff = get_cell_states(th_state, d)

    # compute il2 and myc changes based on antigen load (set vir load to 0 for no ag effect)
    ag = virus_model.pdf(time) * d["vir_load"]
    dt_myc = get_myc(myc, ag, d)  # ag inhibits myc degradation
    dt_il2 = get_il2(tnaive, teff, il2, ag, d)  # ag induces il2 secretion by effectors

    # apply feedback on rate beta
    beta = d["beta"]

    # t cells ODEs
    rate_death = 1.0/d["lifetime_eff"]
    dt_state = core_model(th_state, d, beta, rate_death, beta_p)

    # output
    dt_state = np.concatenate((dt_state, [dt_il2], [dt_myc]))
    return dt_state


def diff_core(th_state, d, beta, rate_death, beta_p):
   
    # differentation, th_state should be array only for T cells (not myc or il2)
    dt_state = np.zeros_like(th_state)
    # calculate number of divisions for intermediate population
    for j in range(len(th_state)):
        if j == 0:
            dt_state[j] = d["b"]-(beta+d["d_naive"])*th_state[j] 
                      
        elif j < (d["alpha"]):
            dt_state[j] = beta*th_state[j-1]-(beta+d["d_prec"])*th_state[j]
        
        elif j == (d["alpha"]):
            dt_state[j] = d["n_div"]*beta*th_state[j-1] + (2*beta_p*th_state[-1]) - (rate_death+beta_p)*th_state[j]       
        
        else:
            dt_state[j] = beta_p*th_state[j-1]-(beta_p+rate_death)*th_state[j]
        
    return dt_state


# =============================================================================
# helper functions
# =============================================================================
def get_cell_states(th_state, d):
    tnaive = th_state[:d["alpha"]]
    teff = th_state[d["alpha"]:]
    tnaive = np.sum(tnaive)
    teff = np.sum(teff)    
    
    return tnaive, teff


def get_myc(myc, ag, d):
    # antigen inhibition of myc degradation
    deg_myc = d["K_ag_myc"]**3 / (ag**3 + d["K_ag_myc"]**3) * (1/d["lifetime_myc"])
    dt_myc = -deg_myc*myc
    return dt_myc


def get_il2(tnaive, teff, il2, ag, d):
    # antigen induces il2 secretion by effector cells
    ag_il2 = (ag**3 / (ag**3 + d["K_ag_il2"]**3))
    # il2 production is naive prod + eff(ag) prod - consumption (eff)
    dt_il2 = d["r_il2"]*tnaive + (d["r_il2_eff"]*ag_il2 - d["up_il2"]*il2) * teff

    return dt_il2

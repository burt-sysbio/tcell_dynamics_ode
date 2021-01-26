# -*- coding: utf-8 -*-
"""
keep all ode models here
"""
import numpy as np
from src.modules.models_helper import *
from scipy.stats import gamma, lognorm
from scipy import interpolate
from scipy.integrate import odeint
from numba import jit

# =============================================================================
# t cell models
# ============================================================================
def th_cell_diff(state, time, d, prolif_model, core_model, virus_model):
    """
    takes state vector to differentiate effector cells as linear chain
    needs alpha and beta(r) of response time distribution, probability
    and number of precursor cells
    """
    # divide array into cell states
    myc = state[-1]
    il2 = state[-2]
    tchronic = state[-3]
    n_chronic = 1
    n_molecules = 2
    th_state = state[:-(n_molecules+n_chronic)]
    tnaive, teff = get_cell_states(th_state, d)
    all_cells = tchronic+teff+tnaive

    # get proliferation and death based on proliferation model (il2/timer)
    beta_p = prolif_model(state, d)

    # add carrying capacity
    beta_p = (1-all_cells/d["K_carr"])*beta_p
    beta_p = beta_p*fb_fc(tchronic, d["neg_fb_chr"], d["K_neg_fb_chr"])

    # compute il2 and myc changes based on antigen load (set vir load to 0 for no ag effect)
    ag = virus_model(time)
    dt_myc = get_myc(myc, ag, d)  # ag inhibits myc degradation
    dt_il2 = get_il2(tnaive, teff, il2, ag, d)  # ag induces il2 secretion by effectors

    # apply feedback on rate beta
    beta = d["beta"]

    # t cells ODEs
    r_death = 1.0/d["lifetime_eff"]

    # ag effects r chronic menten style
    r_chronic = menten(ag, d["r_chronic"], d["K_ag_chr"], 3)
    r_chronic = r_chronic*fb_fc(tchronic, d["pos_fb_chr"], d["K_pos_fb_chr"])


    pcore = [d["b"], d["d_naive"], d["alpha"], d["d_prec"], d["n_div"],
             beta, r_death, beta_p, r_chronic]

    dt_state = core_model(th_state, *pcore)

    # feedback of antigen on chronic cell diff
    dt_chronic = r_chronic*teff
    # feedback of chronic cells on chronic cell diff

    # output
    dt_state = np.concatenate((dt_state, [dt_chronic], [dt_il2], [dt_myc]))
    return dt_state

@jit
def diff_core(th_state, b, d_naive, alpha, d_prec, n_div, beta, r_death, beta_p, r_chronic):
    # differentation, th_state should be array only for T cells (not myc or il2)
    dt_state = np.zeros_like(th_state)
    # calculate number of divisions for intermediate population
    for j in range(len(th_state)):
        if j == 0:
            dt_state[j] = b-(beta+d_naive)*th_state[j]
                      
        elif j < alpha:
            dt_state[j] = beta*th_state[j-1]-(beta+d_prec)*th_state[j]
        
        elif j == alpha:
            dt_state[j] = n_div*beta*th_state[j-1] + (2*beta_p*th_state[-1]) - (r_death+beta_p+r_chronic)*th_state[j]
        
        else:
            dt_state[j] = beta_p*th_state[j-1]-(beta_p+r_death+r_chronic)*th_state[j]
        
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
    deg_myc = (d["K_ag_myc"]**3 / (ag**3 + d["K_ag_myc"]**3)) * (1/d["lifetime_myc"])
    dt_myc = -deg_myc*myc
    return dt_myc


def get_il2(tnaive, teff, il2, ag, d):
    # antigen induces il2 secretion by effector cells
    # note that if ag is inf nan is returned which is bad!
    ag_il2 = ag**3 / (ag**3 + d["K_ag_il2"]**3)
    # il2 production is naive prod + eff(ag) prod - consumption (eff)
    dt_il2 = d["r_il2"]*tnaive + (d["r_il2_eff"]*ag_il2 - d["up_il2"]*il2) * teff
    return dt_il2

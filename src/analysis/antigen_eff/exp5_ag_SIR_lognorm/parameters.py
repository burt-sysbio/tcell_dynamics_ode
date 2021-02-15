# parameters for exp2 const ag chr
# special: set r_chronic !=0

d = {
    # initial conditions
    "initial_cells": 100.0,
    "il2_y0": 1.0,
    # differentiation
    "b": 0,  # model influx
    "alpha": 10,
    "beta": 5.,
    "r_chronic" : 1,
    # proliferation
    "n_div": 1,
    "alpha_p": 10,
    "beta_p": 20.0,
    # death
    "lifetime_eff": 4.12, # 1/0.24 literature data
    "d_prec": 0,
    "d_naive": 0,
    "hill": 3,
    # myc parameters
    "K_myc": 0.01,
    # leave lifetime myc to 1 and change K_myc if neccessary
    "lifetime_myc": 3.125,
    # il2 parameters
    "r_il2": 1e3,
    "r_il2_eff": 1e3,
    "up_il2": 1,
    "base_il2": 0,
    "K_il2": 1.0,  # il2 effect on prolif
    # virus params
    "vir_alpha": 1.0,  # virus alpha (gamma dist)
    "vir_load": 1.0,  # antigen can be set to 0 for no antigen effects
    "vir_beta": 2.0,  # virus beta (gamma dist)
    "K_ag_myc": 0.5,  # antigen inhibits myc degradation
    "K_ag_il2": 1.0,  # antigen induces il2 secretion on teffs
    "K_ag_chr" : 0.5,
    "K_pos_fb_chr" : 1e2,
    "K_neg_fb_chr" : 1e2,
    "neg_fb_chr" : 1,
    "pos_fb_chr" : 1,
    # virus params for ode model
    "vir_growth": 1,
    "vir_death": 1,
    # carrying capacity
    "K_carr" : 1e10,
    "SIR_r0" : 2,
}

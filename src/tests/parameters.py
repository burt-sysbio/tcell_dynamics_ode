d = {
    # initial conditions
    "initial_cells" : 1.0,
    "il2_y0" : 1.0,
    # differentiation
    "b": 0,  # model influx
    "alpha" : 10,
    "beta" : 10.,
    # proliferation
    "n_div": 1,
    "alpha_p" : 10,
    "beta_p" : 100.0,
    # death
    "lifetime_eff" : 1.0,
    "d_prec" : 0,
    "d_naive" : 0,
    "hill": 3,
    # myc parameters
    "K_myc" : 0.001,
    # leave lifetime myc to 1 and change K_myc if neccessary
    "lifetime_myc": 1.0,
    # il2 parameters
    "up_il2": 1,
    "base_il2": 0,
    # leave uptake and K il2 to 1 and change r_il2 if neccessary
    "r_il2": 1e2,
    "r_il2_eff": 1e2,
    "K_il2": 1.0,  # il2 effect on prolif
    "up_il2": 1,
    # virus params
    "vir_alpha" : 1.0,  # virus alpha (gamma dist)
    "vir_load" : 1,  # antigen can be set to 0 for no antigen effects
    "vir_beta" : 5.0,  # virus beta (gamma dist)
    "K_ag_myc" : 0.5,  # antigen inhibits myc degradation
    "K_ag_il2" : 0.5,  # antigen induces il2 secretion on teffs
    # virus params for ode model
    "vir_growth" :1,
    "vir_death" : 1,
}
# -*- coding: utf-8 -*-
"""
keep all ode models here
"""
import numpy as np

# =============================================================================
# linear models
# =============================================================================
def th_cell_diff(th_state, time, model, d, core):
    lifetime_eff, beta_p = model(th_state, time, d)
    assert lifetime_eff > 0
    rate_death = 1.0/lifetime_eff   
    dt_state = th_cell_core(th_state, rate_death, beta_p, d, core)
    
    return dt_state

        
def diff_effector(th_state, teff, d, beta, rate_death, beta_p):
   
    # make empty state vector should not include myc because I add it extra
    dt_state = np.zeros_like(th_state)
    #print(th_state.shape)
    # check that alpha is even number, I need this, so I can use alpha int = 0.5alpha
    assert d["alpha"] % 2 == 0
    alpha_int = int(d["alpha"] / 2)
    # calculate number of divisions for intermediate population
    #n_div1 = (2*mu_div1)/mu_prolif
    #n_div2 = (2*mu_div2)/mu_prolif
    n_div1 = d["n_div"]
    n_div2 = d["n_div"]
    for j in range(len(th_state)):
        #print(j)
        if j == 0:
            dt_state[j] = d["b"]-(beta+d["d_naive"])*th_state[j] 
            
        elif j < alpha_int:
            dt_state[j] = beta*th_state[j-1]-(beta+d["d_naive"])*th_state[j]
            
        elif j == alpha_int:
            dt_state[j] = n_div1*beta*th_state[j-1]-(beta+d["d_prec"])*th_state[j]

        elif j < (d["alpha"]):
            dt_state[j] = beta*th_state[j-1]-(beta+d["d_prec"])*th_state[j]
        
        elif j == (d["alpha"]):
            dt_state[j] = n_div2*beta*th_state[j-1] + (2*beta_p*th_state[-1]) - (rate_death+beta_p)*th_state[j]       
        
        else:
            dt_state[j] = beta_p*th_state[j-1]-(beta_p+rate_death)*th_state[j]
        
    return dt_state

def diff_no_intermediates(th_state, teff, d, beta, rate_death, beta_p):
   
    # make empty state vector should not include myc because I add it extra
    dt_state = np.zeros_like(th_state)
    #print(th_state.shape)   
    # calculate number of divisions for intermediate population
    #n_div1 = (2*mu_div1)/mu_prolif
    #n_div2 = (2*mu_div2)/mu_prolif
    for j in range(len(th_state)):
        #print(j)
        if j == 0:
            dt_state[j] = d["b"]-(beta+d["d_naive"])*th_state[j] 
                      
        elif j < (d["alpha"]):
            dt_state[j] = beta*th_state[j-1]-(beta+d["d_prec"])*th_state[j]
        
        elif j == (d["alpha"]):
            dt_state[j] = d["n_div"]*beta*th_state[j-1] + (2*beta_p*th_state[-1]) - (rate_death+beta_p)*th_state[j]       
        
        else:
            dt_state[j] = beta_p*th_state[j-1]-(beta_p+rate_death)*th_state[j]
        
    return dt_state


def th_cell_core(th_state, rate_death, beta_p, d, differentiate):
    """
    model2
    takes state vector to differentiate effector cells as linear chain
    needs alpha and beta(r) of response time distribution, probability
    and number of precursor cells
    """    
    # divide array into cell states
    myc = th_state[-1]
    il2_ex = th_state[-2]  
    
    n_molecules = 2
    th_state = th_state[:-n_molecules]
    tnaive, tint, teff = get_cell_states(th_state, d)


    # apply feedback on rate beta  
    beta = d["beta"]  
    # check homeostasis criteria    
    # differentiation    

    dt_state = differentiate(th_state, teff, d, beta, rate_death, beta_p)
    d_myc = -(1./d["lifetime_myc"])*myc
    dt_il2_ex = -d["up_il2"]*(tint+teff)*il2_ex
    dt_state = np.concatenate((dt_state, [dt_il2_ex], [d_myc]))
 
    return dt_state 
 
# =============================================================================
# helper functions
# =============================================================================
def get_cell_states(th_state, d):
    assert d["alpha"] % 2 == 0
    assert d["alpha"] > 0
    
    alpha_int = int(d["alpha"] / 2)
    
    # this is for the alpha int model
    # for naive --> eff model use
    #tnaive = th_state[:d["alpha"]] and exclude tint
    #then also change cyto producers il2_producers = tnaive
    tnaive = th_state[:alpha_int]
    tint = th_state[alpha_int:d["alpha"]]
    teff = th_state[d["alpha"]:]
    
    #assert len(tnaive)+len(tint)+len(teff) == len(th_state)
    tnaive = np.sum(tnaive)
    tint = np.sum(tint)
    teff = np.sum(teff)    
    
    return tnaive, tint, teff


def get_cyto_producers(th_state, d):

    tnaive, tint, teff = get_cell_states(th_state, d)
    # for naive --> teff model do
    #il2_producers = tnaive
    #il2_consumers = teff
    
    il2_producers = tint if tint > 0 else 1e-12
    il2_consumers = tint+teff if tint+teff > 0 else 1e-12
    il7_consumers = teff if teff > 0 else 1e-12
    
    arr = np.asarray([il2_producers, il2_consumers, il7_consumers]) > 0
    assert  arr.all()
    
    return il2_producers, il2_consumers, il7_consumers


def menten(conc, vmax, K, hill):   
    # make sure to avoid numeric errors for menten
    assert conc >= 0
    out = (vmax*conc**hill) / (K**hill+conc**hill)
    
    return out


def get_myc(th_state):
    myc = th_state[-1] if th_state[-1] >= 0 else 1e-12
    return myc


def get_il2_ex(th_state):

    il2_ex = th_state[-2] if th_state[-2] >= 0 else 1e-12
    return il2_ex


def get_il2(il2_ex, il2_producers, il2_consumers, d, time):
            
    out = (il2_ex + d["rate_il2"]*il2_producers)/(d["K_il2"]+il2_consumers)
    return out
# =============================================================================
# homeostasis models
# =============================================================================
def null_model(th_state, time, d):
    lifetime_eff, beta_p = d["lifetime_eff"], d["beta_p"]
    return lifetime_eff, beta_p
    

def il2_menten_prolif(th_state, time, d):

    il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d)  
    il2_ex = get_il2_ex(th_state)
    c_il2 = get_il2(il2_ex, il2_producers, il2_consumers, d, time)
    
    vmax = d["beta_p"]
    beta_p = menten(c_il2, vmax, d["K_il2"], d["hill"])
    lifetime_eff = d["lifetime_eff"]
    
    return lifetime_eff, beta_p



def il2_menten_lifetime(th_state, time, d):
    
    il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d)  
    il2_ex = get_il2_ex(th_state)
    c_il2 = get_il2(il2_ex, il2_producers, il2_consumers, d, time)
               
    vmax = d["lifetime_eff"]
    lifetime_eff = menten(c_il2, vmax, d["K_il2"], d["hill"])
    beta_p = d["beta_p"]
    
    return lifetime_eff, beta_p


def C_menten_prolif(th_state, time, d):
    
    il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d) 
    c_C = d["rate_C"] / (d["K_C"]+il7_consumers)

    vmax = d["beta_p"]
    K = d["K_C"]
    hill = d["hill"]
    
    beta_p = menten(c_C, vmax, K, hill)
    lifetime_eff = d["lifetime_eff"]
    
    return lifetime_eff, beta_p


def C_menten_lifetime(th_state, time, d):
    
    il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d) 
    c_C = d["rate_C"] / (d["K_C"]+il7_consumers)

    vmax = d["lifetime_eff"]
    K = d["K_C"]
    hill = d["hill"]
    
    lifetime_eff = menten(c_C, vmax, K, hill)
    beta_p = d["beta_p"]
    
    return lifetime_eff, beta_p


def timer_menten_prolif(th_state, time, d):

    myc = get_myc(th_state)
    il2_ex = get_il2_ex(th_state)
    vmax = d["beta_p"]
    K = d["K_myc"]
    hill = d["hill"]
    
    beta_p = menten(myc, vmax, K, hill)
    lifetime_eff = d["lifetime_eff"]
    
    return lifetime_eff, beta_p

def timer_il2(th_state, time, d):

    myc = get_myc(th_state)
    il2_ex = get_il2_ex(th_state)
    vmax = d["beta_p"]
    K = d["K_myc"]
    hill = d["hill"]

    il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d)  
        
    c_il2 = get_il2(il2_ex, il2_producers, il2_consumers, d, time)
    
    # compute beta p as geometric mean of il2 and myc effect
    product = menten(myc, vmax, K, hill)*menten(c_il2, vmax, K, hill)
    beta_p = np.sqrt(product)
    lifetime_eff = d["lifetime_eff"]
    
    return lifetime_eff, beta_p

def timer_menten_lifetime(th_state, time, d):

    myc = get_myc(th_state)
    vmax = d["lifetime_eff"]
    K = d["K_myc"]
    hill = d["hill"]
    
    lifetime_eff = menten(myc, vmax, K, hill)
    beta_p = d["beta_p"]
    
    return lifetime_eff, beta_p


def thres_prolif(d, time):
    lifetime_eff = d["lifetime_eff"]
    beta_p = d["beta_p"]
    
    decay = 100.
    t0 = time-d["t0"]
    
    if t0 > 0:
        beta_p = beta_p*np.exp(-decay*t0)
        
    return lifetime_eff, beta_p


def thres_lifetime(d, time):
    lifetime_eff = d["lifetime_eff"]
    beta_p = d["beta_p"]

    decay = 100.
    t0 = time-d["t0"]

    if t0 > 0:
        lifetime_eff = lifetime_eff*np.exp(-decay*t0)

    
    return lifetime_eff, beta_p


def test_thres(c, crit, time, d):
    if c < crit:
        d["crit"] = True
        d["t0"] = time

        


def il2_thres_prolif(th_state, time, d):
    lifetime_eff, beta_p = d["lifetime_eff"], d["beta_p"]   
    
    if d["crit"] == True:    
        lifetime_eff, beta_p = thres_prolif(d, time)
    else:
        il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d)
        conc_il2 = d["rate_il2"]*il2_producers/(d["K_il2"]+il2_consumers)  
        if il7_consumers > 0.01:
            test_thres(conc_il2, d["crit_il2"], time, d)
            
    return lifetime_eff, beta_p


def il2_thres_lifetime(th_state, time, d):

    lifetime_eff, beta_p = d["lifetime_eff"], d["beta_p"]   
    
    if d["crit"] == True:    
        lifetime_eff, beta_p = thres_lifetime(d, time)
    else:
        il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d)
        conc_il2 = d["rate_il2"]*il2_producers/(d["K_il2"]+il2_consumers)   
        if il7_consumers > 0.01:           
            test_thres(conc_il2, d["crit_il2"], time, d)
 
    return lifetime_eff, beta_p


def C_thres_prolif(th_state, time, d):
    lifetime_eff, beta_p = d["lifetime_eff"], d["beta_p"]   
    
    if d["crit"] == True:    
        lifetime_eff, beta_p = thres_prolif(d, time)
    else:        
        il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d)  
        conc_C = d["rate_C"] / (d["K_C"]+il7_consumers)
        if il7_consumers > 0.01:
            test_thres(conc_C, d["crit_C"], time, d)
 
            
    return lifetime_eff, beta_p


def C_thres_lifetime(th_state, time, d):
    lifetime_eff, beta_p = d["lifetime_eff"], d["beta_p"]   
    
    if d["crit"] == True:    
        lifetime_eff, beta_p = thres_lifetime(d, time)
    else:
        il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d)  
        conc_C = d["rate_C"] / (d["K_C"]+il7_consumers)
        if il7_consumers > 0.01:
            test_thres(conc_C, d["crit_C"], time, d)
            
    return lifetime_eff, beta_p


def timer_thres_prolif(th_state, time, d):
    lifetime_eff, beta_p = d["lifetime_eff"], d["beta_p"]  
    
    if d["crit"] == True:    
        lifetime_eff, beta_p = thres_prolif(d, time)
    else:
        myc = get_myc(th_state)
        test_thres(myc, d["crit_myc"], time, d)
        
    return lifetime_eff, beta_p


def timer_thres_lifetime(th_state, time, d):
    lifetime_eff, beta_p = d["lifetime_eff"], d["beta_p"]  
    
    if d["crit"] == True:    
        lifetime_eff, beta_p = thres_lifetime(d, time)
    else:
        myc = get_myc(th_state)
        test_thres(myc, d["crit_myc"], time, d)
        
    return lifetime_eff, beta_p
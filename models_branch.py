# -*- coding: utf-8 -*-
"""
keep all ode models here
"""

import numpy as np

# =============================================================================
# linear models
# =============================================================================
def get_il2_producers(tnaive, n_th1_int, n_th2_int, n_th1, n_th2, model = "crosstalk"):
    """
    calculate il2 producers based on model versions
    """
    il2_producers = None
    if model == "crosstalk":
        il2_producers = n_th2_int
    if model == "regular":
        il2_producers = n_th1_int + n_th2_int
    
    return il2_producers


def get_il2_consumers(tnaive, n_th1_int, n_th2_int, n_th1, n_th2, model = "crosstalk"):
    """
    calc il2 consumers based on model version
    """
    il2_consumers = None
    if model == "crosstalk":
        il2_consumers = n_th1
    if model == "regular":
        il2_consumers = n_th1+n_th2
    
    return il2_consumers


def th_cell_branch(th_state, time, model, d, core):
    
    # calculate cytokines based on cell numbers
    tnaive, n_th1_int, n_th2_int, n_th1, n_th2 = assign_states(th_state, d, core)
    myc = th_state[-1]
    
    il2_producers = get_il2_producers(tnaive, n_th1_int, n_th2_int, n_th1, n_th2)
    il2_consumers = get_il2_consumers(tnaive, n_th1_int, n_th2_int, n_th1, n_th2)
    
    ifng, il21, il2 = calc_cytokines(d, il2_producers, n_th1, n_th2)
    il2 = calc_il2_consumption(il2, il2_consumers, d)
        
    # calculate cytokine effects on probability, differentiation and proliferation
    beta1_p, beta2_p = calc_prolif_rates(ifng, il21, il2, d)
    beta1, beta2 = calc_diff_rates(ifng, il21, il2, d)
    p1, p2 = calc_probs(ifng, il21, il2, beta1, beta2, d, core)
    
    # check homeostasis conditions        
    beta1_p = model(beta1_p, il2, myc, time, d)
    beta2_p = model(beta2_p, il2, myc, time, d)
    lifetime_eff1 = d["lifetime_eff1"]
    lifetime_eff2 = d["lifetime_eff2"]
    
    assert lifetime_eff1 > 0 and lifetime_eff2 > 0
    
    death1 = 1.0/lifetime_eff1
    death2 = 1.0/lifetime_eff2
    
    # differentiate according to early or late state
    dt_state = differentiate(th_state, core, beta1, beta2, beta1_p, beta2_p, 
                             p1, p2, death1, death2, d)

    d_myc = -(1./d["lifetime_myc"])*myc
    dt_state = np.concatenate((dt_state, [d_myc]))

    return dt_state


def assign_states(th_state, d, core):
    
    tnaive = th_state[0]
    # nstates depends on core function
    n_states = 0 if core.__name__ == "branch_competetive" else 1
    th1 = th_state[1:(d["alpha1"]+d["alpha1_p"]+n_states)]
    # add -1 because of myc
    th2 = th_state[(d["alpha1"]+d["alpha1_p"]+n_states):-1]     
    n_th1 = np.sum(th1[-d["alpha1_p"]:])
    n_th2 = np.sum(th2[-d["alpha2_p"]:])
    # these are naive cells and th1 / th2 precursor cells
    n_th1_int = np.sum(th1)-n_th1
    n_th2_int = np.sum(th2)-n_th2 
    
    
    states = tnaive, n_th1_int, n_th2_int, n_th1, n_th2

    return states


def calc_cytokines(d, il2_producers, n_th1, n_th2):
    ifng = d["rate_ifng"]*n_th1 + d["ifng_ext"]
    il21 = d["rate_il21"]*n_th2 + d["il21_ext"]
    il2 = d["rate_il2"]*il2_producers+d["il2_ext"]
    
    # make sure no cytokine concentration below 0
    cytokines = np.array([ifng, il21, il2])
    cytokines[cytokines < 0] = 0
    return cytokines


def calc_diff_rates(ifng, il21, il2, d):

    fb_ifn = (d["fb_rate_ifng"]*ifng+d["K_ifng"]) / (ifng+d["K_ifng"])
    fb_il21 = (d["fb_rate_il21"]*il21+d["K_il21"]) / (il21+d["K_il21"])
    
    fb_il2_th1 = (d["fb_rate_il2_th1"]*il2+d["K_il2"]) / (il2+d["K_il2"])
    fb_il2_th2 = (d["fb_rate_il2_th2"]*il2+d["K_il2"]) / (il2+d["K_il2"])
    
    beta1 = d["beta1"]*fb_ifn*fb_il2_th1
    beta2 = d["beta2"]*fb_il21*fb_il2_th2 
    
    return beta1, beta2

    
def calc_prolif_rates(ifng, il21, il2, d):

    fb_ifn = (d["fb_prolif_ifng"]*ifng+d["K_ifng"]) / (ifng+d["K_ifng"])
    fb_il21 = (d["fb_prolif_il21"]*il21+d["K_il21"]) / (il21+d["K_il21"])
    
    fb_il2_th1_p = (d["fb_prolif_il2_th1"]*il2+d["K_il2"]) / (il2+d["K_il2"])
    fb_il2_th2_p = (d["fb_prolif_il2_th2"]*il2+d["K_il2"]) / (il2+d["K_il2"])
    
    beta1_p = d["beta1_p"]*fb_ifn*fb_il2_th1_p
    beta2_p = d["beta2_p"]*fb_il21*fb_il2_th2_p
    
    return beta1_p, beta2_p


def calc_probs(ifng, il21, il2, beta1, beta2, d, core):
    p1_norm = 1.0
    p2_norm = 1.0

    if core.__name__ == "branch_precursor":
	    fb1 = d["fb_ifng"]*ifng**3/(ifng**3+d["K_ifng"]**3)
	    fb2 = d["fb_il21"]*il21**3/(il21**3+d["K_il21"]**3)
	    # account for default probability and feedback strength
	    p1 = (fb1+1)*d["p1_def"]
	    p2 = (fb2+1)*(1-d["p1_def"])
	    ### this is the effective probability after feedback integration 
	    p1_norm = p1/(p1+p2)
	    p2_norm = 1-p1_norm
        
        # this was based
        #p2 = 1.
	    ### calculate fb rate effects
	    # adjust this parameter to effectively change branching prob because beta1 and beta2 also
	    # play a role, note that fb can affect beta1,2
	    #p1 = p1_norm*beta2/(beta1*(1-p1_norm))

    return p1_norm, p2_norm


def differentiate(th_state, core, beta1, beta2, beta1_p, beta2_p, p1, p2, death1, death2, d):
    
    tnaive = th_state[0]
    
    # in precursor model beta is adjusted by probability term
    if core.__name__ == "branch_competetive":
    	dt_th0 = -(beta1+beta2)*tnaive
    	n_states = 0
    if core.__name__ == "branch_precursor":
    	dt_th0 = -d["decision_time"]*tnaive
    	n_states = 1

    th1 = th_state[1:(d["alpha1"]+d["alpha1_p"]+n_states)]
    # add -1 because of myc
    th2 = th_state[(d["alpha1"]+d["alpha1_p"]+n_states):-1]     
    #th1 = th1[-d["alpha1_p"]:]
    #th2 = th2[-d["alpha2_p"]:] 

    dt_th1 = core(th1, tnaive, d["alpha1"], beta1, beta1_p, death1, p1, d)
    dt_th2 = core(th2, tnaive, d["alpha2"], beta2, beta2_p, death2, p2, d)

    dt_state = np.concatenate(([dt_th0], dt_th1, dt_th2))
    
    return dt_state


def branch_competetive(state, th0, alpha, beta, beta_p, death, p, d):
    """
    takes state vector to differentiate effector cells as linear chain
    needs alpha and beta(r) of response time distribution, probability
    and number of precursor cells
    """
    dt_state = np.zeros_like(state)
    #print(len(state))
    
    if alpha == 1:
        for j in range(len(state)):
            if j == 0:
                dt_state[j] = p*beta*th0+2*beta_p*state[-1]-(beta_p+death)*state[j]
            else:
                dt_state[j] = beta_p*state[j-1]- (beta_p+death)*state[j] 
    
    else:                    
        for j in range(len(state)):
            if j == 0:
                dt_state[j] = p*beta*th0 - (beta+d["d_prec"])*state[j]                
            elif j < (alpha-1):
                dt_state[j] = beta*state[j-1]-(beta+d["d_prec"])*state[j]                
            elif j == (alpha-1):
                # the problem with the 4 and 2 is that since differentiation takes 1 day it should divide twice giving 4 cells
                # however, if it has arrived in the final states if should double every half day
                dt_state[j] = beta*state[j-1]+2*beta_p*state[-1] - (death+beta_p)*state[j]  

            else:
                assert j >= alpha
                dt_state[j] = beta_p*state[j-1]- (beta_p+death)*state[j] 
               
    return dt_state


def branch_precursor(state, th0, alpha, beta, beta_p, death, p, d):
    """
    takes state vector to differentiate effector cells as linear chain
    needs alpha and beta(r) of response time distribution, probability
    and number of precursor cells
    """
    dt_state = np.zeros_like(state)

    for j in range(len(state)):
        if j == 0:
            dt_state[j] = th0*p*d["decision_time"] - beta*state[j]             
        elif j < alpha:
            dt_state[j] = beta*state[j-1]- beta*state[j]                
        elif j == alpha:
            # the problem with the 4 and 2 is that since differentiation takes 1 day it should divide twice giving 4 cells
            # however, if it has arrived in the final states if should double every half day
            dt_state[j] = beta*state[j-1]+2*beta_p*state[-1] - (death+beta_p)*state[j] 
            
        else:
            assert j > alpha        
            dt_state[j] = beta_p*state[j-1]-(beta_p+death)*state[j] 
                 
    return dt_state


def calc_il2_consumption(c_il2, il2_consumers, d):
    out = c_il2/(d["K_il2_consumption"]+il2_consumers)
    return out


# =============================================================================
# homeostasis moedls
# =============================================================================
def menten(conc, vmax, K, hill):   
    # make sure to avoid numeric errors for menten
    assert conc >= 0
    out = (vmax*conc**hill) / (K**hill+conc**hill)
    
    return out


def no_regulation(beta_p, c_il2, myc, time, d):
    return beta_p


def timer_il2_branched(beta_p, c_il2, myc, time, d):
   
    vmax = beta_p
    hill = d["hill"] 
    myc = myc if myc >= 0 else 0
    # compute beta p as geometric mean of il2 and myc effect
    product = menten(myc, vmax, d["K_myc"], hill)*menten(c_il2, vmax, d["K_il2"], hill)
    beta_p = np.sqrt(product)
    
    return beta_p


def timer_branched(beta_p, c_il2, myc, time, d):
   
    vmax = beta_p
    hill = d["hill"] 
    myc = myc if myc >= 0 else 0
    # compute beta p as geometric mean of il2 and myc effect
    beta_p = menten(myc, vmax, d["K_myc"], hill)
    
    return beta_p


def il2_branched(beta_p, c_il2, myc, time, d):
   
    vmax = beta_p
    hill = d["hill"] 
    # compute beta p as geometric mean of il2 and myc effect
    beta_p = menten(c_il2, vmax, d["K_il2"], hill)
    
    
    return beta_p
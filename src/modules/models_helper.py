from functools import partial
import numpy as np


def fb_fc(conc, gamma, K, hill = 3):
    conc = conc if conc > 0 else 0
    out = (gamma*conc**hill + K**hill) / (K**hill + conc**hill)
    return out

def menten(conc, vmax, K, hill):
    # make sure to avoid numeric errors for menten
    conc = conc if conc > 0 else 0
    out = (vmax * conc ** hill) / (K ** hill + conc ** hill)

    return out

# =============================================================================
# homeostasis models
# =============================================================================
def null_model(th_state, time, d):
    lifetime_eff, beta_p = d["lifetime_eff"], d["beta_p"]
    return lifetime_eff, beta_p


def prolif_wrapper(name, d):
    """
    return proliferation model (il2, myc or timer/il2 model)
    the index is important
    """
    assert name in ["timer", "il2", "timer_il2"]
    if name == "il2":
        idx = 2
        K = d["K_il2"]

        func = partial(menten_prolif, idx = idx, K = K)
    elif name == "timer":
        idx = 1
        K = d["K_myc"]
        func = partial(menten_prolif, idx = idx, K = K)
    else:
        func = menten_prolif2

    return func


def menten_prolif(state, d, idx, K):
    # index will get either il2 or myc
    val = state[-idx]
    vmax = d["beta_p"]
    hill = d["hill"]
    beta_p = menten(val, vmax, K, hill)

    return beta_p


def menten_prolif2(state, d):
    hill = d["hill"]
    vmax = d["beta_p"]
    p_il2 = menten_prolif(state, 2, vmax, d["K_il2"], hill)
    p_myc = menten_prolif(state, 1, vmax, d["K_myc"], hill)
    beta_p = np.sqrt(p_il2 * p_myc)

    return beta_p


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
    c_C = d["rate_C"] / (d["K_C"] + il7_consumers)

    vmax = d["beta_p"]
    K = d["K_C"]
    hill = d["hill"]

    beta_p = menten(c_C, vmax, K, hill)
    lifetime_eff = d["lifetime_eff"]

    return lifetime_eff, beta_p


def C_menten_lifetime(th_state, time, d):
    il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d)
    c_C = d["rate_C"] / (d["K_C"] + il7_consumers)

    vmax = d["lifetime_eff"]
    K = d["K_C"]
    hill = d["hill"]

    lifetime_eff = menten(c_C, vmax, K, hill)
    beta_p = d["beta_p"]

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
    t0 = time - d["t0"]

    if t0 > 0:
        beta_p = beta_p * np.exp(-decay * t0)

    return lifetime_eff, beta_p


def thres_lifetime(d, time):
    lifetime_eff = d["lifetime_eff"]
    beta_p = d["beta_p"]

    decay = 100.
    t0 = time - d["t0"]

    if t0 > 0:
        lifetime_eff = lifetime_eff * np.exp(-decay * t0)

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
        conc_il2 = d["rate_il2"] * il2_producers / (d["K_il2"] + il2_consumers)
        if il7_consumers > 0.01:
            test_thres(conc_il2, d["crit_il2"], time, d)

    return lifetime_eff, beta_p


def il2_thres_lifetime(th_state, time, d):
    lifetime_eff, beta_p = d["lifetime_eff"], d["beta_p"]

    if d["crit"] == True:
        lifetime_eff, beta_p = thres_lifetime(d, time)
    else:
        il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d)
        conc_il2 = d["rate_il2"] * il2_producers / (d["K_il2"] + il2_consumers)
        if il7_consumers > 0.01:
            test_thres(conc_il2, d["crit_il2"], time, d)

    return lifetime_eff, beta_p


def C_thres_prolif(th_state, time, d):
    lifetime_eff, beta_p = d["lifetime_eff"], d["beta_p"]

    if d["crit"] == True:
        lifetime_eff, beta_p = thres_prolif(d, time)
    else:
        il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d)
        conc_C = d["rate_C"] / (d["K_C"] + il7_consumers)
        if il7_consumers > 0.01:
            test_thres(conc_C, d["crit_C"], time, d)

    return lifetime_eff, beta_p


def C_thres_lifetime(th_state, time, d):
    lifetime_eff, beta_p = d["lifetime_eff"], d["beta_p"]

    if d["crit"] == True:
        lifetime_eff, beta_p = thres_lifetime(d, time)
    else:
        il2_producers, il2_consumers, il7_consumers = get_cyto_producers(th_state, d)
        conc_C = d["rate_C"] / (d["K_C"] + il7_consumers)
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
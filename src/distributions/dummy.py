def SD_r0(r0, power = 0.5, gamma = 1):
    # this works surprisingly well
    out = (1/gamma) / (r0**power - 1)
    return out

def mean_r0(r0, power):
    out = 1 / (r0**power - 1)
    return out
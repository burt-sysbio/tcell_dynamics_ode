import numpy as np
import pandas as pd
from scipy.stats import lognorm as log_pdf


def gen_lognorm_params(self, pname, std, n=20):
    mean = self.parameters[pname]
    sigma, scale = lognorm_params(mean, std)
    sample = log_pdf.rvs(sigma, 0, scale, size=n)

    return sample


def draw_new_params(sim, param_names, heterogeneity):
    for param in param_names:
        mean = self.parameters[param]
        std = mean * (heterogeneity / 100.)
        sigma, scale = lognorm_params(mean, std)
        sample = log_pdf.rvs(sigma, 0, scale, size=1)
        self.parameters[param] = sample


def lognorm_params(mode, stddev):
    """
    Given the mode and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    p = np.poly1d([1, -1, 0, 0, -(stddev/mode)**2])
    r = p.roots
    sol = r[(r.imag == 0) & (r.real > 0)].real
    shape = np.sqrt(np.log(sol))
    scale = mode * sol
    return shape, scale


def gen_arr(sim, pname, scales = (1,1), n = 30):
    edge_names = ["alpha", "alpha_1", "alpha_p"]
    if pname in edge_names:
        arr = np.arange(2, 20, 2)
    else:
        params = sim.parameters
        val = params[pname]
        val_min = 10**(-scales[0])*val
        val_max = 10**scales[1]*val
        arr = np.geomspace(val_min, val_max, n)
    return arr
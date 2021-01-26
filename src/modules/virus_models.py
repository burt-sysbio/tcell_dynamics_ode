from scipy.stats import lognorm, gamma
from scipy.interpolate import interp1d
import numpy as np
from scipy.integrate import odeint

def sir_parameterization(r0, a, b):
    assert r0>=1
    return 1 / (a*r0**b - 1)


def vir_model_SIR(time, d):
    r0 = d["SIR_r0"]
    SD = sir_parameterization(r0, 1.02, 0.37)
    mean = sir_parameterization(r0, 1.05, 0.18)

    shape, scale = get_lognormdist_params(mean, SD)
    mylognorm = lognorm(s=shape, scale = scale)

    def f(t):
        return mylognorm.pdf(t)
    return f


def get_lognormdist_params(mode, stddev):
    """
    Given the mode and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    p = np.poly1d([1, -1, 0, 0, -(stddev/mode)**2])
    r = p.roots
    sol = r[(r.imag == 0) & (r.real > 0)].real
    sol = float(sol)
    shape = np.sqrt(np.log(sol))
    scale = mode * sol
    return shape, scale


def vir_model_const(time, d):
    """
    ag level only depends on vir load, not on time
    """
    time = np.arange(np.min(time), 5 * np.max(time), 0.01)
    s = np.ones_like(time)
    s = s*d["vir_load"]
    f = interp1d(time, s, kind = "zero")
    return f


def vir_model_gamma(time, d):
    """
    should return function object that accepts single argument time
    """
    alpha = d["vir_alpha"]
    beta = d["vir_beta"]
    scale = 1/beta
    mygamma = gamma(a = alpha, scale = scale)
    def f(t):
        return mygamma.pdf(t) * d["vir_load"]
    return f


def vir_model_ode(time, d):
    """
    should return function object that accepts single argument time
    solves ode and return interpolated normalized function object
    """
    y0 = 1
    norm = True

    # extend time array because ode solver sometimes jumps above provided range
    time = np.arange(np.min(time), 5*np.max(time), 0.01)

    def vir_ode(v, t, d):
        dv = (d["vir_growth"]- t*d["vir_death"])*v
        return dv

    s = odeint(vir_ode, y0, time, args=(d,))
    s = s.flatten()
    # normalize by area
    if norm:
        area = np.trapz(s, time)
        assert area != 0
        s = s/area
    f = interp1d(time, s)
    return f
import numpy as np
from scipy.stats import lognorm
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style = "ticks", context = "poster")

x = np.linspace(2,20,200)

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

shape1, scale1 = lognorm_params(10, 1)
shape2, scale2 = lognorm_params(10, 3)


loc = 0
fig, ax = plt.subplots()
ax.plot(x, lognorm.pdf(x, s = shape1, scale = scale1, loc = loc), "k", label = "low heterogeneity")
ax.plot(x, lognorm.pdf(x, s = shape2,  scale = scale2, loc = loc), "k", ls = "--", label = "high heterogeneity")
ax.legend()
ax.set_xlabel(r"$\beta_p$")
ax.set_ylabel("density")
plt.show()
#fig.savefig("lognorm_distribution.svg")
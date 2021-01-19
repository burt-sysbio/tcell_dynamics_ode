import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats as ss
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib
sns.set(style = "ticks", context = "poster")

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def run_pipeline(y0, t, N, beta, gamma):
    # run ODE
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    # norm ODE result and fit to lognorm dist
    I_norm = I / np.trapz(I, t)
    p, o = curve_fit(myfun, t, I_norm, bounds=(0, np.inf))

    # get fit params and compute lognorm params
    shape, scale = p
    sigma, mu = shape, np.log(scale)
    median_lognorm = np.exp(mu)
    mean_lognorm = np.exp(mu+ sigma**2 /2)
    var_lognorm = (np.exp(sigma**2)-1)*np.exp(2*mu+sigma**2)
    sd_lognorm = np.sqrt(var_lognorm)
    # get fit vals and residuals
    yfit = myfun(t, *p)
    res = I_norm - yfit
    rmse = np.sqrt(np.sum(res**2) / len(t))
    df = pd.DataFrame({"S" : S, "I" : I, "R" : R, "I_norm" : I_norm, "yfit" : yfit,
                       "time" : t})

    # prep output data frame
    df = df.melt(id_vars = ["time"], var_name= "species")
    df["beta"] = beta
    df["gamma"] = gamma

    df1 = df[df.species.isin(["S", "I", "R"])]
    df2 = df[df.species.isin(["I_norm", "yfit"])]
    df2["fit_error"] = rmse
    df2["mean_lognorm"] = mean_lognorm
    df2["SD_lognorm"] = sd_lognorm
    df2["median_lognorm"] = median_lognorm
    df2["CV"] = sd_lognorm / mean_lognorm
    df2["lognorm_shape"] = shape
    df2["lognorm_scale"] = scale
    return df1, df2


def myfun(x, shape, scale):
    f = ss.lognorm.pdf(x, s = shape, loc = 0, scale = scale)
    return f

# parameters
N = 1 # N is set to 1 as population density
I0, R0 = 0.01, 0 # Initial number of infected and recovered individuals, I0 and R0.
S0 = N - I0 - R0 # Everyone else, S0, is susceptible to infection initially.
beta = 1.2
gamma = 1
t = np.linspace(0, 500, 5000) # time

# run ODE and plot output
y0 = [S0, I0, R0]
df1, df2 = run_pipeline(y0, t, N, beta, gamma)
g = sns.relplot(data = df2, x = "time", y = "value", hue = "species", kind = "line")
#plt.show()

# run ODE and plot output for different betas
df_list = []
beta_arr = [2,3,8]
for b in beta_arr:
    df1, df2 = run_pipeline(y0, t, N, b, gamma)
    df2["r0"] = np.round(b/gamma, 2)
    df_list.append(df2)
df = pd.concat(df_list)
g = sns.relplot(data = df, x = "time", y = "value", hue = "species", col = "r0",
                kind = "line", facet_kws= {"sharey" : False})
g.set(xlim = [0,50])
#plt.show()
#g.savefig("../figures/antigen_effects/inf_good_range.pdf")

# run ode and plot lognorm fit params for different betas
beta_arr = np.geomspace(gamma, 10*gamma,100)
r0_arr = beta_arr/gamma
df_list = []
for b in beta_arr:
    df1, df2 = run_pipeline(y0,t, N, b, gamma)
    df_list.append(df2)

df = pd.concat(df_list)
df = df[["beta", "SD_lognorm", "mean_lognorm", "CV", "fit_error", "lognorm_shape", "lognorm_scale"]]
df = df.drop_duplicates()
df = df.melt(id_vars= "beta")
df["r0"] = df.beta/gamma

g = sns.relplot(data = df, x = "r0", y = "value", col = "variable", col_wrap= 3,
                facet_kws= {"sharey" : False})

#g.set(xscale = "log")
g.set_titles("{col_name}")
plt.show()
g.savefig("../figures/antigen_effects/SIR_lognorm_fits.pdf")



# take result of pscan and parameterize functions
def fit_SD(x, a, b):
    out = 1 / (a*x**b - 1)
    return out


def fit_mean(x, power):
    out = 1 / (x**power - 1)
    return out


def SIR_fit(df, r0_arr, fit_param, fit_fun):
    ydata = df.loc[df.variable == fit_param, "value"]
    ydata = ydata.values
    popt, pcov = curve_fit(fit_fun, r0_arr[1:], ydata[1:])
    yfit = fit_fun(r0_arr[1:], *popt)
    return ydata, yfit, popt


params = ["SD_lognorm", "mean_lognorm"]
for p in params:
    ydata, yfit, popt = SIR_fit(df, r0_arr, p, fit_SD)
    fig, ax = plt.subplots()

    ax.scatter(r0_arr[1:], ydata[1:], label = p +" - SIR fit",
               s = 10.0)
    ax.plot(r0_arr[1:], yfit, label = "1/(a*x**b-1)", c = "tab:orange", lw = 2)
    ax.set_xlabel("r0")
    ax.set_ylabel("value")
    ax.set_ylim(0,20)
    ax.set_title(f"a={np.round(popt[0],2)}, b={np.round(popt[1],2)}")
    ax.legend()
    plt.show()
    fig.savefig("../figures/antigen_effects/r0_parameterization_"+p+".pdf")
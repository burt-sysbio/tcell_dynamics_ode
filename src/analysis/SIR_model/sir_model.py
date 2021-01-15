import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit
import scipy.stats as ss
import seaborn as sns

sns.set(style = "ticks", context = "poster")
# Total population, N.
N = 100
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 1,0.1
# A grid of time points (in days)
t = np.linspace(0, 40, 500)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111)
ax.plot(t, S/N, 'b', label='Susceptible')
ax.plot(t, I/N, 'r', label='Infected')
ax.plot(t, R/N, 'g', label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number')
plt.show()


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
    mean_lognorm = np.exp(mu+ sigma**2 /2)
    var_lognorm = (np.exp(sigma**2)-1)*np.exp(2*mu+sigma**2)

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
    df2["error"] = rmse
    df2["mean_lognorm"] = mean_lognorm
    df2["SD_lognorm"] = np.sqrt(var_lognorm)
    return df1, df2


def myfun(x, shape, scale):
    f = ss.lognorm.pdf(x, s = shape, loc = 0, scale = scale)
    return f


df1, df2 = run_pipeline(y0, t, N, beta, gamma)

g = sns.relplot(data = df2, x = "time", y = "value", hue = "species", kind = "line")
plt.show()

df_list = []
beta_arr = [0.5,1,4]
for b in beta_arr:
    df1, df2 = run_pipeline(y0, t, N, b, gamma)
    df_list.append(df2)

df = pd.concat(df_list)

g = sns.relplot(data = df, x = "time", y = "value", hue = "species", col = "beta",
                kind = "line")
plt.show()

beta_arr = np.arange(1,10,0.1)

df_list = []
for b in beta_arr:
    df1, df2 = run_pipeline(y0,t, N, b, gamma)
    df_list.append(df2)

df = pd.concat(df_list)
df = df[["beta", "SD_lognorm", "mean_lognorm", "error"]]
df = df.drop_duplicates()
df = df.melt(id_vars= "beta")

g = sns.relplot(data = df, x = "beta", y = "value", col = "variable",
                facet_kws= {"sharey" : False})
plt.show()
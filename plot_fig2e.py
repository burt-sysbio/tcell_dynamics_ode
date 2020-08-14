import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style = "ticks", context = "poster")

df1 = pd.read_csv("readouts_il2.csv")
df1 = df1.iloc[:,1:]
df2 = pd.read_csv("readouts_timer.csv")
df2 = df2.iloc[:,1:]
df1["sim"] = "IL2"
df2["sim"] = "Timer+IL2"

df = pd.concat([df1, df2])
df_tidy = pd.melt(df, id_vars = ["sim", "CV"], value_name = "effect_size", var_name = "readout")

g = sns.relplot(data = df_tidy, x = "CV", y = "effect_size",
                col = "readout", hue = "sim",
                facet_kws= {"sharey" : False})
g.set(xscale = "log")
plt.show()
g.savefig("plot_fig2e_readouts.pdf")

# timecourse plot
df = pd.read_csv("fig2e_timecourse.csv")
g = sns.relplot(data = df, x = "time", y = "cells", hue = "model_name", col = "sd", kind = "line",
                ci = "sd")

g.set_titles("{col_name}")
plt.show()

g.savefig("plot_fig2e_timecourse.pdf")
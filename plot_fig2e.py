import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style = "ticks", context = "poster")

pname = "rate_il2"
df_reads = pd.read_csv("data_fig2e_readouts_"+pname+".csv")
df_timecourse = pd.read_csv("data_fig2e_timecourse_"+pname+".csv")

df_samples = pd.read_csv("data_fig2e_lognorm_samples_"+pname+".csv")
# kick out index column
#
df_tidy = pd.melt(df_reads, id_vars = ["name", "CV"], value_name = "effect_size",
                   var_name = "readout")
#
g = sns.relplot(data = df_tidy, x = "CV", y = "effect_size",
                 col = "readout", hue = "name",
                 facet_kws= {"sharey" : False})
g.set(xscale = "log", xlim = (0.1,10), ylabel = "effect size")
g.set_titles("{col_name}")
plt.show()
g.savefig("fig2e_readouts.pdf")

reads_filter = ["Mean Peak Height", "SD Peak Height", "Mean Response Size"]
df_tidy_filtered = df_tidy[df_tidy.readout.isin(reads_filter)]

g = sns.relplot(data = df_tidy_filtered, x = "CV", y = "effect_size",
                 col = "readout", hue = "name", palette= ["k", "Grey"],
                 facet_kws= {"sharey" : False})
g.set(xscale = "log", xlim = (0.1,10), ylabel = "effect size")
g.set_titles("{col_name}")
plt.show()
g.savefig("fig2e_readouts_filtered.svg")


# timecourse plot
g = sns.relplot(data = df_timecourse, x = "time", y = "cells", row = "model_name",
                col = "CV", kind = "line", ci = "sd", color = "k")
g.set(xlim=(0,8), ylabel = "effector cells")
g.set_titles("")
plt.show()

g.savefig("fig2e_timecourse.pdf")

#sns.set(context="paper", style = "ticks")
fig, ax = plt.subplots(1,3, figsize = (14,4))
histo = df_samples.hist(ax = ax, color = "Grey", grid = False)
for a in ax:
    a.set_xlabel("IL2 secretion rate")

for a,s in zip(ax, df_samples.columns):
    a.set_title("CV=%s" %s)

plt.show()

fig.savefig("fig2e_histo.pdf")
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
#g.savefig("fig2e_readouts.pdf")

reads_filter = ["Mean Peak Height", "SD Peak Height"]
df_tidy_filtered = df_tidy[df_tidy.readout.isin(reads_filter)]

g = sns.relplot(data = df_tidy_filtered, x = "CV", y = "effect_size",
                 col = "readout", hue = "name", facet_kws= {"sharey" : False})
g.set(xscale = "log", xlim = (0.1,10), ylabel = "effect size")
g.set_titles("{col_name}")
plt.show()
g.savefig("fig2e_readouts_filtered.svg")


# timecourse plot
g = sns.relplot(data = df_timecourse, x = "time", y = "cells", row = "model_name",
                col = "CV", kind = "line", ci = "sd")
g.set(xlim=(0,8), ylabel = "effector cells")
g.set_titles("")
plt.show()

g.savefig("fig2e_timecourse.pdf")

#sns.set(context="paper", style = "ticks")
fig, axes = plt.subplots(1,3, figsize = (14,4))
bins = [5,8,20]
for i, bin, s in zip(range(3), bins, df_samples.columns):
    x = df_samples.iloc[:,i]
    axes[i].hist(x, bins = bin, weights=np.zeros_like(x) + 1. / x.size)
    axes[i].set_xlim(0,3.5)
    axes[i].set_ylim(0,0.6)
    axes[i].set_ylabel("rel. frequency")
    axes[i].set_xlabel("IL2 secretion rate")
    axes[i].set_title("CV=%s" %s)
plt.show()


fig.savefig("fig2e_histo.pdf")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style = "ticks", context = "poster")

pname = "up_il2"
df_reads = pd.read_csv("data_fig2e_readouts_"+pname+".csv")
df_timecourse = pd.read_csv("data_fig2e_timecourse_"+pname+".csv")
df_samples = pd.read_csv("data_fig2e_lognorm_samples_"+pname+".csv")
# kick out index column

df_tidy = pd.melt(df_reads, id_vars = ["name", "CV"], value_name = "effect_size",
                  var_name = "readout")

g = sns.relplot(data = df_tidy, x = "CV", y = "effect_size",
                col = "readout", hue = "name",
                facet_kws= {"sharey" : False})
g.set(xscale = "log")
plt.show()
#g.savefig("plot_fig2e_readouts.pdf")

# timecourse plot
g = sns.relplot(data = df_timecourse, x = "time", y = "cells", hue = "model_name",
                col = "CV", kind = "line", ci = "sd")
plt.show()

#g.savefig("plot_fig2e_timecourse.pdf")

fig, ax = plt.subplots(1,len(df_samples.columns), figsize = (14,3))
hist = df_samples.hist(ax = ax)
plt.show()


#fig.savefig("plot_fig2e_histo.pdf")
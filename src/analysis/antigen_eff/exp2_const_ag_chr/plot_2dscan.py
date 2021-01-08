import numpy as np
import src.modules.pl as pl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


file1 = "../output/fig3/ag_pscan2d_gamma_il2.csv"
file2 = "../output/fig3/ag_pscan2d_gamma_timer.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

readouts = ["Area", "Peak", "Peaktime"]
df_list = [df1, df2]
for r in readouts:
    for df in df_list:
        pl.plot_heatmap(df, "norm_min", r, log_color=False)
        plt.show()


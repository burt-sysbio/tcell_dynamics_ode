# plot different virus dynamics

from src.modules.models import vir_model_ode, vir_model_gamma
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = "poster", style = "ticks")

time = np.arange(0, 3, 0.01)
vir_model = vir_model_gamma

d1 = {
    "vir_beta": 2,
    "vir_alpha": 2,
    "vir_load" : 1,
    "vir_growth" : 1,
    "vir_death" : 1.,
}
d2 = {
    "vir_beta": 4,
    "vir_alpha": 4,
    "vir_load": 1,
    "vir_growth": 1,
    "vir_death": 0.5,
}
d3 = {
    "vir_beta": 10,
    "vir_alpha": 10,
    "vir_load": 1,
    "vir_growth": 1,
    "vir_death": 2,
}

dics = [d1,d2,d3]

fig, ax = plt.subplots()
for d in dics:
    f = vir_model(time, d)
    vir = f(time)
    ax.plot(time, vir)
    ax.set_xlabel("time")
    ax.set_ylabel("conc. virus")

plt.tight_layout()
plt.show()
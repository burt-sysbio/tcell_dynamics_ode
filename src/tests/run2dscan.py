import numpy as np
from src.modules.exp import Sim, Simlist
from src.tests.parameters import d
from src.modules.models import vir_model_gamma, vir_model_ode
import src.modules.proc as proc
from joblib import Parallel, delayed
import multiprocessing

# init model
vir_model = vir_model_gamma
time = np.arange(0, 40, 0.01)
pname1 = "vir_alpha"
pname2 = "vir_beta"

# init models
sim = Sim(name = "il2", params = d, time = time, virus_model= vir_model)

res = 20
inputs = proc.get_2dinputs(prange1=(1,100), prange2=(0.1,100), res = res)

n_cor = multiprocessing.cpu_count()
params = [sim, pname1, pname2]

if __name__ == "__main__":
    outputs = Parallel(n_jobs=n_cor)(delayed(proc.get_2doutput)(i, *params) for i in inputs)
    out = proc.get_2dscan(outputs)

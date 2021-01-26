import numpy as np
from copy import deepcopy
from scipy.integrate import odeint
from src.modules.data_model import model_2021_py as model2021
import pandas as pd
import pickle

class Sim:
    def __init__(self,
                 name,
                 time,
                 params,
                 virus_model,
                 model = model2021,
                 n_states = 15):
        self.model = model
        self.default_params = deepcopy(params)
        self.params = params
        self.n = n_states
        self.name = name
        self.time = time
        self.virus_model = virus_model

    def init_model(self):
        y0 = np.zeros(self.n)
        y0[0] = self.params["initial_cells"]
        y0[-1] = 1 # myc
        y0[-2] = 1 # myc prec
        return y0

    def run_sim(self):
        time = self.time
        y0 = self.init_model()
        # initialize virus model for given parameter setting, returns interpolated function
        vir_model = self.virus_model(time, self.params)
        state = odeint(self.model, y0, time, args = (self.params, vir_model))

        # convert to df
        colnames = ["Naive", "Prec", "Prec1", "Th1", "Th1_2", "Tfh", "Tfh_2", "Tfhc", "Tfhc_2",
                    "Tr1", "Tr1_2", "Th1_mem", "Tfh_mem", "Myc", "Myc_prec"]
        df = pd.DataFrame(state, columns = colnames)

        # split the data
        cells = df.iloc[:,:-2]
        molecules = df.iloc[:,-2:]

        # add virus and additional columns
        molecules["Virus"] = vir_model(time)
        cells["time"] = time
        molecules["time"] = time
        molecules["name"] = self.name
        cells["name"] = self.name

        # tidy data
        cells = self.compute_cell_states(cells)
        cells = cells.melt(id_vars=["time", "name"], var_name="cell")
        molecules = molecules.melt(id_vars=["time", "name"], var_name="cell")

        return cells, molecules

    def compute_cell_states(self, df):
        """
        # takes data frame and computes cell states
        !!!! This might need adjustment depending on the antimony model

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """
        df["Precursors"] = df.Prec + df.Prec1
        df["Th1_eff"] = df.Th1 + df.Th1_2 + df.Th1_mem
        df["Tfh_eff"] = df.Tfh + df.Tfh_2 + df.Tfh_mem
        df["Tr1_all"] = df.Tr1 + df.Tr1_2
        df["nonTfh"] = df.Th1_eff + df.Tr1_all
        df["Tfh_chr"] = df.Tfhc + df.Tfhc_2
        df["Tfh_all"] = df.Tfh_chr + df.Tfh_eff
        df["Th_chr"] = df.Tfh_chr + df.Tr1_all
        df["Total_CD4"] = df.Precursors + df.Th1_eff + df.Tfh_all + df.Tr1_all
        df["Th_mem"] = df.Th1_mem + df.Tfh_mem
        df["Th_eff"] = df.Th1_eff + df.Tfh_eff

        return df


    def reset(self):
        self.params = deepcopy(self.default_params)

    def set_fit_params(self, fit_date, fit_name):
        # set model parameters to fit
        # load fit result
        fit_dir = "../../output/fit_results/" + fit_date + "/"
        with open(fit_dir + fit_name + '.p', 'rb') as fp:
            fit_result = pickle.load(fp)

        for key, val in fit_result.items():
            self.params[key] = val



import numpy as np
import pandas as pd
from src.modules.readout_module import *
from itertools import product


def run_timecourses(sim, arr, pname):
    old_parameters = dict(sim.params)

    cell_list = []
    mol_list = []
    for a in arr:
        sim.params[pname] = a
        cells, molecules = sim.run_sim()
        cells["pname"] = pname
        cells["param_value"] = a
        molecules["pname"] = pname
        molecules["param_value"] = a
        cell_list.append(cells)
        mol_list.append(molecules)

    sim.parameters = old_parameters

    cells = pd.concat(cell_list)
    molecules = pd.concat(mol_list)
    return cells, molecules


def norm_molecules(df, d, name = "il2"):
    mols = ["IL2", "MYC", "Virus"]
    if name == "il2":
        params = ["K_il2", "K_myc", "K_ag_il2"]
    else:
        params = ["K_il2", "K_myc", "K_ag_myc"]

    df["EC50"] = 1
    # assign corresponding EC50 value to each molecules and normalize
    for m, p in zip(mols, params):
        df.loc[df["cell"] == m, ["EC50"]] = d[p]

    df["value"] =df.value.values / df.EC50.values

    return df


def pscan(sim, arr, pname, crit_fun=check_criteria2):
    old_parameters = dict(sim.params)

    def myfun(sim, a, pname):
        sim.params[pname] = a
        cells, molecules = sim.run_sim()

        r = get_readouts(cells, crit_fun=crit_fun)
        r["param_value"] = a
        return r

    reads = [myfun(sim, a, pname) for a in arr]
    reads = pd.concat(reads)
    reads["param"] = pname
    reads["name"] = sim.name
    sim.parameters = old_parameters

    ## need to make this tidy
    reads["val_norm"] = reads.groupby(["cell", "readout"])["value"].transform(lambda x: np.log2(x / x.median()))
    reads["val_min"] = reads.groupby(["cell", "readout"])["value"].transform(lambda x: np.log2(x / x.min()))

    return reads


def get_2doutput(input, sim, pname1, pname2, crit_fun=check_criteria2):
    """
    only use withon pscan2d
    take sim object, change two parameters run sim and get readouts
    """
    # change both parameters run sim and get readouts
    p1, p2 = input
    sim.params[pname1] = p1
    sim.params[pname2] = p2
    cells, molecules = sim.run_sim()
    r = get_readouts(cells, crit_fun=crit_fun)

    # add parameter values
    r["pval1"] = p1
    r["pval2"] = p2
    r["pname1"] = pname1
    r["pname2"] = pname2

    return r


def get_2dinputs(prange1, prange2, res, rangefun=np.linspace):
    """
    p1 : tuple (min, max) of param range for pname1
    p2 : tuple (min, max) of param range for pname2
    post process with plot heatmap
    """
    # generate arrays and get cartesian product
    arr1 = rangefun(prange1[0], prange1[1], res)
    arr2 = rangefun(prange2[0], prange2[1], res)
    inputs = product(arr1, arr2)
    return inputs


def get_2dscan(outputs):
    # get readouts for each cart. prod. of param comb.
    out = pd.concat(outputs)
    # normalize to median
    out["val_norm"] = out.groupby(["cell", "readout"]).value.transform(lambda x: np.log2(x / x.median()))
    out["norm_min"] = out.groupby(["cell", "readout"]).value.transform(lambda x: np.log2(x / x.min()))
    return out


def get_readouts(df, crit_fun):
    cols = ["time", "name", "cell", "value"]

    assert all(col in df.columns for col in cols)
    if len(df.columns) > len(cols):
        print("multiple columns detected, please double check if Simlist was used")

    # helper function, this is applied for each cell to get all readouts

    def f(df):
        funs = [get_peak_height, get_area, get_peaktime, get_duration]

        # check readout quality criteria for every cell type and only
        # get readouts if criteria are True or crit_fun is None
        if crit_fun is not None:
            if not crit_fun(df):
                reads = np.empty(len(funs))
                reads[:] = np.nan
            else:
                reads = [fun(df.time, df.value) for fun in funs]
        else:
            reads = [fun(df.time, df.value) for fun in funs]

        read_names = ["Peak", "Area", "Peaktime", "Decay"]
        s = pd.Series(reads, index=read_names)
        return s

    out = df.groupby("cell").apply(f)
    out = out.reset_index()
    out = pd.melt(out, id_vars="cell", var_name="readout")
    return out


def vary_param_norm(df, arr, edge_names, normtype):
    """
    deprecated, pscan can do this now. leave for earlier scripts that might need it
    take df from pscan and normalize
    Parameters
    ----------
    readout_list : list
        readouts for diff param values.
    arr : array
        parameter values.
    edgenames : list of strings
        parameter names.
    normtype : string, should be either "first" or "middle"
        normalize to middle or beginning of arr.

    Returns
    -------
    df : data frame
        normalized readouts
    """
    # double check which columns are in data frame is also important for merging dfs later
    assert (df.columns == ["value", "readout", "param_value", "pname", "name"])
    # merge df with normalization df
    norm = arr[int(len(arr) / 2)]
    if normtype == "first":
        norm = arr[0]
    df2 = df[df.param_value == norm]

    df2 = df2.rename(columns={"value": "ynorm"})
    df2 = df2.drop(columns=["param_value"])

    df = df.merge(df2, on=['readout', 'name'], how='left')

    # compute log2FC
    logseries = df["value"] / df["ynorm"]
    logseries = logseries.astype(float)

    df["log2FC"] = np.log2(logseries)
    df = df.drop(columns=["ynorm"])

    # add xnorm column to normalise x axis for param scans
    df["xnorm"] = df["param_value"] / norm

    if pname in edge_names:
        df["param_value"] = df["param_value"] / (df["param_value"] * df["param_value"])

    return df

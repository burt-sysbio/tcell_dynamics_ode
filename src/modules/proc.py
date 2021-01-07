import numpy as np
import pandas as pd
from src.modules.readout_module import *
from itertools import product

def pscan(sim, arr, pname):
    old_parameters = dict(sim.params)

    def myfun(sim, a, pname):
        sim.params[pname] = a
        cells, molecules = sim.run_sim()
        r = get_readouts(cells)
        r["param_value"] = a
        return r

    reads = [myfun(sim, a, pname) for a in arr]
    reads = pd.concat(reads)
    reads["param"] = pname
    reads["name"] = sim.name
    sim.parameters = old_parameters

    ## need to make this tidy
    reads["val_norm"] = reads.groupby(["cell", "readout"])["value"].transform(lambda x: np.log2(x/x.median()))
    return reads


def get_2doutput(input, sim, pname1, pname2):
    """
    only use withon pscan2d
    take sim object, change two parameters run sim and get readouts
    """
    # change both parameters run sim and get readouts
    p1,p2 = input
    sim.params[pname1] = p1
    sim.params[pname2] = p2
    cells, molecules = sim.run_sim()
    r = get_readouts(cells)

    # add parameter values
    r["pval1"] = p1
    r["pval2"] = p2
    r["pname1"] = pname1
    r["pname2"] = pname2
    return r


def get_2dinputs(prange1, prange2, res, rangefun = np.geomspace):
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
    out["val_norm"] = out.groupby(["cell", "readout"]).value.transform(lambda x: np.log2(x/x.median()))
    out["norm_min"] = out.groupby(["cell", "readout"]).value.transform(lambda x: np.log2(x / x.min()))
    return out


def get_readouts(df):
    cols = ["time", "name", "cell", "value"]
    assert (df.columns == cols).all()
    # helper function, this is applied for each cell to get all readouts
    def f(df):
        funs = [get_peak_height, get_area, get_peaktime, get_duration]
        thres = 1e-1
        if (df.value>thres).any():
            reads = [fun(df.time, df.value) for fun in funs]
        else:
            #print(df)
            reads = np.empty(len(funs))
            reads[:] = np.nan
            #reads = [0, 0, 0, 0]

        read_names = ["Peak", "Area", "Peaktime", "Decay"]
        s = pd.Series(reads, index = read_names)
        return s

    out = df.groupby("cell").apply(f)
    out = out.reset_index()
    out = pd.melt(out, id_vars = "cell", var_name= "readout")
    return out


def vary_param_norm(df, arr, edge_names, normtype):
    """
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
import numpy as np
import pandas as pd
from src.modules.readout_module import *


def pscan(sim, arr, pname):
    old_parameters = dict(sim.params)
    reads = []
    for a in arr:
        sim.params[pname] = a
        cells, molecules = sim.run_sim()
        r = get_readouts(cells)
        r["param_value"] = a
        reads.append(r)

    reads = pd.concat(reads)
    reads["param"] = pname
    reads["name"] = sim.name
    sim.parameters = old_parameters

    ## need to make this tidy
    return reads


def get_readouts(df):
    cols = ["time", "name", "cell", "value"]
    assert (df.columns == cols).all()

    # helper function, this is applied for each cell to get all readouts
    def f(df):
        funs = [get_peak_height, get_area, get_peaktime, get_duration]
        reads = [fun(df.time, df.value) for fun in funs]
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
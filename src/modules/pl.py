import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib

sns.set(context = "poster", style = "ticks")


def plot_timecourse(df, cells = ["teff", "tchronic", "all_cells"], hue = None, col = None,
                    row = None, yscale = "log", style = None, palette = None, *kwargs):
    """
    take either cells or molecules from run sim object
    """
    # provide cells to plot in cells array
    df = df.loc[df.cell.isin(cells)]
    # only focus on effector cells, not chronic and total cells
    g = sns.relplot(data=df, x="time", palette = palette, hue = hue, col=col, row = row,
                    y="value", style = style, kind="line", *kwargs)
    ylim = (1e-1, None) if yscale == "log" else (None, None)
    g.set(yscale=yscale, ylim=ylim, ylabel = "cells")
    g.set_titles("{col_name}")
    return g


def plot_pscan(df, xscale = "log", yscale = None, cells = ["teff", "tchronic", "all_cells"], hue = None,
               value_col = "val_norm", col = "readout", row = None, palette = None,
               **kwargs):
    """
    take df generated through pscan function
    """
    if (len(cells) > 1) & (hue is None):
        hue = "cell"
    df = df.loc[df.cell.isin(cells) & (df.readout != "Decay")]
    g = sns.relplot(data = df, x = "param_value", y = value_col, col = col, row = row,
                    hue = hue, facet_kws= {"sharey" : False}, kind = "scatter", palette= palette,
                    **kwargs)

    g.set(xlabel = df.param.iloc[0], ylabel = "effect size", xscale = xscale,
          xlim = (df.param_value.min(), df.param_value.max()))

    if yscale is not None:
        g.set(yscale = "log")
    g.set_titles("{col_name}")
    return g


def plot_heatmap(df, value_col, readout, log_color,
                 vmin=None, vmax=None, cmap="Reds", log_axes=True):
    """
    take df generated from 2dscan and plot single heatmap for a given readout
    note that only effector cells are plotted
    value_col: could be either val norm or value as string
    log_color: color representation within the heatmap as log scale, use if input arr was log scaled
    """
    # process data (df contains all readouts and all cells
    df = df.loc[(df.cell == "teff") & (df.readout == readout)]
    arr1 = df.pval1.drop_duplicates()
    arr2 = df.pval2.drop_duplicates()
    assert (len(arr1) == len(arr2))

    # arr1 and arr2 extrema are bounds, and z should be inside those bounds
    z_arr = df[value_col].values
    z = z_arr.reshape((len(arr1), len(arr2)))
    z = z[:-1, :-1]
    # transform because reshape somehow transposes this
    z=z.T
    # check if color representation should be log scale
    sm, norm = get_colorscale(log_color, cmap, vmin, vmax)

    # plot data
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pcolormesh(arr1, arr2, z, norm = norm, cmap=cmap)

    # tick reformatting
    loc_major = ticker.LogLocator(base=10.0, numticks=100)
    loc_minor = ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1, 0.1), numticks=12)

    # adjust scales
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        #ax.xaxis.set_major_locator(loc_major)
        #ax.xaxis.set_minor_locator(loc_minor)
    ax.set_xlabel(df.pname1.iloc[0])
    ax.set_ylabel(df.pname2.iloc[0])

    cbar = plt.colorbar(sm, ax=ax)
    cbar_label = readout + " logFC"
    cbar.set_label(cbar_label)
    plt.tight_layout()

    return fig


def get_colorscale(hue_log, cmap, vmin = None, vmax = None):
    if hue_log:
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # make mappable for colorbar
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    return sm, norm


def plot_timecourses(df,
                     hue_log,
                     pl_cbar,
                     cells = None,
                     scale=None,
                     xlim=(None, None),
                     ylim=(None, None),
                     cmap="Greys",
                     cbar_scale=1.,
                     ticks=None,
                     hue = "param_value",
                     vmin = None,
                     vmax = None,
                     sharey = False,
                     ylabel = "cell FC d0",
                     leg_title = None,
                     show_titles = True,
                     **kwargs):
    """
    pl_cbar: bool - draw color bar?
    hue_log : bool - use log scal for color mapping
    cells : list - which cell types?
    leg_title - str - legend title

    scale - None or "log" - yscale
    ticks : True if colorbar ticks should be adjusted
    """
    if cells is not None:
            df = df.loc[df.cell.isin(cells)]

    # parameter for scaling of color palette in sns plot
    pname = df.pname.iloc[0]
    arr = df.param_value.drop_duplicates()

    if (cmap == "Greys") or (cmap == "Blues"):
        sm, hue_norm = get_colorscale(hue_log, cmap, vmin, vmax)
    else:
        sm, hue_norm = None, None

    # hue takes the model name, so this should be a scalar variable
    # can be generated by change_param function

    g = sns.relplot(x="time", y="value", kind="line", data=df, hue=hue,
                    hue_norm=hue_norm, palette=cmap,
                    facet_kws={"despine": False, "sharey" : sharey}, **kwargs)

    g.set(xlim=xlim, ylim=ylim)
    if scale is not None:
        g.set(yscale = "log")

    ax = g.axes[0][0]
    ax.set_ylabel(ylabel)

    if show_titles:
        g.set_titles("{col_name}")
    if leg_title is not None:
        g._legend.set_title(leg_title)

    # if ticks are true take the upper lower and middle part as ticks
    # for colorbar
    if pl_cbar:
        assert sm is not None
        if ticks == True:
            if hue_log:
                ticks = np.geomspace(np.min(arr), np.max(arr), 3)
            else:
                ticks = np.linspace(np.min(arr), np.max(arr), 3)

            cbar = g.fig.colorbar(sm, ticks=ticks)
            cbar.ax.set_yticklabels(np.round(cbar_scale * ticks, 2))
        else:
            cbar = g.fig.colorbar(sm, ticks=ticks)
        # add colorbar

        cbar.set_label(pname)
        cbar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    return g

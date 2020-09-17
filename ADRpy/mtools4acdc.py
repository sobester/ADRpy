#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. _mtools4acdc_module:

Miscellaneous Utilities
-----------------------

This module contains miscellaneous tools to support aircraft
engineering calculations and data analysis.

"""

# pylint: disable-msg=W0102, R0913
# W0102: erroneous warning from pylint, suggesting '[]' default
# R0913: number of inputs (panelplot) - this feels more readable
# pylint: disable=locally-disabled, too-many-locals

import math
from numbers import Number
import numpy as np
import matplotlib.pyplot as plt
import pandas
from cycler import cycler


def panelplot_with_shared_y(vaxis, haxes, hlimits, vlabel,
                            hlabels, hlines, hlinecols,
                            figpar=[10, 6, 100],
                            tex=False, fam='sans-serif'):
    """Multi-panel plots with a shared y-axis, e.g., for atmosphere profiles.
    See the Jupyter notebook Introduction_to_Modelling_the_Atmosphere... in
    the docs/ADRpy directory for usage examples.
    """

    npanels = len(haxes)
    figobj, axes = plt.subplots(1, npanels, sharey=True)

    plt.rc('text', usetex=tex)
    plt.rc('font', family=fam)

    for i in range(npanels):
        axes[i].plot(haxes[i], vaxis)
        axes[i].set_xlim(hlimits[i])
        axes[i].grid(True)
        axes[i].set_xlabel(hlabels[i])
        for j, hli in enumerate(hlines):
            axes[i].plot(hlimits[i], [hli, hli], color=hlinecols[j])

    axes[0].set_ylabel(vlabel)

    figobj.set_figwidth(figpar[0])
    figobj.set_figheight(figpar[1])
    figobj.dpi = figpar[2]

    return figobj, axes


def recastasnpfloatarray(scalarorvec):
    """Recasts an arbitrary argument as a numpy float array. Used
    internally by some of the constraint calculations to increase
    robustness, though the use of numpy arrays as inputs is the
    recommended approach in most cases.
    """
    if isinstance(scalarorvec, Number):
        scalarorvec = [scalarorvec]
    # Convert to Numpy array if list
    scalarorvec = np.asarray(scalarorvec, dtype=float)
    return scalarorvec


def polyblend(time, time_f, signal_i, signal_f):
    """A smooth blend between two levels of a signal. Suitable for
    approximating thrust variations associated with spool-up and
    spool-down, etc.
    """
    normtime = time / time_f
    scale = signal_f - signal_i
    if time < 0:
        return signal_i
    if time >= time_f:
        return signal_f
    return signal_i + scale * (normtime ** 2) * (3 - 2 * normtime)


def panelplot_with_shared_x(haxis, vaxes, hlabel,
                            vlabels, vlines, vlinecols,
                            figpar=[6, 10, 100],
                            tex=False, fam='sans-serif',
                            fontsizes=[8, 8, 8]):
    """Multi-panel plots with a shared x-axis, e.g., for time series.
    For plotting flight data (e.g., as seen in accident reports), use
    `fdrplot`.  
    """

    npanels = len(vaxes)
    figobj, axes = plt.subplots(npanels, 1, sharex=True)
    
    plt.rc('text', usetex=tex)
    plt.rc('font', family=fam)

    # fontsizes is of the format: [axis labels, axis ticks, legend text]

    plt.rc('axes', labelsize=fontsizes[0])
    plt.rc('xtick', labelsize=fontsizes[1])
    plt.rc('ytick', labelsize=fontsizes[1])
    plt.rc('legend', fontsize=fontsizes[2])
    
    for i in range(npanels):
        axes[i].grid(True)
        if type(vaxes[i])==list:
            for j, vdata in enumerate(vaxes[i]):
                axes[i].plot(haxis, vdata, label=vlabels[i][j+1])

            axes[i].set_ylabel(vlabels[i][0])
            axes[i].legend(loc='best')
        else:
            axes[i].set_ylabel(vlabels[i])
            axes[i].plot(haxis, vaxes[i])
            axes[i].grid(True)
            axes[i].set_ylabel(vlabels[i])
        vertrange = axes[i].get_ylim()
        for j, vli in enumerate(vlines):
            axes[i].plot([vli, vli], vertrange, color=vlinecols[j])

    axes[npanels-1].set_xlabel(hlabel)

    figobj.set_figwidth(figpar[0])
    figobj.set_figheight(figpar[1]) 
    figobj.dpi = figpar[2]

    return figobj, axes


def fdrplot(timeseriescsvfile, timeline, panels, markers, figpars):
    """Generates a multi-panel time series plot, suitable, for example,
    for the analysis of flight test data.

    **Parameters:**

        timeseriescsvfile
            CSV file containing the data to be visualised. It must have a
            header containing the names of the channels, with each corresponding
            signal in the column underneath its name. There should be a time
            column (elapsed time in seconds) and the data should be ordered
            according to this column.

        timeline
            List with four entries: the header of the time column in the CSV
            file (string), the label of the time (horizontal) axis (string),
            the time at which the plot should start (float) and the time at
            which the plot should end (float).

        panels
            List of lists of variables to be plotted. Each list represents
            one panel on the plot, and is of the form `[panel y-label, channel1,
            channel2, ...]` (all strings), `channel1`, `channel2`, etc. should
            be column headings from the CSV file. They will also become
            the legend labels. There should be at least two lines (two
            panels). See example below.

        markers
            A list of two lists, specifying the location and color of
            vertical marker lines (e.g., to mark some landmark point
            in the flight, such as the beginning of rotation and 
            lift-off in the example below).

        figpars
            A list of two lists and an integer of the form `[[fig width,
            fig height, dpi], [axes label font size, tick font size, 
            legend font size], colourscheme]`. Set `colourscheme` to
            1 for a clean, minimalistic look (recommended).
            See example below.

    **Outputs:**

        f
            figure object

        axes
            subplot axes handles

        flightdata_tf
            `pandas` dataframe containing the plotted data (the original CSV
            file data trimmed as specified by the time limits). 

    **Example - visualising a take-off:** ::

        import ADRpy
        from ADRpy import mtools4acdc as adrpytools
        import os

        timeline = ['Time', 'Time (s)', 20, 60]

        panels = [
            ['Angles (deg)', 'True AoA', 'Pitch angle'],
            ['Altitude (ft)', 'IRS Alt', 'GPS Alt', 'Press alt'],
            ['Body rotation rates (deg/s)', 'Roll rate', 'Pitch rate', 'Yaw rate'],
            ['Speed (knots)', 'IAS', 'TAS', 'GPS GS']
        ]
        
        timeseriescsvfile = os.path.join(ADRpy.__path__[0], "data", "sample_takeoff_data.csv")

        markers = [[40, 43.8], ['grey','grey']]

        figpars = [[6, 10, 300], [8, 8, 8], 1]

        figobj, axes, flightdata = adrpytools.fdrplot(
            timeseriescsvfile, timeline, panels, markers, figpars)

    Output: ::

        A four-panel time series plot of 11 channels of data.
    """

    # Unpacking the timeline - contains the header of the time 
    # column, the time label for the whole plot and the edges of the 
    # required timeframe.
    tcol = timeline[0]
    tlabel = timeline[1]
    ti = timeline[2]
    tf = timeline[3]
    
    vlines = markers[0]
    vlinecols = markers[1]
    
    plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
    
    rawflightdata = pandas.read_csv(timeseriescsvfile)
    
    flightdata_tf = rawflightdata[(rawflightdata[tcol]>=ti) &
                                  (rawflightdata[tcol]<=tf)]

    timeaxis = flightdata_tf[tcol]
    
    data_per_panel = []
    panel_y_labels = []
    
    for panel in panels:
        panel_y_labels.append(panel[0])
        dataline = []
        for signalname in panel[1:]:
            dataline.append(flightdata_tf[signalname])
        data_per_panel.append(dataline) 
   
    
    f, axes = panelplot_with_shared_x(
        timeaxis, data_per_panel, tlabel, panels,
        vlines, vlinecols,
        figpar=figpars[0], fontsizes=figpars[1])

    if len(figpars)>2 and figpars[2]==1:
        for ax in axes:
            ax.patch.set_facecolor('#deedfa')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.grid(color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='x', labelcolor='black')
            ax.tick_params(axis='y', colors='white')
            ax.tick_params(axis='y', labelcolor='black')

    plt.tight_layout(pad=0, h_pad=0, w_pad=None, rect=None)
    plt.show()
    
    return f, axes, flightdata_tf


def _coordtrans(x, z, theta_rad):
    xt = []
    zt = []
    for i, _  in enumerate(x):
        xt.append(x[i] * math.cos(theta_rad)  + z[i] * math.sin(theta_rad))
        zt.append(-x[i] * math.sin(theta_rad) + z[i] * math.cos(theta_rad))
    return xt, zt


def iterable(obj):
    """Use to determine if object "obj" is iterable. Returns a boolean."""
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


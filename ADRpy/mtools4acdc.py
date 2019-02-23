#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. _mtools4acdc_module:

Miscellaneous Utilities
-----------------------

This module contains miscellaneous tools to support aircraft
engineering calculations.

"""

__author__ = "Andras Sobester"


# pylint: disable-msg=W0102, R0913
# W0102: erroneous warning from pylint, suggesting '[]' default
# R0913: number of inputs (panelplot) - this feels more readable

from numbers import Number
import numpy as np
import matplotlib.pyplot as plt



def panelplot_with_shared_y(vaxis, haxes, hlimits, vlabel,
                            hlabels, hlines, hlinecols, figpar=[10, 6, 100]):
    """Multi-panel plots with a shared y-axis, e.g., for atmosphere profiles"""

    npanels = len(haxes)
    figobj, axes = plt.subplots(1, npanels, sharey=True)

    # Use LaTeX to interpret figure labels, etc.
    # (the cell may need to be run twice)
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')

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
    """Recasts an arbitrary argument as a numpy float array"""
    if isinstance(scalarorvec, Number):
        scalarorvec = [scalarorvec]
    # Convert to Numpy array if list
    scalarorvec = np.asarray(scalarorvec, dtype=float)
    return scalarorvec

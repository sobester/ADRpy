#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. _mtools4acdc_module:

Miscellaneous Utilities
-----------------------

This module contains miscellaneous tools to support aircraft
engineering calculations.

"""

# pylint: disable-msg=W0102, R0913
# W0102: erroneous warning from pylint, suggesting '[]' default
# R0913: number of inputs (panelplot) - this feels more readable
# pylint: disable=locally-disabled, too-many-locals

from numbers import Number
import numpy as np
import matplotlib.pyplot as plt


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

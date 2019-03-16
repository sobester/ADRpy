#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""weathertools.py:
Weather tools for aircraft performance calculations.

METHODS: readmethistfile, decodemetar
"""

# pylint: disable-msg=W0702
# Non-critical if except fails too, also relying on error handling
# from the metar package here

__author__ = "Andras Sobester"

from metar import Metar

def readmethistfile(filename):
    """Reads a text file containing a METAR on each line"""

    _fstr = filename + '.methist'

    try:
        fhandle = open(_fstr, 'r')
        metarlist = fhandle.readlines()
        # Remove the equal sign (if present) and new line character
        # from the end of each line
        if metarlist[0][-2] == '=':
            metarlist = [line[:-2] for line in metarlist]
        else:
            metarlist = [line[:-1] for line in metarlist]
        return metarlist
    except FileNotFoundError:
        print("No METAR history file found.")
        return None


def decodemetar(metar_str):
    """Decodes a METAR presented as a string"""
    # Wraps the python-metar package, working around a date parsing error
    # (python-metar fails if the day is past the current day and out of
    # range for the previous month - this would make it unsuitable for
    # processing historical METARs)

    try:
        decoded_obs = Metar.Metar(metar_str)
    except:
        # No month has fewer than 28 days, so this should be safe
        metar_str = metar_str[0:4] + ' 28' + metar_str[7:]
        decoded_obs = Metar.Metar(metar_str)

    return decoded_obs

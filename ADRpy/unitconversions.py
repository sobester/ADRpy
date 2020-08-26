#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. _unitconversions_module:

Unit Conversions
----------------

This module contains tools for converting between units commonly used in
aircraft design.

"""

__author__ = "Andras Sobester"

import scipy.constants as sc

def c2f(temp_c):
    """Convert temperature value from Celsius to Fahrenheit"""
    return temp_c * 9/5 + 32

def feet2m(length_feet):
    """Converts length value from feet to meters"""
    return length_feet * 0.3048

def feet22m2(area_ft2):
    """Converts area value from feet squared to meters squared"""
    return area_ft2 / 10.7639

def m22feet2(area_m2):
    """Converts area value from meters squared to feet squared"""
    return area_m2 * 10.7639

def m2km(length_m):
    """Converts length value from meters to kilometres"""
    return length_m / 1000.0

def km2m(length_km):
    """Converts length value from kilometres to metres"""
    return length_km * 1000.0

def c2k(temp_c):
    """Convert temperature value from Celsius to Kelvin"""
    return temp_c + 273.15

def k2c(temp_k):
    """Convert temperature value from Kelvin to Celsius"""
    return temp_k - 273.15

def c2r(temp_c):
    """Convert temperature value from Celsius to Rankine"""
    return (temp_c + 273.15) * 9 / 5

def r2c(temp_r):
    """Convert temperature value from Rankine to Celsius"""
    return (temp_r - 491.67) * 5 / 9

def k2r(temp_k):
    """Convert temperature value from Kelvin to Rankine"""
    return temp_k * 9 / 5

def r2k(temp_r):
    """Convert temperature value from Rankine to Kelvin"""
    return temp_r * 5 / 9

def pa2mbar(press_pa):
    """Convert pressure value from Pascal to mbar"""
    return press_pa * 0.01

def mbar2pa(press_mbar):
    """Convert pressure value from mbar to Pascal"""
    return press_mbar / 0.01

def inhg2mbar(press_inhg):
    """Convert pressure value from inHg to mbar"""
    return press_inhg * 33.8639

def mbar2inhg(press_mbar):
    """Convert pressure value from mbar to inHg"""
    return press_mbar / 33.8639

def mbar2lbfft2(press_mbar):
    """Convert pressure value from mbar to lb/ft^2"""
    return press_mbar * 2.08854

def lbfft22mbar(press_lbfft2):
    """Convert pressure value from lb/ft^2 to mbar"""
    return press_lbfft2 / 2.08854

def mps2kts(speed_mps):
    """Convert speed value from m/s to knots"""
    return speed_mps * 1.9438445

def kts2mps(speed_kts):
    """Convert speed value knots to mps"""
    return speed_kts * 0.5144444

def m2feet(length_m):
    """Convert length value from meters to feet"""
    return length_m / 0.3048

def pa2kgm2(pressure_pa):
    """Convert pressure value from Pa to kg/m^2"""
    return pressure_pa * 0.1019716212978

def kg2n(mass_kg):
    """Converts mass in kg to weight in N"""
    return mass_kg * sc.g

def n2kg(force_n):
    """Converts force in N to mass in kg"""
    return force_n / sc.g

def kgm22pa(pressure_kgm2):
    """Convert pressure value from kg/m^2 to Pa"""
    return pressure_kgm2 * sc.g

def fpm2mps(speed_fpm):
    """Convert speed value from feet/min to m/s"""
    mpm = feet2m(speed_fpm)
    return  mpm / 60.0

def lbs2kg(mass_lbs):
    """Convert mass value from lbs to kg"""
    return mass_lbs * 0.453592

def kg2lbs(mass_kg):
    """Convert mass value from kg to lbs"""
    return mass_kg / 0.453592

def lbf2n(force_lbf):
    """Convert force from lbf to N"""
    return force_lbf / 0.224809

def n2lbf(force_n):
    """Convert force from N to lbf"""
    return force_n * 0.224809

def wn2wkg(powertoweight_wn):
    """Convert power to weight (Watt/N) to W/kg"""
    return powertoweight_wn * sc.g

def wn2kwkg(powertoweight_wn):
    """Convert power to weight (Watt/N) to kW/kg"""
    return wn2wkg(powertoweight_wn) / 1000.0

def wn2hpkg(powertoweight_wn):
    """Convert power to weight (Watt/N) to hp/kg"""
    return wn2kwkg(powertoweight_wn) * 1.34102

def kw2hp(power_kw):
    """Convert power from kW to HP"""
    return power_kw * 1.34102

def hp2kw(power_hp):
    """Convert power from HP to kW"""
    return power_hp / 1.34102

def kgm32sft3(density_kgm3):
    """Convert density from kg/m^3 to slugs/ft^3"""
    return density_kgm3 * 0.00194032

def sft32kgm3(density_slft3):
    """Convert density from slugs/ft^3 to kg/m^3"""
    return density_slft3 / 0.00194032

def tas2eas(tas, localairdensity_kgm3):
    """Convert True Air Speed to Equivalent Air Speed"""
    return tas * ((localairdensity_kgm3 / 1.225) ** 0.5)

def eas2tas(eas, localairdensity_kgm3):
    """Convert True Air Speed to Equivalent Air Speed"""
    return eas / ((localairdensity_kgm3 / 1.225) ** 0.5)

def pa2lbfft2(pressure_pa):
    """Convert pressure from Pascal to lbf(pound-force)/ft^2"""
    return pressure_pa * 0.020885434273039

def lbfft22pa(pressure_lbft2):
    """Convert pressure from lbf(pound-force)/ft^2 to Pascal"""
    return pressure_lbft2 / 0.020885434273039

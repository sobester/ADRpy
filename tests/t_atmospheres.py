#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""t_atmospheres.py:
Unit tests for the atmospheres module
"""

import unittest
import numpy as np
from ADRpy import atmospheres as at

class TestUM(unittest.TestCase):
    """Unit tests for the atmospheres module against the 1976 US std. atm."""

    # Source for validation data: NASA-TM-X-74335

    def setUp(self):
        pass

    def test_geoptest(self):
        """Tests the geometric -> geopotential altitude conversion"""
        print("Geopotential conversion.")
        # Z = 10,000m geometric corresponds to H = 9,984.3m geopotential altitude
        self.assertEqual(round(10 * at.geom2geop45m(10000)), round(10 * 9984.3))

    def test_geomtest(self):
        """Tests the geopotential -> geometric altitude conversion"""
        print("Geometric conversion.")
        # Z = 10,000m geometric corresponds to H = 9,984.3m geopotential altitude
        self.assertEqual(round(10 * at.geop2geom45m(9984.3)), round(10 * 10000))

    def test_isa_sl(self):
        """Tests the atmosphere class instantiated for an ISA at SL"""
        print("ISA (SL).")
        alt_m = 0
        isa = at.Atmosphere()
        self.assertEqual(isa.airtemp_c(alt_m), 15)
        self.assertEqual(round(100 * isa.airpress_mbar(alt_m)), 101325)
        self.assertEqual(round(1000 * isa.airdens_kgpm3(alt_m)), 1225)
        self.assertEqual(round(100 * isa.vsound_mps(alt_m)), round(100 * 340.29))

    def test_isa_10k_geop(self):
        """Tests the atmosphere class instantiated for an ISA at 10km geopotential"""
        print("ISA (10,000m geopotential).")
        # ISA at 10,000m geopotential, 10,015.8m geometric altitude.
        alt_m = 10000
        isa = at.Atmosphere()
        self.assertEqual(isa.airtemp_c(alt_m), -50)
        self.assertEqual(round(100 * isa.airpress_mbar(alt_m)), 26436)
        self.assertEqual(round(100000 * isa.airdens_kgpm3(alt_m)), 41271)
        self.assertEqual(round(1000 * isa.vsound_mps(alt_m)), round(1000 * 299.463))

    def test_isa_10k_geom(self):
        """Tests the atmosphere class instantiated for an ISA at 10km geometric alt"""
        print("ISA (10,000m geometric).")
        # ISA at Z = 10,000m geometric, H = 9,984.3m geopotential altitude.
        alt_m = 9984.3
        isa = at.Atmosphere()
        self.assertEqual(round(10000 * isa.airtemp_c(alt_m)), round(10000 * -49.8979))
        self.assertEqual(round(100 * isa.airpress_mbar(alt_m)), round(100 * 264.999))
        self.assertEqual(round(100000 * isa.airdens_kgpm3(alt_m)), round(100000 * 0.413511))
        self.assertEqual(round(1000 * isa.vsound_mps(alt_m)), round(1000 * 299.532))

    def test_isa_minus5k_geop(self):
        """Tests the atmosphere class instantiated for an ISA at -5km geopotential alt"""
        print("ISA (-5,000m geopotential).")
        # ISA at Z = -4996.1m geometric, H = -5000m geopotential altitude.
        alt_m = -5000
        isa = at.Atmosphere()
        self.assertEqual(round(1000 * isa.airtemp_c(alt_m)), round(1000 * 47.5002))
        self.assertEqual(round(10 * isa.airpress_mbar(alt_m)), round(10 * 1776.88))
        self.assertEqual(round(10000 * isa.airdens_kgpm3(alt_m)), round(10000 * 1.93048))
        self.assertEqual(round(1000 * isa.vsound_mps(alt_m)), round(1000 * 358.972))

    def test_runwayclass(self):
        """Tests the methods of the runways class"""
        print("Runway class.")
        # LHR 09L/27R, Elev 79m, 'le' = 09L
        rwy = at.Runway('EGLL', rwyno=0)
        self.assertEqual(rwy.le_ident, '09L')
        self.assertEqual(round(rwy.le_heading_degt), 90)
        self.assertEqual(round(rwy.le_elevation_ft), 79)

        # Wind examples. Convention: headwind and right are +

        # Wind example 1: down the runway headwind (+) of 10 (any unit)
        wind_dirs_deg = 89.6
        wind_speeds_kt = 10
        [runwaycomp, crosscomp] = rwy.windcomponents(wind_dirs_deg, wind_speeds_kt)
        self.assertEqual(round(runwaycomp * 100), 1000)
        self.assertEqual(round(crosscomp * 100), 0)

        # Wind example 2: perpendicular xwind from right (+) of 10 (any unit)
        wind_dirs_deg = 89.6 + 90
        wind_speeds_kt = 10
        [runwaycomp, crosscomp] = rwy.windcomponents(wind_dirs_deg, wind_speeds_kt)
        self.assertEqual(round(runwaycomp * 100), 0)
        self.assertEqual(round(crosscomp * 100), 1000)

        # Wind example 3: the two previous cases plus another case, given as a list
        # Additional case is wind of 10 units from 300 deg
        wind_dirs_deg = [89.6, 89.6 + 90, 300]
        wind_speeds_kt = [10, 10, 10]
        [runwaycomp, crosscomp] = rwy.windcomponents(wind_dirs_deg, wind_speeds_kt)
        np.testing.assert_array_equal(np.round(runwaycomp * 100), [1000, 0, -863])
        np.testing.assert_array_equal(np.round(crosscomp * 100), [0, 1000, -506])

    def test_m310(self):
        """Tests access to the MIL HDBK 310 atmospheres"""
        print("MIL HDBK 310 - high temp at 10km.")
        obs1p, _ = at.mil_hdbk_310('high', 'temp', 10)
        m310_high10_1p = at.Atmosphere(profile=obs1p)
        sltemp_c = m310_high10_1p.airtemp_c(100)
        self.assertEqual(round(1000 * sltemp_c), round(1000 * 25.501))

if __name__ == '__main__':
    unittest.main()

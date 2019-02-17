#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""t_constraints.py:
Unit tests for the constraints module
"""

import unittest
from ADRpy import constraintanalysis as ca
from ADRpy import atmospheres as at

class TestUM(unittest.TestCase):
    """Unit tests for the constraints module."""

    def setUp(self):
        pass

    def test_take_off(self):
        """Tests the take-off constraint calculation"""

        print("Take-off constraint test.")

        designbrief = {'rwyelevation_m':1000, 'groundrun_m':1200}
        designdefinition = {'aspectratio':7.3, 'bpr':3.9, 'tr':1.05}
        designperformance = {'CDTO':0.04, 'CLTO':0.9, 'CLmaxTO':1.6, 'mu_R':0.02}

        wingloadinglist_pa = [2000, 3000, 4000, 5000]

        atm = at.Atmosphere()
        concept = ca.AircraftConcept(designbrief, designdefinition, designperformance, atm)

        tw_sl, liftoffspeed_mps, _ = concept.twrequired_to(wingloadinglist_pa)

        self.assertEqual(round(10000 * tw_sl[0]), round(10000 * 0.19397876))
        self.assertEqual(round(10000 * liftoffspeed_mps[0]), round(10000 * 52.16511207))

        self.assertEqual(round(10000 * tw_sl[3]), round(10000 * 0.41110154))
        self.assertEqual(round(10000 * liftoffspeed_mps[3]), round(10000 * 82.48028428))


if __name__ == '__main__':
    unittest.main()

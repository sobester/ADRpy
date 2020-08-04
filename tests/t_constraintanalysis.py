import random
import math
import unittest

import numpy as np

from ADRpy import constraintanalysis as ca


class TestConstraintAnalysisModule(unittest.TestCase):

    def setUp(self):
        # Create an aircraft test library, to be populated
        self.ac_lib = []

        # AIRCRAFT 0: Business Jet (Learjet 45XR)
        l45xr_brief = {'rwyelevation_m': 1000, 'groundrun_m': 1200,  # Take-off Constraint
                       'climbalt_m': 1000, 'climbspeed_kias': 250, 'climbrate_fpm': 1000,  # Climb Constraint
                       'cruisealt_m': 15000, 'cruisespeed_ktas': 445,  # Cruise Constraint
                       'servceil_m': 16000, 'secclimbspd_kias': 250,  # Service Ceiling Constraint
                       'stloadfactor': 2, 'turnalt_m': 5000, 'turnspeed_ktas': 300}  # Turn Constraint
        l45xr_def = {'aspectratio': 7.3, 'sweep_le_deg': 10, 'sweep_25_deg': 8, 'bpr': 3.9, 'tr': 1.05,
                     'weightfractions': {'turn': 1.0, 'climb': 1.0, 'cruise': 0.85, 'servceil': 0.85}}
        l45xr_perf = {'CDTO': 0.04, 'CLTO': 0.9, 'CLmaxTO': 1.6, 'CLmaxclean': 1.42, 'mu_R': 0.02,
                      'CDminclean': 0.02}
        self.ac_lib.append([l45xr_brief, l45xr_def, l45xr_perf])

        # AIRCRAFT 1: Single Engine Piston Propeller Aircraft (Cirrus SR22)
        cr22_brief = {'rwyelevation_m': 0, 'groundrun_m': 313,  # Take-off Constraint
                      'stloadfactor': 1.5, 'turnalt_m': 1000, 'turnspeed_ktas': 100,  # Turn Constraint
                      'climbalt_m': 0, 'climbspeed_kias': 101, 'climbrate_fpm': 1398,  # Climb Constraint
                      'cruisealt_m': 3048, 'cruisespeed_ktas': 182, 'cruisethrustfact': 1.0,  # Cruise Constraint
                      'servceil_m': 6580, 'secclimbspd_kias': 92,  # Service Ceiling Constraint
                      'vstallclean_kcas': 69}  # Minimum Stall speed
        cr22_def = {'aspectratio': 10.12, 'sweep_le_deg': 2, 'sweep_25_deg': 0, 'bpr': -1, 'wingarea_m2': 13.46,
                    'weightfractions': {'turn': 1.0, 'climb': 1.0, 'cruise': 0.853, 'servceil': 1.0}}
        cr22_perf = {'CDTO': 0.0414, 'CLTO': 0.59, 'CLmaxTO': 1.69, 'CLmaxclean': 1.45, 'mu_R': 0.02,
                     'CDminclean': 0.0254, 'etaprop': {'take-off': 0.65, 'climb': 0.8, 'cruise': 0.85,
                                                       'turn': 0.85, 'servceil': 0.8}}
        self.ac_lib.append([cr22_brief, cr22_def, cr22_perf])

        # AIRCRAFT 2: Fighter (F/A-18C Hornet)
        fa18c_brief = {'groundrun_m': 427, 'servceil_m': 15240,
                       'cruisealt_m': 12000, 'cruisespeed_ktas': 570}
        fa18c_def = {'aspectratio': 4, 'sweep_le_deg': 27, 'sweep_25_deg': 20, 'wingarea_m2': 38,
                     'weight_n': 23541 * 9.81}
        fa18c_perf = {}
        self.ac_lib.append([fa18c_brief, fa18c_def, fa18c_perf])

        # AIRCRAFT 3: Business Jet (Aircraft Concept)
        conceptjet_brief = {'rwyelevation_m': [-100, 1000], 'groundrun_m': [1200, 1400],
                            'climbalt_m': [-100, 1000], 'climbspeed_kias': [240, 250], 'climbrate_fpm': [1000, 1200],
                            'cruisealt_m': [14000, 15000], 'cruisespeed_ktas': [440, 480],
                            'servceil_m': [16000, 17000], 'secclimbspd_kias': [240, 260],
                            'stloadfactor': [2, 2.5], 'turnalt_m': [4000, 5000], 'turnspeed_ktas': [280, 330]}
        conceptjet_def = {'aspectratio': [6, 8], 'sweep_le_deg': [9, 12], 'sweep_25_deg': [6, 8], 'bpr': [4.9, 5.2],
                          'tr': [1.05, 1.5], 'weight_n': 95000,
                          'weightfractions': {'turn': 1.0, 'climb': 1.0, 'cruise': 0.85, 'servceil': 0.85}}
        conceptjet_perf = {'CDTO': [0.04, 0.045], 'CLTO': [0.85, 0.9], 'CLmaxTO': [1.6, 1.8],
                           'CLmaxclean': [1.42, 1.46], 'mu_R': 0.02, 'CDminclean': [0.02, 0.025]}
        self.ac_lib.append([conceptjet_brief, conceptjet_def, conceptjet_perf])

        # AIRCRAFT 4: SEPPA (Aircraft Concept)
        conceptsep_brief = {'rwyelevation_m': 0, 'groundrun_m': 313,
                            'stloadfactor': [1.5, 1.65], 'turnalt_m': [1000, 1075], 'turnspeed_ktas': [100, 110],
                            'climbalt_m': 0, 'climbspeed_kias': 101, 'climbrate_fpm': 1398,
                            'cruisealt_m': [2900, 3200], 'cruisespeed_ktas': [170, 175], 'cruisethrustfact': 1.0,
                            'servceil_m': [6500, 6650], 'secclimbspd_kias': 92,
                            'vstallclean_kcas': 69}
        conceptsep_def = {'aspectratio': [10, 11], 'sweep_le_deg': 2, 'sweep_25_deg': 0, 'bpr': -1,
                          'wingarea_m2': 13.46, 'weight_n': 15000,
                          'weightfractions': {'turn': 1.0, 'climb': 1.0, 'cruise': 0.853, 'servceil': 1.0}}
        conceptsep_perf = {'CDTO': 0.0414, 'CLTO': 0.59, 'CLmaxTO': 1.69, 'CLmaxclean': 1.45, 'mu_R': 0.02,
                           'CDminclean': [0.0254, 0.026], 'etaprop': {'take-off': 0.65, 'climb': 0.8, 'cruise': 0.85,
                                                                      'turn': 0.85, 'servceil': 0.8}}
        self.ac_lib.append([conceptsep_brief, conceptsep_def, conceptsep_perf])

        return

    def test_0createobject(self):
        """Tests parameters were copied in correctly"""

        print("Create concept test.")

        # Choose and build a random aircraft from the aircraft test library

        self.ac_rand_i = random.randint(0, len(self.ac_lib) - 1)
        self.ac_random = ca.AircraftConcept(self.ac_lib[self.ac_rand_i][0],
                                            self.ac_lib[self.ac_rand_i][1], self.ac_lib[self.ac_rand_i][2])

        # For each design dictionary of the random aircraft picked
        for dictindex in range(len(self.ac_lib[self.ac_rand_i])):
            # Go through each item of a single design dictionary from the test library and confirm it copied correctly
            for i, (k, test_value) in enumerate(self.ac_lib[self.ac_rand_i][dictindex].items()):
                if type(self.ac_random.designspace[dictindex][k]) == list:
                    dictvalue = sum(self.ac_random.designspace[dictindex][k]) \
                                / len(self.ac_random.designspace[dictindex][k])
                else:
                    dictvalue = self.ac_random.designspace[dictindex][k]

                self.assertEqual(dictvalue, test_value)

        # Build the last aircraft in the aircraft test library

        self.ac_last = ca.AircraftConcept(self.ac_lib[-1][0], self.ac_lib[-1][1], self.ac_lib[-1][2])

        # For each design dictionary of the random aircraft picked
        for dictindex in range(len(self.ac_lib[-1])):
            # Go through each item of a single design dictionary from the test library and confirm it copied correctly
            for i, (k, test_value) in enumerate(self.ac_lib[-1][dictindex].items()):
                if type(self.ac_last.designspace[dictindex][k]) == list:
                    dictvalue = sum(self.ac_last.designspace[dictindex][k]) \
                                / len(self.ac_last.designspace[dictindex][k])
                else:
                    dictvalue = self.ac_last.designspace[dictindex][k]

                self.assertEqual(dictvalue, test_value)

        return

    def test_estimateliftslope(self):
        """Tests the generation of predicted lift-slope with mach number"""

        print("Lift-curve slope estimation test.")

        # Use Aircraft 2: F/A-18C
        acindex = 2
        concept = ca.AircraftConcept(self.ac_lib[acindex][0], self.ac_lib[acindex][1], self.ac_lib[acindex][2])

        macharray = np.arange(0.1, 4, 0.1)
        liftslopelist = []
        for mach_inf in macharray:
            liftslopelist.append(concept.estimate_liftslope(mach_inf=mach_inf))

        # Assert that the lift curve slope is never below or equal to zero
        self.assertGreater(min(liftslopelist), 0)

        return

    def test_findchordsweep(self):
        """Tests the calculation of sweep angle in radians, at some point of the wing chord"""

        print("Find arbitrary chord-sweep test.")

        # Use Aircraft 1: Piston propeller aircraft
        acindex = 1
        concept = ca.AircraftConcept(self.ac_lib[acindex][0], self.ac_lib[acindex][1], self.ac_lib[acindex][2])

        sweep_le_rad = concept.sweep_le_rad
        sweep_25_rad = concept.sweep_25_rad

        self.assertEqual(concept.findchordsweep_rad(0), sweep_le_rad)
        self.assertEqual(concept.findchordsweep_rad(0.25), sweep_25_rad)

        return

    def test_induceddragfact(self):
        """Tests the induced drag factor K calculation"""

        print("Induced drag factor test.")

        # Use Aircraft 2: F/A-18C
        acindex = 2
        concept = ca.AircraftConcept(self.ac_lib[acindex][0], self.ac_lib[acindex][1], self.ac_lib[acindex][2])

        macharray = np.arange(0.1, 2, 0.1)
        kpredlist = []

        for mach_inf in macharray:
            kpredlist.append(concept.induceddragfact(mach_inf=mach_inf, cl_req=1.2))

        # Assert that the induced drag is never below or equal to zero
        self.assertGreater(min(kpredlist), 0)

        return

    def test_twreqconstraint(self):
        """Tests the thrust-to-weight constraint calculations"""

        # Use Aircraft 0: Business Jet
        acindex = 0
        concept = ca.AircraftConcept(self.ac_lib[acindex][0], self.ac_lib[acindex][1], self.ac_lib[acindex][2])
        wingloadinglist_pa = [2000, 3000, 4000, 5000, 6000, 7000]

        # Investigate the climb constraint
        print("T/W Climb constraint test.")
        tw_climb = concept.twrequired_clm(wingloading_pa=wingloadinglist_pa)

        testarray1 = np.array([0.15268568, 0.1239213, 0.11219318, 0.10727957, 0.10577321, 0.10621385])
        self.assertIs(tw_climb.all(), testarray1.all())

        # Investigate the cruise constraint
        print("T/W Cruise constraint test.")
        tw_cruise = concept.twrequired_crs(wingloading_pa=wingloadinglist_pa)

        testarray1 = np.array([0.37261153, 0.31997436, 0.31512577, 0.32939262, 0.35321719, 0.38250331])
        self.assertEqual(tw_cruise.all(), testarray1.all())

        # Investigate the service ceiling constraint
        print("T/W Service Ceiling constraint test.")
        tw_serviceceil = concept.twrequired_sec(wingloading_pa=wingloadinglist_pa)

        testarray1 = np.array([0.46419224, 0.34031968, 0.28625287, 0.26010837, 0.24792501, 0.24371946])
        self.assertEqual(tw_serviceceil.all(), testarray1.all())

        # Investigate the take-off constraint
        print("T/W Take-off constraint test.")
        tw_sl, liftoffspeed_mpstas, _ = concept.twrequired_to(wingloadinglist_pa)

        self.assertEqual(round(10000 * tw_sl[0]), round(10000 * 0.19397876))
        self.assertEqual(round(10000 * liftoffspeed_mpstas[0]), round(10000 * 52.16511207))
        self.assertEqual(round(10000 * tw_sl[3]), round(10000 * 0.41110154))
        self.assertEqual(round(10000 * liftoffspeed_mpstas[3]), round(10000 * 82.48028428))

        # Investigate the cruise constraint
        print("T/W Turn constraint test.")
        tw_turn = concept.twrequired_trn(wingloading_pa=wingloadinglist_pa)

        testarray1 = np.array([0.21826547, 0.21048143, 0.22608074, 0.25103339, 0.28066271, 0.31296442])
        testarray2 = np.array([0.45627288, 0.68440931, 0.91254575, 1.14068219, 1.36881863, 1.59695506])
        testarray3 = np.array([0.21826547, 0.21048143, 0.22608074, 0.25103339, 0.28066271, np.nan])
        self.assertEqual(tw_turn[0].all(), testarray1.all())
        self.assertEqual(tw_turn[1].all(), testarray2.all())
        self.assertEqual(tw_turn[2].all(), testarray3.all())

        return

    def test_twreqsensitivity(self):
        """Tests the statistical analysis method for the one-at-a-time inquiry of T/W sensitivity to input parameters"""

        print("T/W Sensitivity (One-at-a-time) test.")

        # Use Aircraft 3: Custom Business Jet
        acindex = 3
        concept = ca.AircraftConcept(self.ac_lib[acindex][0], self.ac_lib[acindex][1], self.ac_lib[acindex][2])
        wingloadinglist_pa = np.arange(2000, 8000, 50)

        customlabelling = {'aspectratio': 'AR',
                           'sweep_le_deg': '$\\Lambda_{LE}$',
                           'sweep_mt_deg': '$\\Lambda_{MT}$'}

        concept.propulsionsensitivity_monothetic(wingloading_pa=wingloadinglist_pa, y_var='tw', x_var='ws',
                                                 customlabels=customlabelling)

        # Use Aircraft 4: Custom SEPPA
        acindex = 4
        concept = ca.AircraftConcept(self.ac_lib[acindex][0], self.ac_lib[acindex][1], self.ac_lib[acindex][2])
        wingloadinglist_pa = np.arange(700, 2500, 5)

        customlabelling = {'aspectratio': 'AR',
                           'sweep_le_deg': '$\\Lambda_{LE}$',
                           'sweep_mt_deg': '$\\Lambda_{MT}$'}
        concept.propulsionsensitivity_monothetic(wingloading_pa=wingloadinglist_pa, y_var='p', x_var='s',
                                                 customlabels=customlabelling)

        return

    def test_vstall(self):
        """Tests the stall speed method"""

        print("Stall speed method (vstall_kias) test.")

        designperformance = {'CLmaxTO': 1.6}

        concept = ca.AircraftConcept({}, {}, designperformance, {})

        wingloading_pa = 3500

        self.assertEqual(round(10000 * concept.vstall_kias(wingloading_pa, 'take-off')),
                         round(10000 * 116.166934173))

        return

    def test_wig(self):
        """Tests the wing in ground effect factor calculation"""

        print("WIG factor test.")

        designdef = {'aspectratio': 8}
        wingarea_m2 = 10
        wingspan_m = math.sqrt(designdef['aspectratio'] * wingarea_m2)

        for wingheight_m in [0.6, 0.8, 1.0]:
            designdef['wingheightratio'] = wingheight_m / wingspan_m
            aircraft = ca.AircraftConcept({}, designdef, {}, {})

        self.assertEqual(round(10000 * aircraft.wigfactor()),
                         round(10000 * 0.7619047))

        return


if __name__ == '__main__':
    unittest.main()

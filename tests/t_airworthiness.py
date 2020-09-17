import unittest

from ADRpy import airworthiness as aw

from ADRpy import unitconversions as co


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
        l45xr_def = {'aspectratio': 7.3, 'sweep_le_deg': 10, 'sweep_mt_deg': 8, 'bpr': 3.9, 'tr': 1.05,
                     'weight_n': 95000,
                     'weightfractions': {'turn': 1.0, 'climb': 1.0, 'cruise': 0.85, 'servceil': 0.85}}
        l45xr_perf = {'CDTO': 0.04, 'CLTO': 0.9, 'CLmaxTO': 1.6, 'CLmaxclean': 1.42, 'mu_R': 0.02,
                      'CDminclean': 0.02}
        self.ac_lib.append([l45xr_brief, l45xr_def, l45xr_perf])

        # AIRCRAFT 1: Single Engine Piston Propeller Aircraft (Cirrus SR22)
        sr22_brief = {'rwyelevation_m': 0, 'groundrun_m': 313,  # Take-off Constraint
                      'stloadfactor': 1.5, 'turnalt_m': 1000, 'turnspeed_ktas': 100,  # Turn Constraint
                      'climbalt_m': 0, 'climbspeed_kias': 101, 'climbrate_fpm': 1398,  # Climb Constraint
                      'cruisealt_m': 3048, 'cruisespeed_ktas': 182, 'cruisethrustfact': 1.0,  # Cruise Constraint
                      'servceil_m': 6580, 'secclimbspd_kias': 92,  # Service Ceiling Constraint
                      'vstallclean_kcas': 69}  # Minimum Stall speed
        sr22_def = {'aspectratio': 10.12, 'sweep_le_deg': 2, 'sweep_mt_deg': 0, 'bpr': -1, 'wingarea_m2': 13.46,
                    'weight_n': co.lbf2n(3400),
                    'weightfractions': {'turn': 1.0, 'climb': 1.0, 'cruise': 0.853, 'servceil': 1.0}}
        sr22_perf = {'CDTO': 0.0414, 'CLTO': 0.59, 'CLmaxTO': 1.69, 'CLmaxclean': 1.45, 'CLminclean': -0.8,
                     'mu_R': 0.02, 'CDminclean': 0.0254, 'etaprop': {'take-off': 0.65, 'climb': 0.8, 'cruise': 0.85,
                                                                     'turn': 0.85, 'servceil': 0.8}}
        self.ac_lib.append([sr22_brief, sr22_def, sr22_perf])

        # AIRCRAFT 2: Fighter (F/A-18C Hornet)
        fa18c_brief = {'groundrun_m': 427, 'servceil_m': 15240,
                       'cruisealt_m': 12000, 'cruisespeed_ktas': 570}
        fa18c_def = {'aspectratio': 4, 'sweep_le_deg': 27, 'sweep_25_deg': 20, 'wingarea_m2': 38,
                     'weight_n': 23541 * 9.81}
        fa18c_perf = {}
        self.ac_lib.append([fa18c_brief, fa18c_def, fa18c_perf])

        # AIRCRAFT 3: Business Jet (Aircraft Concept)
        conceptjet_brief = {'rwyelevation_m': [-100, 1000], 'groundrun_m': [1000, 1400],
                            'climbalt_m': [-100, 1000], 'climbspeed_kias': [240, 250], 'climbrate_fpm': [1000, 1200],
                            'cruisealt_m': [13000, 14000], 'cruisespeed_ktas': [440, 480],
                            'servceil_m': [14000, 14500], 'secclimbspd_kias': [240, 260],
                            'stloadfactor': [2, 2.5], 'turnalt_m': [4000, 5000], 'turnspeed_ktas': [280, 330]}
        conceptjet_def = {'aspectratio': [6, 8], 'sweep_le_deg': [9, 12], 'sweep_25_deg': [6, 8], 'bpr': [3.8, 4.0],
                          'tr': [1.05, 1.5], 'weight_n': 95000,
                          'weightfractions': {'turn': 1.0, 'climb': 1.0, 'cruise': 0.85, 'servceil': 0.85}}
        conceptjet_perf = {'CDTO': [0.04, 0.045], 'CLTO': [0.85, 0.9], 'CLmaxTO': [1.6, 1.8],
                           'CLmaxclean': [1.42, 1.46], 'mu_R': 0.02, 'CDminclean': [0.02, 0.025]}
        self.ac_lib.append([conceptjet_brief, conceptjet_def, conceptjet_perf])

        # AIRCRAFT 4: SEPPA (Aircraft Concept)
        conceptsep_brief = {'rwyelevation_m': [0, 100], 'groundrun_m': [310, 330],
                            'stloadfactor': [1.5, 1.65], 'turnalt_m': [1000, 1075], 'turnspeed_ktas': [100, 110],
                            'climbalt_m': 0, 'climbspeed_kias': 101, 'climbrate_fpm': 1398,
                            'cruisealt_m': [2900, 3200], 'cruisespeed_ktas': [170, 175], 'cruisethrustfact': 1.0,
                            'servceil_m': [6500, 7650], 'secclimbspd_kias': 92,
                            'vstallclean_kcas': 69}
        conceptsep_def = {'aspectratio': [10, 11], 'sweep_le_deg': 2, 'sweep_25_deg': 0, 'bpr': -1,
                          'wingarea_m2': 13.46, 'weight_n': 15000,
                          'weightfractions': {'turn': 1.0, 'climb': 1.0, 'cruise': 0.853, 'servceil': 1.0}}
        conceptsep_perf = {'CDTO': 0.0414, 'CLTO': 0.59, 'CLmaxTO': 1.69, 'CLmaxclean': 1.45, 'mu_R': 0.02,
                           'CDminclean': [0.0254, 0.026], 'etaprop': {'take-off': 0.65, 'climb': 0.8, 'cruise': 0.85,
                                                                      'turn': 0.85, 'servceil': 0.8}}
        self.ac_lib.append([conceptsep_brief, conceptsep_def, conceptsep_perf])

        # AIRCRAFT 5: Small Unmanned Fixed-Wing (Keane et al.)
        keane_uav_brief = {'rwyelevation_m': 0, 'groundrun_m': 60,  # <- Take-off requirements
                           'stloadfactor': 1.41, 'turnalt_m': 0, 'turnspeed_ktas': 40,  # <- Turn requirements
                           'climbalt_m': 0, 'climbspeed_kias': 46.4, 'climbrate_fpm': 591,  # <- Climb requirements
                           'cruisealt_m': 122, 'cruisespeed_ktas': 58.3, 'cruisethrustfact': 1.0,
                           # <- Cruise requirements
                           'servceil_m': 152, 'secclimbspd_kias': 40,  # <- Service ceiling requirements
                           'vstallclean_kcas': 26.4}  # <- Required clean stall speed

        keane_uav_def = {'aspectratio': 9.0, 'sweep_le_deg': 2, 'sweep_mt_deg': 0, 'bpr': -1,
                         'weightfractions': {'turn': 1.0, 'climb': 1.0, 'cruise': 1.0, 'servceil': 1.0},
                         'weight_n': 15 * 9.81}

        keane_uav_perf = {'CDTO': 0.0898, 'CLTO': 0.97, 'CLmaxTO': 1.7, 'CLmaxclean': 1.0, 'mu_R': 0.17,
                          'CDminclean': 0.0418,
                          'etaprop': {'take-off': 0.6, 'climb': 0.6, 'cruise': 0.6, 'turn': 0.6, 'servceil': 0.6}}
        self.ac_lib.append([keane_uav_brief, keane_uav_def, keane_uav_perf])

        # AIRCRAFT 6: Snorri Gudmundsson's example
        gudmundsson_brief = {'cruisealt_m': 0, 'cruisespeed_ktas': 107, 'maxlevelspeed_ktas': 140}

        gudmundsson_def = {'aspectratio': (38 ** 2) / 130, 'wingarea_m2': co.feet22m2(130),
                           'weight_n': co.lbf2n(1320), 'weightfractions': {'cruise': 1}}

        gudmundsson_perf = {'CLmaxclean': 1.45, 'CLminclean': -1, 'CLslope': 6.28}

        self.ac_lib.append([gudmundsson_brief, gudmundsson_def, gudmundsson_perf])

        return

    def test_flightenvelope(self):
        """Tests the generation of the flight envelope"""

        print("CS 23.333 Flight Envelope Plot Test.")

        vndefinitions={'divespeed_keas': 150, 'certcat': 'norm'}

        # Use Aircraft 6: Snorri Gudmundsson's example
        acindex = 6
        concept = aw.CertificationSpecifications(self.ac_lib[acindex][0], self.ac_lib[acindex][1],
                                                 self.ac_lib[acindex][2], None, None, csbrief=vndefinitions)

        wingloading_pa = concept.acobj.weight_n / concept.acobj.wingarea_m2

        concept.flightenvelope(wingloading_pa=wingloading_pa, textsize=10)

        return

    def test_paragraph335(self):
        """Tests the generation of design airspeed limits"""

        print("CS 23.335 Design Airspeeds Test.")

        # Use Aircraft 1: SEPPA SR-22
        acindex = 1
        concept = aw.CertificationSpecifications(self.ac_lib[acindex][0], self.ac_lib[acindex][1],
                                                 self.ac_lib[acindex][2])

        wingloadinglist_pa = [2000, 3000]
        _ = concept._paragraph335(wingloading_pa=wingloadinglist_pa)

        return


if __name__ == '__main__':
    unittest.main()

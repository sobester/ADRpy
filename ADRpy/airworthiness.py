#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. _airworthiness_module:

Airworthiness module
--------------------

This module contains tools for the analysis of an aircraft design from the
point of view of meeting initial airworthiness requirements.

"""

__author__ = "Yaseen Reza"

import math
import warnings

import numpy as np
from matplotlib import text as mpl_text
from matplotlib import pyplot as plt
from scipy import constants

from ADRpy import constraintanalysis as ca
from ADRpy import unitconversions as co
from ADRpy import mtools4acdc as actools


class CertificationSpecifications:
    """Aircraft concept class designed for capturing those elements of the definition
    of a design that are required for the preparation of some of the analyses needed
    for determining the initial airworthiness of an aircraft concept. Currently ADRpy
    facilitates one such type of analysis - the construction of the V-n diagram of an
    aircraft concept, as prescribed by Part 23 (as detailed, e.g., by EASA's CS-23).

    **Parameters:**

    brief
        Dictionary. See :code:`AircraftConcept` in :code:`constraintanalysis.py`.

    design
        Dictionary. See :code:`AircraftConcept` in :code:`constraintanalysis.py`.

    performance
        Dictionary. See :code:`AircraftConcept` in :code:`constraintanalysis.py`.

    designatm
        `Atmosphere <https://adrpy.readthedocs.io/en/latest/#atmospheres.Atmosphere>`_
        class object. See :code:`AircraftConcept` in :code:`constraintanalysis.py`.
        
    propulsion
        Tuple or String. See :code:`AircraftConcept` in :code:`constraintanalysis.py`.

    csbrief
        Dictionary. Definition of key parameters relating to establishing the initial
        airworthiness of an aircraft. It contains the following key names:

        certcat
            String. Used to specify the intended certification category of the aircraft,
            as per Part 23. Acceptable values are: :code:`'norm'` (normal), 
            :code:`'util'` (utility), :code:`'comm'` (commuter), or :code:`'aero'` (aerobatic).
            Optional, defaults to :code:`'norm'`.

        altitude_m
            Float. The altitude (in metres) at which the calculations are to be performed.
            Optional, defaults to 0.

        cruisespeed_keas
            Float. The design cruise speed (in knots, equivalent airspeed). Commonly denoted V_C.

        divespeed_keas
            Float. The design dive speed (in knots, equivalent airspeed). Commonly denoted V_D.

        maxlevelspeed_keas
            Float. The design maximum level flight speed at sea level (in knots, equivalent
            airspeed). Commonly denoted V_H.

        weightfraction
            Float. The fraction (nondimensional) of the maximum take-off weight at which
            the calculations are to be performed. Optional, defaults to 1.

    """

    def __init__(self, brief=None, design=None, performance=None, designatm=None, propulsion=None, csbrief=None):

        # Assign a default, if needed, to the csbrief dictionary
        if csbrief is None:
            csbrief = {}

        # Build an aircraft object based on the design dictionaries and atmosphere object
        self.acobj = ca.AircraftConcept(brief=brief, design=design, performance=performance, designatm=designatm)

        # Specify default flags or parameters for the Vn definitions dictionary, if parameter is left unspecified

        default_csbrief = {
            # Certification category
            'certcat': 'norm',  # Default category is normal
            # Vn altitude query
            'altitude_m': 0,  # Assign sea level (h = 0 metres)
            # Design Airspeeds
            'cruisespeed_keas': False,  # Flag as not specified
            'divespeed_keas': False,  # Flag as not specified
            'maxlevelspeed_keas': False,  # Flag as not specified
            # Vn loading query, fraction of MTOW
            'weightfraction': 1  # Assume V-n diagram applies to MTOW loading
        }

        # Use the templates (default dictionaries) to populate missing values in the provided design dictionaries

        # Iterate through the defaults dictionary
        for _, (defaults_k, defaults_v) in enumerate(default_csbrief.items()):
            # If a parameter of csbrief is left unspecified by the user, copy in the default value
            if defaults_k not in csbrief:
                csbrief.update({defaults_k: defaults_v})

        # FURTHER COMPREHENSION: If design cruise speed was not specified in the csbrief dictionary argument
        if csbrief['cruisespeed_keas'] is False:
            # Check to see if a cruise speed can instead be swiped from the design brief
            if self.acobj.cruisespeed_ktas is False:
                self.cruisespeed_keas = False
            else:
                rho_kgpm3 = self.acobj.designatm.airdens_kgpm3(csbrief['altitude_m'])
                self.cruisespeed_keas = co.tas2eas(self.acobj.cruisespeed_ktas, rho_kgpm3)
        else:
            self.cruisespeed_keas = csbrief['cruisespeed_keas']

        # Populate object attributes
        self.category = csbrief['certcat']
        self.divespeed_keas = csbrief['divespeed_keas']
        self.maxlevelspeed_keas = csbrief['maxlevelspeed_keas']
        self.altitude_m = csbrief['altitude_m']
        self.weightfraction = csbrief['weightfraction']

        return

    def vs_keas(self, loadfactor):
        """Equivalent air speed in knots for stall at design weight, for some loadfactor"""

        if self.acobj.weight_n is False:
            defmsg = 'Maximum take-off weight was not specified in the design definitions dictionary'
            raise ValueError(defmsg)

        if self.acobj.wingarea_m2 is False:
            defmsg = 'Reference wing area was not specified in the design definitions dictionary'
            raise ValueError(defmsg)

        weight_n = self.acobj.weight_n
        wingarea_m2 = self.acobj.wingarea_m2

        if loadfactor >= 0:

            if self.acobj.clmaxclean is False:
                perfmsg = 'Maximum lift coefficient in clean configuration was not specified in performance dictionary'
                raise ValueError(perfmsg)
            cl = self.acobj.clmaxclean

        else:

            if self.acobj.clminclean is False:
                perfmsg = 'Minimum lift coefficient in clean configuration was not specified in performance dictionary'
                raise ValueError(perfmsg)
            cl = self.acobj.clminclean

        rho0_kgm3 = self.acobj.designatm.airdens_kgpm3(altitudes_m=0)
        vs_keas = co.mps2kts(math.sqrt((loadfactor * weight_n) / (0.5 * rho0_kgm3 * wingarea_m2 * cl)))

        return vs_keas

    def _meanchord_m(self):
        """Function for finding the Standard Mean Chord (SMC), a.k.a Mean Geometric Chord, and the
         Mean Aerodynamic Chord (MAC) in a constant taper, trapezoidal wing.

        mac_m = (2 / 3) * c_root * (1 + lambda + lambda ** 2) / (1 + lambda)"""

        taperratio = self.acobj.roottaperratio
        wingarea_m2 = self.acobj.wingarea_m2
        aspectratio = self.acobj.aspectratio

        smc_m = math.sqrt(wingarea_m2 / aspectratio)

        c_root = 2 / (1 + taperratio) * math.sqrt(wingarea_m2 / aspectratio)
        mac_m = (2 / 3) * c_root * (1 + taperratio + taperratio ** 2) / (1 + taperratio)

        return {'SMC': smc_m, 'MAC': mac_m}

    def _paragraph333(self):
        """Flight envelope, as defined by CS-23.

        For normal, utility, commuter, and aerobatic categories of aircraft, returns the maximum
        and minimum limit loads from symmetrical manoeuvres, as well as gust types each category
        must be designed to withstand.

        **Outputs:**

        manoeuvre_dict
            dictionary, with aircraft categories :code:`'norm'`,:code:`'util'`, :code:`'comm'`,
            and :code:`'aero'`. Embedded within each category is a dictionary of minimum and
            maximum manoeuvre limit loads the aircraft should be designed to withstand, in
            different flight conditions. Limit loads in each category are :code:`'npos_D'`,
            :code:`'nneg_C'`, and :code:`'nneg_D'`, relating to cruise and dive speed limits.

        gustmps_dict
            dictionary, with aircraft categories :code:`'norm'`,:code:`'util'`, :code:`'comm'`,
            and :code:`'aero'`. Embedded within each category is a dictionary of derived gust
            velocities U_de, that each category is expected to encounter and must be designed to
            withstand. As per CS-23.333, only :code:`'Ub_mps'` is pertinent to commuter category
            aircraft (rough gusts), whereas :code:`'Uc_mps'` and :code:`'Ud_mps'` apply to all.
        """

        altitude_m = self.altitude_m

        # Create a dictionary of empty dictionaries for each aircraft category
        cs23categories_list = ['norm', 'util', 'comm', 'aero']
        manoeuvre_dict = dict(zip(cs23categories_list, [{} for _ in range(len(cs23categories_list))]))
        gustmps_dict = dict(zip(cs23categories_list, [{} for _ in range(len(cs23categories_list))]))

        # (b) Manoeuvring Envelope

        # (b)(1, 2)
        manoeuvrelimitloads = self._paragraph337()

        for category in cs23categories_list:
            # The aeroplane is to be subjected to sym. manoeuvres that result in +ve limit load for speeds up to V_D
            manoeuvre_dict[category].update({'npos_D': manoeuvrelimitloads[category]['npos_min']})
            # The aeroplane is to be subjected to sym. manoeuvres that result in -ve limit load for speeds up to V_C
            manoeuvre_dict[category].update({'nneg_C': manoeuvrelimitloads[category]['nneg_max']})

        # (b)(3)
        manoeuvre_dict['norm'].update({'nneg_D': 0})
        manoeuvre_dict['util'].update({'nneg_D': -1})
        manoeuvre_dict['comm'].update({'nneg_D': 0})
        manoeuvre_dict['aero'].update({'nneg_D': -1})

        # (c) Gust Envelope

        # (c)(1)
        gustb_mps = np.interp(altitude_m, [co.feet2m(20000), co.feet2m(50000)], [co.feet2m(66), co.feet2m(38)]),
        gustc_mps = np.interp(altitude_m, [co.feet2m(20000), co.feet2m(50000)], [co.feet2m(50), co.feet2m(25)]),
        gustd_mps = np.interp(altitude_m, [co.feet2m(20000), co.feet2m(50000)], [co.feet2m(25), co.feet2m(12.5)])

        gustmps_dict['norm'].update({'Uc_mps': gustc_mps[0], 'Ud_mps': gustd_mps})

        gustmps_dict['util'].update({'Uc_mps': gustc_mps[0], 'Ud_mps': gustd_mps})

        gustmps_dict['comm'].update({'Ub_mps': gustb_mps[0], 'Uc_mps': gustc_mps[0], 'Ud_mps': gustd_mps})

        gustmps_dict['aero'].update({'Uc_mps': gustc_mps[0], 'Ud_mps': gustd_mps})

        # (c)(2) Gust loading assumptions and load factor variation with speed

        return manoeuvre_dict, gustmps_dict

    def _paragraph335(self):
        """Design airspeeds, as per CS-23.335 (see also 14 CFR 23.335).

        For all categories of aircraft, this specification item produces limits for design
        airspeeds.

        **Outputs:**

        eas_dict
            dictionary, containing minimum and maximum allowable design airspeeds in KEAS.

        """

        if self.cruisespeed_keas is False:
            cruisemsg = 'Cruise speed not specified in the csbrief or designbrief dictionary.'
            raise ValueError(cruisemsg)
        vc_keas = self.cruisespeed_keas

        if self.acobj.clmaxclean is False:
            perfmsg = 'CLmaxclean must be specified in the performance dictionary.'
            raise ValueError(perfmsg)
        clmaxclean = self.acobj.clmaxclean

        if self.acobj.weight_n is False:
            designmsg = 'Maximum take-off weight must be specified in the design dictionary.'
            raise ValueError(designmsg)

        if self.acobj.wingarea_m2 is False:
            designmsg = 'Reference wing area must be specified in the design dictionary.'
            raise ValueError(designmsg)
        wingloading_pa = self.acobj.weight_n / self.acobj.wingarea_m2

        wingloading_lbft2 = co.pa2lbfft2(wingloading_pa)

        # Create a dictionary of empty dictionaries for each aircraft category
        cs23categories_list = ['norm', 'util', 'comm', 'aero']
        eas_dict = dict(zip(cs23categories_list, [{} for _ in range(len(cs23categories_list))]))

        # (a) Design cruising speed, V_C

        # (a)(1, 2)
        vcfactor_1i = np.interp(wingloading_lbft2, [20, 100], [33, 28.6])
        vcfactor_1ii = np.interp(wingloading_lbft2, [20, 100], [36, 28.6])

        eas_dict['norm'].update({'vcmin_keas': vcfactor_1i * np.sqrt(wingloading_lbft2)})
        eas_dict['util'].update({'vcmin_keas': vcfactor_1i * np.sqrt(wingloading_lbft2)})
        eas_dict['comm'].update({'vcmin_keas': vcfactor_1i * np.sqrt(wingloading_lbft2)})
        eas_dict['aero'].update({'vcmin_keas': vcfactor_1ii * np.sqrt(wingloading_lbft2)})

        # (a)(3)
        if self.maxlevelspeed_keas is False:
            for category in cs23categories_list:
                eas_dict[category].update({'vcsoftmax_keas': False})
        else:
            for category in cs23categories_list:
                eas_dict[category].update({'vcsoftmax_keas': 0.9 * self.maxlevelspeed_keas}) # This is not strict

        # (a)(4) Requires Mach

        # (b) Design dive speed, V_D

        # (b) (1, 2, 3)
        vdfactor_2i = np.interp(wingloading_lbft2, [20, 100], [1.4, 1.35])
        vdfactor_2ii = np.interp(wingloading_lbft2, [20, 100], [1.5, 1.35])
        vdfactor_2iii = np.interp(wingloading_lbft2, [20, 100], [1.55, 1.35])

        eas_dict['norm'].update({'vdmin_keas': np.fmax(1.25 * vc_keas, vdfactor_2i * eas_dict['norm']['vcmin_keas'])})
        eas_dict['util'].update({'vdmin_keas': np.fmax(1.25 * vc_keas, vdfactor_2ii * eas_dict['util']['vcmin_keas'])})
        eas_dict['comm'].update({'vdmin_keas': np.fmax(1.25 * vc_keas, vdfactor_2i * eas_dict['comm']['vcmin_keas'])})
        eas_dict['aero'].update({'vdmin_keas': np.fmax(1.25 * vc_keas, vdfactor_2iii * eas_dict['aero']['vcmin_keas'])})

        # (b)(4) Requires Mach

        # (c) Design manoeuvring speed, V_A

        # (c)(1, 2)
        wfract = self.weightfraction
        vs_keas = self.vs_keas(loadfactor=wfract)
        manoeuvrelimits = self._paragraph337()

        for category in cs23categories_list:
            eas_dict[category].update({'vamin_keas': vs_keas * math.sqrt(manoeuvrelimits[category]['npos_min'])})
            eas_dict[category].update({'vasoftmax_keas': vc_keas}) # This is not strict

        # (d) Design speed for maximum gust intensity, V_B

        # (d)(1)
        _, gustspeedsmps = self._paragraph333()
        gustloads, k_g, liftslope_prad = self._paragraph341(speedatgust_keas={'Uc': vc_keas})

        vs1_keas = self.vs_keas(loadfactor=wfract)

        for category in cs23categories_list:

            if category == 'comm':
                gust_de_mps = gustspeedsmps[category]['Ub_mps']
            else:
                gust_de_mps = gustspeedsmps[category]['Uc_mps']

            trueloading_pa = wingloading_pa * wfract
            rho0_kgm3 = self.acobj.designatm.airdens_kgpm3(altitudes_m=0)

            a = 1
            b = -(liftslope_prad / clmaxclean) * k_g * gust_de_mps
            c = -2 * trueloading_pa / (rho0_kgm3 * clmaxclean) * wfract
            vbmin1_keas = co.mps2kts((-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a))
            eas_dict[category].update({'vbmin1_keas': vbmin1_keas})

            vbmin2_keas = vs1_keas * math.sqrt(gustloads[category]['npos_Uc'])
            eas_dict[category].update({'vbmin2_keas': vbmin2_keas})

            vbmin_keas = np.fmin(vbmin1_keas, vbmin2_keas)
            eas_dict[category].update({'vbmin_keas': vbmin_keas})

            # (d)(2)
            eas_dict[category].update({'vbmax_keas': np.fmax(vbmin_keas, vc_keas)})

        return eas_dict

    def _paragraph337(self):
        """Limit manoeuvring load factors, as per CS-23.337 (see also 14 CFR 23.337).

        **Outputs:**

        limitload_dict
            dictionary, with aircraft categories :code:`'norm'`,:code:`'util'`, :code:`'comm'`,
            and :code:`'aero'`. Contained within each category is another dictionary of the
            absolute maximum negative limit load, and minimum positive limit load due to aircraft,
            manoeuvre, under keys :code:`'npos_min'` and :code:`'nneg_max'`.

        """
        if self.acobj.weight_n is False:
            designmsg = 'Maximum take-off weight must be specified in the design dictionary.'
            raise ValueError(designmsg)

        if self.acobj.wingarea_m2 is False:
            designmsg = 'Reference wing area must be specified in the design dictionary.'
            raise ValueError(designmsg)

        mtow_n = self.acobj.weight_n

        mtow_lbf = co.n2lbf(mtow_n)

        # Create a dictionary of empty dictionaries for each aircraft category
        cs23categories_list = ['norm', 'util', 'comm', 'aero']
        limitload_dict = dict(zip(cs23categories_list, [{} for _ in range(len(cs23categories_list))]))

        # (a) Positive Limit Manoeuvring Load

        # (a) (1, 2, 3)
        nposminimum = 2.1 + 24000 / (mtow_lbf + 10000)

        limitload_dict['norm'].update({'npos_min': min(3.8, nposminimum)})
        limitload_dict['util'].update({'npos_min': min(4.4, nposminimum)})
        limitload_dict['comm'].update({'npos_min': min(3.8, nposminimum)})
        limitload_dict['aero'].update({'npos_min': min(6.0, nposminimum)})

        # (b) Negative Limit Manoeuvring Load

        # (b) (1, 2)
        limitload_dict['norm'].update({'nneg_max': -0.4 * limitload_dict['norm']['npos_min']})
        limitload_dict['util'].update({'nneg_max': -0.4 * limitload_dict['norm']['npos_min']})
        limitload_dict['comm'].update({'nneg_max': -0.4 * limitload_dict['norm']['npos_min']})
        limitload_dict['aero'].update({'nneg_max': -0.5 * limitload_dict['norm']['npos_min']})

        # (c) Manoeuvring load factors lower than specified above may be used if the aircraft can
        # not exceed these values in flight.

        return limitload_dict

    def _paragraph341(self, speedatgust_keas):
        """Gust load factors (in cruising conditions), as per CS-23.341 (see also 14 CFR 23.341).

        **Parameters:**

        speedatgust_keas
            dictionary, containing the airspeeds at which each gust condition from CS-23.333
            should be evaluated. The airspeed at each condition in KEAS should be passed as
            the value to one or more of the following keys: :code:`'Ub'`, :code:`'Uc'`, and :code:`'Ud'`.

        **Outputs:**

        gustload_dict
            dictionary, with aircraft categories :code:`'norm'`,:code:`'util'`, :code:`'comm'`,
            and :code:`'aero'`. Contained within each category is another dictionary of the
            absolute maximum negative limit load, and minimum positive limit load due to gust,
            under keys :code:`'npos_Ub'`, :code:`'npos_Uc'`, :code:`'npos_Ud'`, :code:`'nneg_Uc'`
            and:code:`'nneg_Ud'`.
        k_g
            float, the gust alleviation factor

        liftslope_prad
            float, the liftslope_prad as calculated by the :code:`constraintanalysis` module, under
            cruise conditions.

        """

        altitude_m = self.altitude_m
        rho_kgm3 = self.acobj.designatm.airdens_kgpm3(altitude_m)

        if self.cruisespeed_keas is False:
            cruisemsg = 'Cruise speed not specified in the csbrief or designbrief dictionary.'
            raise ValueError(cruisemsg)
        cruisespeed_mpstas = co.kts2mps(co.eas2tas(self.cruisespeed_keas, localairdensity_kgm3=rho_kgm3))
        mach = self.acobj.designatm.mach(airspeed_mps=cruisespeed_mpstas, altitude_m=altitude_m)

        if self.acobj.weight_n is False:
            designmsg = 'Maximum take-off weight must be specified in the design dictionary.'
            raise ValueError(designmsg)

        if self.acobj.wingarea_m2 is False:
            designmsg = 'Reference wing area must be specified in the design dictionary.'
            raise ValueError(designmsg)
        wingloading_pa = self.acobj.weight_n / self.acobj.wingarea_m2

        liftslope_prad = self.acobj.liftslope_prad(mach_inf=mach)
        rho0_kgm3 = self.acobj.designatm.airdens_kgpm3()
        wfract = self.weightfraction
        trueloading_pa = wingloading_pa * wfract

        _, gusts_mps = self._paragraph333()

        # Aeroplane mass ratio
        mu_g = (2 * trueloading_pa) / (rho_kgm3 * self._meanchord_m()['SMC'] * liftslope_prad * constants.g)

        # Gust alleviation factor
        k_g = 0.88 * mu_g / (5.3 + mu_g)

        # Gust load factors
        cs23categories_list = ['norm', 'util', 'comm', 'aero']
        gustload_dict = dict(zip(cs23categories_list, [{} for _ in range(len(cs23categories_list))]))

        for category in cs23categories_list:

            for _, (gusttype, gustspeed_mps) in enumerate(gusts_mps[category].items()):

                suffix = gusttype.split('_')[0]

                if suffix in speedatgust_keas:
                    airspeed_keas = [value for key, value in speedatgust_keas.items() if suffix in key][0]
                    airspeed_mps = co.kts2mps(airspeed_keas)
                    q = k_g * rho0_kgm3 * gustspeed_mps * airspeed_mps * liftslope_prad / (2 * trueloading_pa) / wfract
                    poskey = 'npos_' + suffix
                    negkey = 'nneg_' + suffix
                    gustload_dict[category].update({poskey: 1 + q})
                    gustload_dict[category].update({negkey: 1 - q})

        return gustload_dict, k_g, liftslope_prad

    def flightenvelope(self, textsize=None, figsize_in=None, show=True):
        """Construction of the flight envelope, as per CS-23.333(d), see also 14 CFR 23.333. Calling
        this method will plot the flight envelope at a single wing-loading. For examples, see below 
        and in the Jupyter notebook :code:`Constructing V-n diagrams.ipynb` included in
        :code:`docs/ADRpy/notebooks`.

        Note that this V-n diagram should only be seen as indicative. When preparing the documentation
        for establishing the airworthiness of an aircraft, the engineer responsible for the structural
        aspects of the airworthiness must conduct his/her own calculations in establishing the flight
        envelope. 

        **Parameters:**

        textsize
            integer, sets a representative reference fontsize that text in the output plot scale
            themselves in accordance to. Optional, defaults to 10.

        figsize_in
            list, used to specify custom dimensions of the output plot in inches. Image width
            must be specified as a float in the first entry of a two-item list, with height as
            the remaining item. Optional, defaults to 12 inches wide by 7.5 inches tall.

        show
            boolean, used to specify if the plot should be displayed. Optional, defaults to True.

        **Outputs:**

        coords_poi
            dictionary, containing keys :code:`A` through :code:`G`, with values of coordinate tuples.
            These are "points of interest", the speed [KEAS] at which they occur, and the load factor
            they are attributed to.


        **Example** ::

            from ADRpy import airworthiness as aw
            from ADRpy import unitconversions as co
            from ADRpy import atmospheres as at

            designbrief = {}

            designdef = {'aspectratio': 11.1, 'wingarea_m2': 12.1, 'weight_n': 5872}

            designperf = {'CLmaxclean': 1.45, 'CLminclean': -1, 'CLslope': 6.28}

            designpropulsion = "piston" # not specifically needed for the V-n diagram here, required simply for 
                                        # consistency with other classes and to support features included in later releases 

            designatm = at.Atmosphere() # set the design atmosphere to a zero-offset ISA
    
            csbrief={'cruisespeed_keas': 107, 'divespeed_keas': 150,
            'altitude_m': 0,
            'weightfraction': 1, 'certcat': 'norm'}

            concept = aw.CertificationSpecifications(designbrief, designdef, designperf, designatm, designpropulsion, csbrief)

        """

        category = self.category
        cs23categories_list = ['norm', 'util', 'comm', 'aero']
        if category not in cs23categories_list:
            designmsg = 'Valid aircraft category not specified, please select from "{0}", "{1}", "{2}", or "{3}".' \
                .format(cs23categories_list[0], cs23categories_list[1], cs23categories_list[2], cs23categories_list[3])
            raise ValueError(designmsg)
        catg_names = {'norm': "Normal", 'util': "Utility", 'comm': "Commuter", 'aero': "Aerobatic"}

        if textsize is None:
            textsize = 10

        default_figsize_in = [12, 7.5]
        if figsize_in is None:
            figsize_in = default_figsize_in
        elif type(figsize_in) == list:
            if len(figsize_in) != 2:
                argmsg = 'Unsupported figure size, should be length 2, found {0} instead - using default parameters.' \
                    .format(len(figsize_in))
                warnings.warn(argmsg, RuntimeWarning)
                figsize_in = default_figsize_in

        rho0_kgm3 = self.acobj.designatm.airdens_kgpm3()

        if self.cruisespeed_keas is False:
            cruisemsg = 'Cruise speed not specified in the csbrief or designbrief dictionary.'
            raise ValueError(cruisemsg)
        vc_keas = self.cruisespeed_keas

        if self.divespeed_keas is False:
            divemsg = 'Dive speed not specified in the csbrief or designbrief dictionary, using minimum allowable ' \
                      'speed as specified in CS 23.335(b)(2).'
            warnings.warn(divemsg, RuntimeWarning)

        if self.acobj.clmaxclean is False:
            perfmsg = 'Clmaxclean not specified in the performance dictionary'
            raise ValueError(perfmsg)
        clmax = self.acobj.clmaxclean

        if self.acobj.clminclean is False:
            perfmsg = 'Clminclean not specified in the performance dictionary'
            raise ValueError(perfmsg)
        clmin = self.acobj.clminclean

        if self.acobj.weight_n is False:
            designmsg = 'Maximum take-off weight must be specified in the design dictionary.'
            raise ValueError(designmsg)

        if self.acobj.wingarea_m2 is False:
            designmsg = 'Reference wing area must be specified in the design dictionary.'
            raise ValueError(designmsg)
        wingloading_pa = self.acobj.weight_n / self.acobj.wingarea_m2

        wfract = self.weightfraction
        trueloading_pa = wingloading_pa * wfract
        speedlimits_dict = self._paragraph335()[category]
        manoeuvreload_dict, gustspeeds_dict = self._paragraph333()

        # V_C, Cruise Speed
        vcmin_keas = float(speedlimits_dict['vcmin_keas'])
        vcmax_keas = speedlimits_dict['vcsoftmax_keas']
        vc_keas = max(vc_keas, vcmin_keas)
        if (vc_keas > vcmax_keas) and vcmax_keas is not False:
            info = "CS 23.335(a)(3): V_C need not exceed 0.9 * V_H Sea Level."
            warnings.warn(info, UserWarning)

        # V_D, Dive Speed
        vdmin_keas = float(speedlimits_dict['vdmin_keas'])
        if self.divespeed_keas is False:
            vd_keas = vdmin_keas
        else:
            vd_keas = max(self.divespeed_keas, vdmin_keas)

        # V_A, Manoeuvring Speed
        vamin_keas = speedlimits_dict['vamin_keas']
        # vamax_keas = speedlimits_dict['vasoftmax_keas']
        va_keas = vamin_keas
        if va_keas > vc_keas:
            info = "CS 23.335(c)(2): V_A need not exceed V_C used in design."
            warnings.warn(info, UserWarning)

        # V_B, Gust Penetration Speed
        # vbmin_keas = float(speedlimits_dict['vbmin_keas'])
        vbpen_keas = float(speedlimits_dict['vbmin1_keas'])
        # vbmax_keas = float(speedlimits_dict['vbmax_keas'])
        vb_keas = vbpen_keas

        # V_S, Stall Speed
        vs_keas = self.vs_keas(loadfactor=(1*wfract))

        # V_invS, Inverted Stalling Speed
        vis_keas = self.vs_keas(loadfactor=(-1*wfract))
        if vis_keas < vs_keas:
            argmsg = 'Inverted-flight stall speed < Level-flight stall speed, consider reducing design Manoeuvre Speed.'
            warnings.warn(argmsg, RuntimeWarning)

        # V_invA, Inverted Manoeuvring Speed
        viamin_keas = vis_keas * math.sqrt(abs(manoeuvreload_dict[category]['nneg_C']))

        # Gust coordinates
        airspeed_atgust_keas = {'Ub': vb_keas, 'Uc': vc_keas, 'Ud': vd_keas}
        gustloads, _, _ = self._paragraph341(speedatgust_keas=airspeed_atgust_keas)

        # Manoeuvre coordinates
        coords_manoeuvre = {}
        coordinate_list = ['x', 'y']
        # Curve OA
        oa_x = np.linspace(0, va_keas, 100, endpoint=True)
        oa_y = rho0_kgm3 * (co.kts2mps(oa_x)) ** 2 * clmax / trueloading_pa / 2 / wfract
        coords_manoeuvre.update({'OA': dict(zip(coordinate_list, [list(oa_x), list(oa_y)]))})
        # Points D, E, F
        coords_manoeuvre.update({'D': dict(zip(coordinate_list, [vd_keas, manoeuvreload_dict[category]['npos_D'] * (
                    wingloading_pa / trueloading_pa)]))})
        coords_manoeuvre.update({'E': dict(zip(coordinate_list, [vd_keas, manoeuvreload_dict[category]['nneg_D'] * (
                    wingloading_pa / trueloading_pa)]))})
        coords_manoeuvre.update({'F': dict(zip(coordinate_list, [vc_keas, manoeuvreload_dict[category]['nneg_C'] * (
                    wingloading_pa / trueloading_pa)]))})
        # Curve GO
        go_x = np.linspace(viamin_keas, 0, 100, endpoint=True)
        go_y = 0.5 * rho0_kgm3 * co.kts2mps(go_x) ** 2 * clmin / trueloading_pa / wfract
        coords_manoeuvre.update({'GO': dict(zip(coordinate_list, [list(go_x), list(go_y)]))})

        # Flight Envelope coordinates
        coords_envelope = {}
        # Stall Line OS
        coords_envelope.update({'OS': dict(zip(coordinate_list, [[vs_keas, vs_keas], [0, (1/wfract)]]))})
        # Curve+Line SC
        sc_x = np.linspace(vs_keas, vc_keas, 100, endpoint=True)
        sc_y = []
        max_ygust = float(gustloads[category][list(gustloads[category].keys())[0]])
        b_ygustpen = float(rho0_kgm3 * (co.kts2mps(vbpen_keas)) ** 2 * clmax / trueloading_pa / 2 / wfract)
        c_ygust = float(gustloads[category]['npos_Uc'])
        d_ymano = float(manoeuvreload_dict[category]['npos_D'] / wfract)
        for speed in sc_x:
            # If below minimum manoeuvring speed or gust intersection speed, keep on the stall curve
            if (speed <= va_keas) or (speed <= vbpen_keas):
                sc_y.append(rho0_kgm3 * (co.kts2mps(speed)) ** 2 * clmax / trueloading_pa / 2 / wfract)
            # Else the flight envelope is the max of the gust/manoeuvre envelope sizes
            else:  # vbpen_keas > va_keas
                sc_y.append(max(np.interp(speed, [vb_keas, vc_keas], [b_ygustpen, c_ygust]), d_ymano))
        coords_envelope.update({'SC': dict(zip(coordinate_list, [list(sc_x), sc_y]))})
        # Line CD
        cd_x = np.linspace(vc_keas, vd_keas, 100, endpoint=True)
        cd_y = []
        d_ygust = float(gustloads[category]['npos_Ud'])
        for speed in cd_x:
            cd_y.append(float(max(np.interp(speed, [vc_keas, vd_keas], [float(sc_y[-1]), d_ygust]), d_ymano)))
        coords_envelope.update({'CD': dict(zip(coordinate_list, [list(cd_x), cd_y]))})
        # Point E
        e_ygust = float(gustloads[category]['nneg_Ud'])
        e_ymano = manoeuvreload_dict[category]['nneg_D'] * wfract
        e_y = min(e_ygust, e_ymano)
        coords_envelope.update({'E': dict(zip(coordinate_list, [vd_keas, e_y]))})
        # Line EF
        ef_x = np.linspace(vd_keas, vc_keas, 100, endpoint=True)
        ef_y = []
        f_ygust = float(gustloads[category]['nneg_Uc'])
        f_ymano = manoeuvreload_dict[category]['nneg_C'] / wfract
        for speed in ef_x:
            ef_y.append(min(np.interp(speed, [vc_keas, vd_keas], [f_ygust, e_ygust]), e_ymano))
        coords_envelope.update({'EF': dict(zip(coordinate_list, [list(ef_x), ef_y]))})
        # Curve+Line FS
        fs_x = np.linspace(vc_keas, vis_keas, 100, endpoint=True)
        fs_y = []
        for speed in fs_x:
            fs_ystall = rho0_kgm3 * (co.kts2mps(speed)) ** 2 * clmin / trueloading_pa / 2 / wfract
            # If below minimum manoeuvring speed or gust intersection speed, keep on the stall line
            if speed < viamin_keas:
                fs_y.append(fs_ystall)
            else:
                fs_y.append(max(min(np.interp(speed, [0, vc_keas], [1, f_ygust]), f_ymano), fs_ystall))
        coords_envelope.update({'FS': dict(zip(coordinate_list, [list(fs_x), fs_y]))})
        # Stall Line iSO
        coords_envelope.update({'iSO': dict(zip(coordinate_list, [[vis_keas, vis_keas, vs_keas], [-1/wfract, 0, 0]]))})

        # Points of Interest coordinates - These are points that appear in the CS-23.333(d) example
        coords_poi = {}
        coords_poi.update({'A': (va_keas, d_ymano),
                           'B': (vb_keas, b_ygustpen),
                           'C': (vc_keas, sc_y[-1]),
                           'D': (vd_keas, cd_y[-1]),
                           'E': (vd_keas, e_y),
                           'F': (vc_keas, ef_y[-1]),
                           'G': (viamin_keas, go_y[0])
                           })
        if category == 'comm':
            coords_poi.update({'B': (vb_keas, b_ygustpen)})
        if vbpen_keas > vc_keas:
            del coords_poi['B']

        yposlim = max(max_ygust, coords_poi['C'][1], coords_poi['D'][1])
        yneglim = min(coords_poi['E'][1], coords_poi['F'][1])

        if show:
            # Plotting parameters
            fontsize_title = 1.20 * textsize
            fontsize_label = 1.05 * textsize
            fontsize_legnd = 1.00 * textsize
            fontsize_tick = 0.90 * textsize

            fig = plt.figure(figsize=figsize_in)
            fig.canvas.set_window_title('ADRpy airworthiness.py')

            ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])
            ax.set_title("EASA CS-23 Amendment 4 - Flight Envelope ({0} Category)".format(catg_names[category]),
                         fontsize=fontsize_title)
            ax.set_xlabel("Airspeed [KEAS]", fontsize=fontsize_label)
            ax.set_ylabel("Load Factor [-]", fontsize=fontsize_label)
            ax.tick_params(axis='x', labelsize=fontsize_tick)
            ax.tick_params(axis='y', labelsize=fontsize_tick)

            # Gust Lines plotting
            xlist = []
            ylist = []
            for gustindex, (gustloadkey, gustload) in enumerate(gustloads[category].items()):
                gusttype = gustloadkey.split('_')[1]
                gustspeed_mps = round(gustspeeds_dict[category][str(gusttype + '_mps')], 2)
                xlist += [0, airspeed_atgust_keas[gusttype]]
                ylist += [1, gustload]
                if gustload >= 0:
                    # Calculate where gust speed annotations should point to
                    xannotate = 0.15 * xlist[-1]
                    yannotate = 0.15 * (ylist[-1] - 1) + 1
                    xyannotate = xannotate, yannotate
                    # Calculate where the actual annotation text with the gust speed should be positioned
                    offsetx = (abs(4) * (15.24 - gustspeed_mps)) / 4
                    offsety = float((4 - gustindex + 1) * 3 * (12 if gustload < 0 else 10)) ** 0.93
                    label = "$U_{de} = $" + str(gustspeed_mps) + " $ms^{-1}$"
                    # Produce the annotation
                    ax.annotate(label, xy=xyannotate, textcoords='offset points', xytext=(offsetx, offsety),
                                fontsize=fontsize_legnd,
                                arrowprops={'arrowstyle': '->', 'color': 'black', 'alpha': 0.8})
            # Plot the gust lines
            coords = np.array(xlist, dtype=object), np.array(ylist, dtype=object)
            for gustline_idx in range(len(gustloads[category])):
                # Individual gust line coordinates
                xcoord = coords[0][gustline_idx * 2:(gustline_idx * 2) + 2]
                ycoord = coords[1][gustline_idx * 2:(gustline_idx * 2) + 2]
                # Gust lines should be extended beyond the flight envelope, improving their visibility
                xcoord_ext = np.array([xcoord[0], xcoord[1] * 5], dtype=object)
                ycoord_ext = np.array([ycoord[0], ((ycoord[1] - 1) * 5) + 1], dtype=object)
                # Plot the gust lines, but make sure only one label appears in the legend
                if gustline_idx == 0:
                    ax.plot(xcoord_ext, ycoord_ext, c='blue', ls='-.', lw=0.9, alpha=0.8, label='Gust Lines')
                else:
                    ax.plot(xcoord_ext, ycoord_ext, c='blue', ls='-.', lw=0.9, alpha=0.8)

            # Flight Envelope plotting
            xlist = []
            ylist = []
            for _, (k, v) in enumerate(coords_envelope.items()):
                if type(v['x']) != list:
                    xlist.append(v['x'])
                    ylist.append(v['y'])
                else:
                    xlist += v['x']
                    ylist += v['y']
            coords = np.array(xlist, dtype=object), np.array(ylist, dtype=object)
            ax.plot(*coords, c='black', ls='-', lw=1.4, label='Flight Envelope')
            ax.fill(*coords, c='grey', alpha=0.20)

            # Points of Interest plotting - These are points that appear in the CS-23.333(d) example
            class AnyObject(object):
                def __init__(self, text, color):
                    self.my_text = text
                    self.my_color = color

            class AnyObjectHandler(object):
                def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                    patch = mpl_text.Text(x=0, y=0, text=orig_handle.my_text, color=orig_handle.my_color,
                                          verticalalignment=u'baseline', horizontalalignment=u'left',
                                          fontsize=fontsize_legnd)
                    handlebox.add_artist(patch)
                    return patch

            handles_objects_list = []
            labels_list = []
            handler_map = {}
            # First, annotate the V-n diagram with the points of interest clearly labelled
            for _, (k, v) in enumerate(coords_poi.items()):
                # If the speed to be annotated has a positive limit load, annotate with a green symbol, not red
                clr = 'green' if k in ['A', 'B', 'C', 'D'] else 'red'
                handles_objects_list.append(AnyObject(k, clr))
                labels_list.append(
                    ("V: " + str(round(float(v[0]), 1))).ljust(10) + ("| n: " + str(round(float(v[1]), 2))))
                handler_map.update({handles_objects_list[-1]: AnyObjectHandler()})
                offs_spd = vc_keas if c_ygust > max_ygust else vb_keas
                offset = (textsize / 2 if v[0] > offs_spd else (0 - textsize), textsize / 10 if v[1] > 0 else (0 - textsize))
                ax.annotate(k, xy=v, textcoords='offset points', xytext=offset, fontsize=fontsize_label, color=clr)
                plt.plot(*v, 'x', color=clr)
            # Second, create a legend which contains the V-n parameters of each point of interest
            vnlegend = ax.legend(handles_objects_list, labels_list, handler_map=handler_map, loc='center left',
                                 title="V-n Speed [KEAS]; Load [-]", bbox_to_anchor=(1, 0.4),
                                 title_fontsize=fontsize_label,
                                 prop={'size': fontsize_legnd, 'family': 'monospace'})

            # Manoeuvre Envelope plotting
            xlist = []
            ylist = []
            for _, (k, v) in enumerate(coords_manoeuvre.items()):
                if type(v['x']) != list:
                    xlist.append(v['x'])
                    ylist.append(v['y'])
                else:
                    xlist += v['x']
                    ylist += v['y']
            coords = np.array(xlist, dtype=object), np.array(ylist, dtype=object)
            ax.plot(*coords, c='orange', ls='--', lw=1.4, alpha=0.9, label='Manoeuvre Envelope')

            # Create the primary legend
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.75), prop={'size': fontsize_legnd})
            # Add the secondary legend, without destroying the original
            ax.add_artist(vnlegend)
            ax.set_xlim(0, 1.1 * vd_keas)
            ax.set_ylim(1.3 * yneglim, 1.3 * yposlim)
            plt.grid()
            plt.show()
            plt.close(fig=fig)

        return coords_poi

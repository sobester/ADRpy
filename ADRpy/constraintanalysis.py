#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.. _constraints_module:

Constraint Analysis Module
--------------------------

This module contains tools for the constraint analysis of fixed
wing aircraft.

"""

__author__ = "Andras Sobester"

# Other contributors: Yaseen Reza

# pylint: disable=locally-disabled, too-many-instance-attributes
# pylint: disable=locally-disabled, too-many-branches
# pylint: disable=locally-disabled, too-many-statements
# pylint: disable=locally-disabled, too-many-locals
# pylint: disable=locally-disabled, too-many-lines

import math
import warnings
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants
from scipy.interpolate import interp1d

from ADRpy import atmospheres as at
from ADRpy import unitconversions as co
from ADRpy import mtools4acdc as actools
from ADRpy import propulsion as pdecks


class AircraftConcept:
    """Definition of a basic aircraft concept. An object of this class defines an
    aircraft design in terms of the *brief* it is aiming to meet, high
    level *design* variables that specify it, key parameters that describe
    its *performance*, as well as the *atmosphere* it operates in. These are
    the four arguments that define an object of the AircraftConcept class.
    The first three are dictionaries, as described below, the last is an object
    of `Atmosphere <https://adrpy.readthedocs.io/en/latest/#atmospheres.Atmosphere>`_
    class.

    **Parameters:**

    brief
        Dictionary. Definition of the design brief, that is, the requirements
        the design seeks to meet. Contains the following key names:

        climbalt_m
            Float. The altitude (in metres) where the climb rate requirement is specified.
            Optional, defaults to zero (sea level).

        climbspeed_kias
            Float. The airspeed (in knots, indicated) at which the required climb
            rate has to be achieved.

        climbrate_fpm
            Float. Required climb rate (in feet per minute) at the altitude specified
            in the *climbalt_m* entry (above).

        cruisealt_m
            Float. The altitude at which the cruise speed requirement will be defined.

        cruisespeed_ktas
            Float. The required cruise speed (in knots, true airspeed) at the
            altitude specified in the *cruisealt_m* entry (above).

        cruisethrustfact
            Float. The fraction (nondimensional) of the maximum available thrust at which
            the cruise speed requirement must be achieved.

        servceil_m
            Float. The required service ceiling in meters (that is, the altitude at which
            the maximum rate of climb drops to 100 feet per minute).

        secclimbspd_kias
            Float. The speed (knots indicated airspeed) at which the service ceiling
            must be reached. This should be an estimate of the best rate of climb speed.

        vstallclean_kcas
            Float. The maximum acceptable stall speed (in knots, indicated/calibrated).

        groundrun_m
            Float. Length (in metres) of take-off ground run in meters at the elevation
            defined by the *rwyelevation_m* entry of the dictionary. This is a basic,
            100% N1, no wind, zero runway gradient ground run.

        rwyelevation_m
            Float. The elevation (in metres) of the runway againts which the take-off
            constraint is defined. Optional, defaults to zero (sea level).

        to_headwind_kts
            Float. The speed of the take-off headwind (in knots), parallel to the runway.
            Optional, defaults to zero.

        to_slope_perc
            Float. The percent gradient of the runway in the direction of travel. Optional,
            defaults to zero.

        stloadfactor
            Float. Load factor to be sustained by the aircraft in a steady, level turn.

        turnalt_m
            Float. Altitude (in metres) where the turn requirement is defined.
            Optional, defaults to zero (sea level).

        turnspeed_ktas
            Float. True airspeed (in knots) at which the turn requirement (above) has to be met.
            Since the dynamics of turning flight is dominated by inertia, which depends
            on ground speed, the turn speed is specified here as TAS (on the zero wind assumption).
            If you'd rather specify this as IAS/CAS/EAS,
            use `eas2tas <https://adrpy.readthedocs.io/en/latest/#atmospheres.Atmosphere.eas2tas>`_
            first to obtain the TAS value.


        Example design brief::

            brief = {'rwyelevation_m':0, 'groundrun_m':313,
                     'stloadfactor': 1.5, 'turnalt_m': 1000, 'turnspeed_ktas': 100,
                     'climbalt_m': 0, 'climbspeed_kias': 101, 'climbrate_fpm': 1398,
                     'cruisealt_m': 3048, 'cruisespeed_ktas': 182, 'cruisethrustfact': 1.0,
                     'servceil_m': 6580, 'secclimbspd_kias': 92,
                     'vstallclean_kcas': 69}

    design
        Dictionary. Definition of key, high level design variables that define the future
        design.

        aspectratio
            Float. Wing aspect ratio. Optional, defaults to 8.

        sweep_le_deg
            Float. Main wing leading edge sweep angle (in degrees). Optional, defaults to
            zero (no sweep).

        sweep_mt_deg
            Float. Main wing sweep angle measured at the maximum thickness point. Optional,
            defaults to value of 'sweep_le_deg'.

        sweep_25_deg
            Float. Main wing sweep angle measured at the quarter chord point. Optional,
            defaults to ~29% sweep_le_deg, ~71% sweep_mt_deg.

        roottaperratio
            Float. Standard definition of wing tip chord to root chord ratio, zero for sharp,
            pointed wing-tip delta wings. Optional, defaults to the theoretical optimal value
            as a function of the quarter-chord sweep angle.

        wingarea_m2
            Float. Total reference area of the wing (in metres squared).

        wingheightratio
            Float. The ratio of altitude h to wingspan b, used for the calculation of ground
            effect. Optional, defaults to 100 (produces a ground effect factor of near unity).

        bpr
            Float. Specifies the propulsion system type. For jet engines (powered by axial
            gas turbines) this should be the bypass ratio (hence *'bpr'*).

            *Deprecated: Set to -1 for piston engines, -2 for turboprops and -3 if no power/thrust
            corrections are needed (e.g., for electric motors).

        spooluptime_s
            Float. Time in seconds for the engine to reach take-off thrust. Optional, defaults to 5.

        totalstaticthrust_n
            Float. Maximum thrust achievable at zero airspeed.

        tr
            Float. Throttle ratio for gas turbine engines. *tr = 1* means that the Turbine Entry
            Temperature will reach its maximum allowable value in sea level standard day
            conditions, so higher ambient temperatures will result in power loss. Higher *tr*
            values mean thrust decay starting at higher altitudes.

        weight_n
            Float. Specifies the maximum take-off weight of the aircraft.

        weightfractions
            Dictionary. Specifies at what fraction of the maximum take-off weight do various
            constraints have to be met. It should contain the following keys: *take-off*,
            *climb*, *cruise*, *turn*, *servceil*. Optional, each defaults to 1.0 if not
            specified.

        runwayalpha_deg
            Float. Angle of attack the main wing encounters during take-off roll. Optional,
            defaults to 0.

        runwayalpha_max_deg
            Float. Maximum permitted angle of attack before lift-off.

    performance
        Dictionary. Definition of key, high level design performance estimates.

        CD0TO
            Float. Zero-lift drag coefficient in the take-off configuration

        CDTO
            Float. Take-off drag coefficient. Optional, defaults to 0.09.

        CDminclean
            Float. Zero lift drag coefficient in clean configuration. Optional, defaults to 0.03.

        mu_R
            Float. Coefficient of rolling resistance on the wheels. Optional, defaults to 0.03.

        CL0TO
            Float. Zero-alpha lift coefficient.

        CLTO
            Float. Take-off lift coefficient. Optional, defaults to 0.95.

        CLmaxTO
            Float. Maximum lift coefficient in take-off conditions. Optional, defaults to 1.5.

        CLmaxclean
            Float. Maximum lift coefficient in flight, in clean configuration.

        CLminclean
            Float. Minimum lift coefficient in flight, in clean configuration. Typically negative.

        CLslope
            Float. Lift-curve slope gradient, or Cl/alpha of a design aerofoil (or wing that may
            be considered 2D) in incompressible flow. Optional, defaults to the flat plate theory
            maximum of 2*Pi.

        etaprop
            Dictionary. Propeller efficiency in various phases of the mission.
            It should contain the following keys: *take-off*, *climb*, *cruise*,
            *turn*, *servceil*. Optional, unspecified entries in the dictionary
            default to the following values:

            :code: `etap = {'take-off': 0.45, 'climb': 0.75, 'cruise': 0.85, 'turn': 0.85, 'servceil': 0.65}`

    designatm
            `Atmosphere <https://adrpy.readthedocs.io/en/latest/#atmospheres.Atmosphere>`_
            class object. Specifies the virtual atmosphere in which all the design calculations
            within the *AircraftConcept* class will be performed. Optional, defaults to the
            International Standard Atmosphere.

    propulsion
            Tuple. Contains at maximum two objects of the ADRpy propulsion module, specifying
            the nature of the aircraft propulsion system. The first item of the tuple should be
            an ADRpy propulsion EngineDeck class object, followed by an optional ADRpy
            PropellerDeck class object.

            String. An alternative to specifying propulsion objects, specify a generic type of
            propulsion from either :code: `"turboprop"`, :code: `"piston"`, :code: `"electric"`,
            or :code: `"jet"`.
    """

    def __init__(self, brief=None, design=None, performance=None, designatm=None, propulsion=None):

        # Parse the input arguments
        self.brief = brief = {} if not brief else brief
        self.design = design = {} if not design else design
        self.performance = performance = {} if not performance else performance
        self.designatm = at.Atmosphere(offset_deg=0) if not designatm else designatm

        # If propulsion is an iterable (string, or a tuple of length 2), accept the input
        if actools.iterable(propulsion):
            if ((isinstance(propulsion, tuple)) and (len(propulsion) == 2)) or isinstance(propulsion, str):
                self.propulsion = propulsion
            else:
                self.propulsion = False
        # If a single object is passed as an argument, and it is not None, accept the input and package as tuple
        elif not isinstance(propulsion, type(None)):
            self.propulsion = (propulsion, None)
        else:
            self.propulsion = False

        # Specify the default flags or parameters for the design brief, if parameter is left unspecified

        default_brief = {
            # Climb Constraint
            'climbalt_m': 0,  # Assign sea level (h = 0 metres)
            'climbspeed_kias': False,  # Flag as not specified
            'climbrate_fpm': False,  # Flag as not specified

            # Cruise Constraint
            'cruisealt_m': False,  # Flag as not specified
            'cruisespeed_ktas': False,  # Flag as not specified
            'cruisethrustfact': 1.0,  # Assume 100% throttle in cruise

            # Service Ceiling Constraint
            'servceil_m': False,  # Flag as not specified
            'secclimbspd_kias': False,  # Flag as not specified

            # Stall Constraint
            'vstallclean_kcas': False,  # Flag as not specified

            # Take-off Constraint
            'groundrun_m': False,  # Flag as not specified
            'rwyelevation_m': 0,  # Assign sea level (h = 0 metres)
            'to_headwind_kts': 0,
            'to_slope_perc': 0,

            # Turn Constraint
            'stloadfactor': False,  # Flag as not specified
            'turnalt_m': 0,  # Assign sea level (h = 0 metres)
            'turnspeed_ktas': False,  # Flag as not specified
        }

        # Specify the default flags or parameters for the design definition, if parameter is left unspecified

        default_design = {
            # Definitions of basic aircraft geometry
            'aspectratio': 8,
            'sweep_le_deg': False,  # Flag as not specified (further comprehension required)
            'sweep_mt_deg': False,  # Flag as not specified (further comprehension required)
            'sweep_25_deg': False,  # Flag as not specified (further comprehension required)
            'roottaperratio': False,  # Flag as not specified (further comprehension required)
            'wingarea_m2': False,  # Flag as not specified
            'wingheightratio': 100,  # Large number if unspecified, to provide WIG factor of near unity

            # Properties of aircraft propulsion system
            'bpr': -1,  # Piston engine propulsion type
            'spooluptime_s': 5,
            'totalstaticthrust_n': False,  # Flag as not specified
            'tr': 1.07,  # Throttle ratio (theta break)

            # Aircraft weight and loading
            'weight_n': False,  # Flag as not specified
            'weightfractions': {  # Assume all constraints apply at the same weight (e.g., electrically powered a/c)
                'climb': 1.0, 'cruise': 1.0, 'servceil': 1.0, 'turn': 1.0
            },
            # Runway alpha
            'runwayalpha_deg': 0,
            'runwayalpha_max_deg': False,  # Flag as not specified
        }

        # Specify the default flags or parameters for the design performance, if parameter is left unspecified

        default_performance = {
            # Drag/Resistive coefficients
            'CD0TO': False,  # Flag as not specified
            'CDTO': 0.09,
            'CDminclean': 0.03,
            'mu_R': 0.03,

            # Lift coefficients
            'CL0TO': False,  # Flag as not specified
            'CLTO': 0.95,
            'CLmaxTO': 1.5,
            'CLmaxclean': False,  # Flag as not specified
            'CLminclean': False,  # Flag as not specified
            'CLslope': 2 * math.pi,

            # Propeller Efficiency
            'etaprop': {  # Flag as not specified, further comprehension required
                'climb': False, 'cruise': False, 'servceil': False, 'take-off': False, 'turn': False
            }
        }

        # Use the templates (default dictionaries) to populate missing values in the provided design dictionaries

        # Aggregate the design brief, design, and performance dictionaries, as a "library" of design dictionaries
        default_designlib = [default_brief, default_design, default_performance]
        designlib = [brief, design, performance]
        # Take a single design dictionary out of the library
        for chapter_i in range(len(default_designlib)):
            # Iterate through the items in the default design dictionary pulled from the defaults library
            for _, (dict_k, dict_v) in enumerate(default_designlib[chapter_i].items()):

                # If a parameter was not specified, copy in a flag/value/sub-dictionary from the default dictionary
                if dict_k not in designlib[chapter_i]:
                    if type(dict_v) == dict:
                        designlib[chapter_i][dict_k] = dict_v.copy()
                    else:
                        designlib[chapter_i][dict_k] = dict_v
                # Else if a parameter was specified as a dictionary, check sub-params were given (or else populate them)
                elif type(designlib[chapter_i][dict_k]) == dict:
                    for _, (subdict_k, subdict_v) in enumerate(default_designlib[chapter_i][dict_k].items()):
                        if subdict_k not in designlib[chapter_i][dict_k]:
                            designlib[chapter_i][dict_k][subdict_k] = subdict_v

        # Populate the AircraftConcept object with attributes

        # FURTHER COMPREHENSION: If sweep angles were not specified
        if design['sweep_le_deg'] is False:
            design['sweep_le_deg'] = 0
        if design['sweep_mt_deg'] is False:
            design['sweep_mt_deg'] = design['sweep_le_deg']
        # If x/c=25% sweep not specified, assume max thickness occurs at x/c=35% and interpolate
        if design['sweep_25_deg'] is False:
            design['sweep_25_deg'] = ((2 * design['sweep_le_deg']) + (5 * design['sweep_mt_deg']) / 7)

        # FURTHER COMPREHENSION: If taper ratio was not given, use an estimate for the optimal taper ratio
        # https://www.fzt.haw-hamburg.de/pers/Scholz/OPerA/OPerA_PRE_DLRK_12-09-10_MethodOnly.pdf
        self.taperdefaultflag = False
        if design['roottaperratio'] is False:
            if type(design['sweep_25_deg']) == list:
                avgsweep25deg = sum(design['sweep_25_deg']) / len(design['sweep_25_deg'])
            else:
                avgsweep25deg = design['sweep_25_deg']
            design['roottaperratio'] = 0.45 * math.exp(-0.0375 * avgsweep25deg)
            self.taperdefaultflag = True

        # FURTHER COMPREHENSION: If flight-phase etaprop was not declared, assign defaults
        self.etadefaultflag = 0
        default_etaprop = {
            'climb': 0.75, 'cruise': 0.85, 'servceil': 0.65, 'take-off': 0.45, 'turn': 0.85
        }
        for _, (subdict_k, subdict_v) in enumerate(performance['etaprop'].items()):
            if subdict_v is False:
                self.etadefaultflag += 1
                performance['etaprop'][subdict_k] = default_etaprop[subdict_k]
        self.etaprop = performance['etaprop']

        # Package all parameters, specified and unspecified, into a library attribute of the potential design space
        self.designspace = deepcopy(designlib)

        # If a parameter was specified as a list, the parameter attribute should return an average of the list
        for chapter_i in range(len(default_designlib)):
            for _, (dict_k, dict_v) in enumerate(default_designlib[chapter_i].items()):
                if type(designlib[chapter_i][dict_k]) == list:
                    designlib[chapter_i][dict_k] = sum(designlib[chapter_i][dict_k]) / len(designlib[chapter_i][dict_k])

        # Package all parameters, specified and unspecified, into a library of nominal design values
        self.designstate = [brief, design, performance]

        # Climb Constraint
        self.climbalt_m = brief['climbalt_m']
        self.climbspeed_kias = brief['climbspeed_kias']
        self.climbrate_fpm = brief['climbrate_fpm']

        # Cruise Constraint
        self.cruisealt_m = brief['cruisealt_m']
        self.cruisespeed_ktas = brief['cruisespeed_ktas']
        self.cruisethrustfact = brief['cruisethrustfact']

        # Service Ceiling Constraint
        self.servceil_m = brief['servceil_m']
        self.secclimbspd_kias = brief['secclimbspd_kias']

        # Stall Constraint
        self.vstallclean_kcas = brief['vstallclean_kcas']

        # Take-off Constraint
        self.groundrun_m = brief['groundrun_m']
        self.rwyelevation_m = brief['rwyelevation_m']
        self.to_headwind_kts = brief['to_headwind_kts']
        self.to_slope_perc = brief['to_slope_perc']
        self.to_slope_rad = math.atan(self.to_slope_perc / 100)
        self.to_slope_deg = math.degrees(self.to_slope_rad)

        # Turn Constraint
        self.turnalt_m = brief['turnalt_m']
        self.turnspeed_ktas = brief['turnspeed_ktas']
        self.stloadfactor = brief['stloadfactor']

        # Aircraft Geometry
        self.aspectratio = design['aspectratio']
        self.sweep_le_deg = design['sweep_le_deg']
        self.sweep_le_rad = math.radians(self.sweep_le_deg)
        self.sweep_25_deg = design['sweep_25_deg']
        self.sweep_25_rad = math.radians(self.sweep_25_deg)
        self.sweep_mt_deg = design['sweep_mt_deg']
        self.sweep_mt_rad = math.radians(self.sweep_mt_deg)
        self.roottaperratio = design['roottaperratio']
        self.wingarea_m2 = design['wingarea_m2']
        self.wingheightratio = design['wingheightratio']

        # Properties of Propulsion System
        self.bpr = design['bpr']
        self.spooluptime_s = design['spooluptime_s']
        self.totalstaticthrust_n = design['totalstaticthrust_n']
        self.throttle_r = design['tr']

        # Aircraft Loading
        self.weight_n = design['weight_n']
        self.climb_weight_fraction = design['weightfractions']['climb']
        self.cruise_weight_fraction = design['weightfractions']['cruise']
        self.sec_weight_fraction = design['weightfractions']['servceil']
        self.turn_weight_fraction = design['weightfractions']['turn']

        # Runway alpha
        self.runwayalpha_deg = design['runwayalpha_deg']
        self.runwayalpha_max_deg = design['runwayalpha_max_deg']

        # Drag/Resistive coefficients
        self.CD0TO = performance['CD0TO']
        self.cdto = performance['CDTO']
        self.cdminclean = performance['CDminclean']
        self.mu_r = performance['mu_R']

        # Lift coefficients
        self.CL0TO = performance['CL0TO']
        self.clto = performance['CLTO']
        self.clmaxclean = performance['CLmaxclean']
        self.clminclean = performance['CLminclean']
        self.clmaxto = performance['CLmaxTO']
        self.a_0i = performance['CLslope']

        # Propulsion Efficiency
        self.etaprop_climb = performance['etaprop']['climb']
        self.etaprop_cruise = performance['etaprop']['cruise']
        self.etaprop_sec = performance['etaprop']['servceil']
        self.etaprop_to = performance['etaprop']['take-off']
        self.etaprop_turn = performance['etaprop']['turn']

        return

    # Three different estimates the Oswald efficiency factor:

    def oswaldspaneff1(self):
        """Raymer's Oswald span efficiency estimate, sweep < 30, moderate AR"""
        return 1.78 * (1 - 0.045 * (self.aspectratio ** 0.68)) - 0.64

    def oswaldspaneff2(self):
        """Oswald span efficiency estimate due to Brandt et al."""
        sqrtterm = 4 + self.aspectratio ** 2 * (1 + (math.tan(self.sweep_mt_rad)) ** 2)
        return 2 / (2 - self.aspectratio + math.sqrt(sqrtterm))

    def oswaldspaneff3(self):
        """Raymer's Oswald span efficiency estimate, swept wings"""
        return 4.61 * (1 - 0.045 * (self.aspectratio ** 0.68)) * ((math.cos(self.sweep_le_rad)) ** 0.15) - 3.1

    def oswaldspaneff4(self, mach_inf=None):
        """Method for estimating the oswald factor from basic aircraft geometrical parameters;
        Original method by Mihaela Nita and Dieter Scholz, Hamburg University of Applied Sciences
        https://www.dglr.de/publikationen/2012/281424.pdf

        The method returns an estimate for the Oswald efficiency factor of a planar wing. The mach
        correction factor was fitted around subsonic transport aircraft, and therefore this method
        is recommended only for use in subsonic analysis with free-stream Mach < 0.69.

        **Parameters:**

        mach_inf
            float, Mach number at which the Oswald efficiency factor is to be estimated, required
            to evaluate compressibility effects. Optional, defaults to 0.3 (incompressible flow).

        **Outputs:**

        e
            float, predicted Oswald efficiency factor for subsonic transport aircraft.

        """

        if mach_inf is None:
            argmsg = 'Mach number unspecified, defaulting to incompressible flow condition'
            warnings.warn(argmsg, RuntimeWarning)
            mach_inf = 0.3

        # THEORETICAL OSWALD FACTOR: For calculating the inviscid drag due to lift only

        taperratio = self.roottaperratio

        # Calculate Hoerner's delta/AR factor for unswept wings (with NASA's swept wing study, fitted for c=25% sweep)
        dtaperratio = -0.357 + 0.45 * math.exp(-0.0375 * self.sweep_25_deg)
        tapercorr = taperratio - dtaperratio
        k_hoernerfactor = (0.0524 * tapercorr ** 4) - (0.15 * tapercorr ** 3) + (0.1659 * tapercorr ** 2) - (
                0.0706 * tapercorr) + 0.0119
        e_theo = 1 / (1 + k_hoernerfactor * self.aspectratio)

        # CORRECTION FACTOR F: Kroo's correction factor due to reduced lift from fuselage presence
        dfoverb_all = 0.114
        ke_fuse = 1 - 2 * (dfoverb_all ** 2)

        # CORRECTION FACTOR D0: Correction factor due to viscous drag from generated lift
        ke_d0 = 0.85

        # CORRECTION FACTOR M: Correction factor due to compressibility effects on induced drag
        mach_compressible = 0.3
        # M. Nita and D. Scholz, constants from statistical analysis of subsonic aircraft
        if mach_inf > mach_compressible:
            ke_mach = -0.001521 * (((mach_inf / mach_compressible) - 1) ** 10.82) + 1
        else:
            ke_mach = 1

        e = e_theo * ke_fuse * ke_d0 * ke_mach

        if e < 1e-3:
            e = 1e-3
            calcmsg = 'Specified Mach ' + str(mach_inf) + ' is out of bounds for oswaldspaneff4, e_0 ~= 0'
            warnings.warn(calcmsg, RuntimeWarning)

        return e

    def induceddragfact(self, whichoswald=None, mach_inf=None):
        """Lift induced drag factor k estimate (Cd = Cd0 + K.Cl^2) based on the relationship
            (k = 1 / pi * AR * e_0).

        **Parameters:**

        whichoswald
            integer, used to specify the method(s) to estimate e_0 from. Specifying a single digit
            integer selects a single associated method, however a concatenated string of integers
            can be used to specify that e_0 should be calculated from the average of several.
            Optional, defaults to methods 2 and 4.

        mach_inf
            float, the free-stream flight mach number. Optional, defaults to 0.3 (incompressible
            flow prediction).

        **Outputs:**

        induceddragfactor
            float, an estimate for the coefficient of Cl^2 in the drag polar (Cd = Cd0 + K.Cl^2)
            based on various estimates of the oswald efficiency factor.

        **Note**
        This method does not contain provisions for 'wing-in-ground-effect' factors.

        """

        # Identify all the digit characters passed in the whichoswald argument, and assemble as a list of single digits
        oswaldeff_list = []
        if type(whichoswald) == int:
            selection_list = [int(i) for i in str(whichoswald)]
            # k = 1 / pi.AR.e
            if 1 in selection_list:
                oswaldeff_list.append(self.oswaldspaneff1())
            if 2 in selection_list:
                oswaldeff_list.append(self.oswaldspaneff2())
            if 3 in selection_list:
                oswaldeff_list.append(self.oswaldspaneff3())
            if 4 in selection_list:
                # If whichoswald = 4 was *specifically* selected, then throw a warning if Mach was not given
                if mach_inf is None:
                    argmsg = 'Mach number unspecified, defaulting to incompressible flow condition'
                    warnings.warn(argmsg, RuntimeWarning)
                    mach_inf = 0.3
                oswaldeff_list.append(self.oswaldspaneff4(mach_inf=mach_inf))

        # If valid argument(s) were given, take the average of their Oswald results
        if len(oswaldeff_list) > 0:
            oswaldeff = sum(oswaldeff_list) / len(oswaldeff_list)
        # Else default to estimate 2 and 4, Brandt and Nita incompressible
        else:
            oswaldeff = 0.5 * (self.oswaldspaneff2() + self.oswaldspaneff4(mach_inf=0.3))

        return 1.0 / (math.pi * self.aspectratio * oswaldeff)

    def findchordsweep_rad(self, xc_findsweep):
        """Calculates the sweep angle at a given chord fraction, for a constant taper wing

        **Parameters:**

        xc_findsweep
            float, the fraction of chord along which the function is being asked to determine the
            sweep angle of. Inputs are bounded as 0 <= xc_findsweep <= 1 (0% to 100% chord),
            where x/c = 0 is defined as the leading edge.

        **Outputs:**

        sweep_rad
            float, this is the sweep angle of the given chord fraction, for a constant taper wing.
        """

        if xc_findsweep is None:
            argmsg = 'Function can not find the sweep angle without knowing the x/c to investigate'
            raise ValueError(argmsg)

        elif not (0 <= xc_findsweep <= 1):
            argmsg = 'Function was called with an out of bounds chord, tried (0 <= x/c <= 1)'
            raise ValueError(argmsg)

        sweeple_rad = self.sweep_le_rad
        sweep25_rad = self.sweep_25_rad

        # Use rate of change of sweep angle with respect to chord progression
        sweep_roc = (sweeple_rad - sweep25_rad) / -0.25
        sweep_rad = sweeple_rad + sweep_roc * xc_findsweep

        return sweep_rad

    def liftslope_prad(self, mach_inf=None):
        """Method for estimating the lift-curve slope from aircraft geometry; Methods from
        http://naca.central.cranfield.ac.uk/reports/arc/rm/2935.pdf (Eqn. 80), by D. Kuchemann;
        DATCOM 1978;

        Several methods for calculating supersonic and subsonic lift-slopes are aggregated to
        produce a model for the lift curve with changing free-stream Mach number.

        **Parameters:**

        mach_inf
            float, the free-stream flight mach number. Optional, defaults to 0.3 (incompressible
            flow prediction).

        **Outputs:**

        liftslope_prad
            float, the predicted lift slope as an average of several methods of computing it, for
            a 'thin' aerofoil (t/c < 5%) - assuming the aircraft is designed with supersonic flight
            in mind. Units of rad^-1.

        **Note**

        Care must be used when interpreting this function in the transonic flight regime. This
        function departs from theoretical models for 0.6 <= Mach_free-stream <= 1.4, and instead
        uses a weighted average of estimated curve-fits and theory to predict transonic behaviour.

        """

        if mach_inf is None:
            argmsg = 'Mach number unspecified, defaulting to incompressible flow condition'
            warnings.warn(argmsg, RuntimeWarning)
            mach_inf = 0.3

        aspectr = self.aspectratio
        a_0i = self.a_0i
        piar = math.pi * aspectr
        oswald = self.oswaldspaneff2()
        sweep_25_rad = self.sweep_25_rad
        sweep_le_rad = self.sweep_le_rad

        # Define transition points of models by Mach
        puresubsonic_mach = 0.6
        puresupsonic_mach = 1.4
        lowertranson_mach = 0.8
        uppertranson_mach = 1.3

        def a_subsonic(machsub):
            """From subsonic mach, determine an approximate lift slope"""
            slopeslist_sub = []

            beta_00 = math.sqrt(1 - machsub ** 2)
            beta_le = math.sqrt(1 - (machsub * math.cos(sweep_le_rad)) ** 2)

            # Subsonic 3-D Wing Lift Slope, with Air Compressibility and Sweep Effects
            sqrt_term = 1 + (piar / a_0i / math.cos(sweep_25_rad)) ** 2 * beta_00 ** 2
            a_m0 = piar / (1 + math.sqrt(sqrt_term))
            slopeslist_sub.append(a_m0)

            # High-Aspect-Ratio Straight Wings
            a_m3 = a_0i / (beta_00 + (a_0i / piar / oswald))
            slopeslist_sub.append(a_m3)

            # Low-Aspect-Ratio Straight Wings
            a_m4 = a_0i / (math.sqrt(beta_00 ** 2 + (a_0i / piar) ** 2) + (a_0i / piar))
            slopeslist_sub.append(a_m4)

            # Low-Aspect-Ratio Swept Wings
            a_0_le = a_0i * math.cos(sweep_le_rad)
            a_m5 = a_0_le / (math.sqrt(beta_le ** 2 + (a_0_le / piar) ** 2) + (a_0_le / piar))
            slopeslist_sub.append(a_m5)

            # DATCOM model for sub-sonic lift slope
            sweep_50_rad = self.findchordsweep_rad(xc_findsweep=0.5)
            kappa = a_0i / (2 * math.pi)  # Implementation of 2D lift slope needs checking here
            a_m6 = (2 * piar) / (
                    2 + math.sqrt((aspectr * beta_00 / kappa) ** 2 * (1 + (math.tan(sweep_50_rad) / beta_00) ** 2) + 4))
            slopeslist_sub.append(a_m6)

            # D. Kuchemann's method for subsonic, straight or swept wings
            sweep_50_rad = self.findchordsweep_rad(xc_findsweep=0.5)
            a_0c = a_0i / beta_00
            a_0_50 = a_0c * math.cos(sweep_50_rad)  # Mid-chord sweep, lift slope
            sweep_eff = sweep_50_rad / (1 + (a_0_50 / piar) ** 2) ** 0.25
            a_0eff = a_0c * math.cos(sweep_eff)  # Effective sweep
            powerterm = 1 / (4 * (1 + (abs(sweep_eff) / (0.5 * math.pi))))
            n_s = 1 - (1 / (2 * (1 + (a_0eff / piar) ** 2) ** powerterm))  # Shape parameter, swept wing
            a_m7 = ((2 * a_0eff * n_s) / (1 - (math.pi * n_s) * (1 / math.tan(math.pi * n_s)) + (
                    (4 * a_0eff * n_s ** 2) / piar)))
            slopeslist_sub.append(a_m7)

            # D. Kuchemann's method for delta wings with pointed tips and straight trailing edges, up to AR ~ 2.5
            a_0c = a_0i / beta_00
            a_m8 = (a_0c * aspectr) / (math.sqrt(4 + aspectr ** 2 + (a_0c / math.pi) ** 2) + a_0c / math.pi)
            slopeslist_sub.append(a_m8)

            return sum(slopeslist_sub) / len(slopeslist_sub)

        def a_supersonic(machsuper):
            """From supersonic mach, determine an approximate lift slope"""
            slopeslist_sup = []

            beta_00 = math.sqrt(machsuper ** 2 - 1)

            # Supersonic Delta Wings
            if sweep_le_rad != 0:  # Catch a divide by zero if the LE sweep is zero (can't be a delta wing)
                if machsuper < 1:
                    sweep_shock_rad = 0
                else:
                    sweep_shock_rad = math.acos(1 / machsuper)  # NOT MACH ANGLE, this is sweep! Mach 1 = 0 deg sweep

                m = math.tan(sweep_shock_rad) / math.tan(sweep_le_rad)
                if 0 <= m <= 1:  # Subsonic leading edge case
                    lambda_polynomial = m * (0.38 + (2.26 * m) - (0.86 * m ** 2))
                    a_m2 = (2 * math.pi ** 2 * (1 / math.tan(sweep_le_rad))) / (math.pi + lambda_polynomial)
                else:  # Supersonic leading edge case, linear inviscid theory
                    a_m2 = 4 / beta_00
                slopeslist_sup.append(a_m2)

            # High-Aspect-Ratio Straight Wings
            a_m3 = 4 / beta_00
            slopeslist_sup.append(a_m3)

            # Low-Aspect-Ratio Straight Wings
            a_m4 = 4 / beta_00 * (1 - (1 / 2 / aspectr / beta_00))
            slopeslist_sup.append(a_m4)

            return sum(slopeslist_sup) / len(slopeslist_sup)

        if mach_inf < puresubsonic_mach:  # Subsonic regime, Mach_inf < mach_sub
            liftslope_prad = a_subsonic(mach_inf)

        elif mach_inf > puresupsonic_mach:  # Supersonic regime, Mach_inf > mach_sup
            liftslope_prad = a_supersonic(mach_inf)

        else:  # Transonic regime, mach_sub < Mach_inf < mach_sup

            # Thickness-to-chord ratio
            tcratio = 0.05

            # Find where the lift-slope peaks
            def slopepeak_mach(aspectratio):
                # http://naca.central.cranfield.ac.uk/reports/1955/naca-report-1253.pdf
                # Assume quadratic fit of graph data from naca report 1253
                def genericquadfunc(x, a, b, c):
                    return a * x ** 2 + b * x + c

                # These are pregenerated static values, to save computational resources
                popt = np.array([3.2784414, -9.73119668, 6.0546588])

                # Create x-data for A(t/c)^(1/3), and then y-data for Speed parameter (M ** 2 - 1) / ((t/c) ** (2/3))
                xfitdata = np.linspace(0, 1.5, 200)
                yfitdata = genericquadfunc(xfitdata, *popt)

                # Convert to x-data for AR, and y-data for Mach
                arfitdata = xfitdata / (tcratio ** (1 / 3))
                machfitdata = (yfitdata * (tcratio ** (2 / 3)) + 1) ** 0.5

                # For a provided aspect ratio, determine if the quadratic or the linear relation should be used
                arinfit_index = np.where(arfitdata >= aspectratio)[0]
                if len(arinfit_index) > 0:
                    machquery = machfitdata[min(arinfit_index)]
                else:
                    machquery = min(machfitdata)
                return machquery

            # This is the Mach number where the liftslope_prad should peak
            mach_apk = slopepeak_mach(aspectratio=aspectr)
            cla_son = (math.pi / 2) * aspectr  # Strictly speaking, an approximation only true for A(t/c)^(1/3) < 1

            delta = 3e-2
            x_mach = []
            y_cla = []
            # Subsonic transition points
            x_mach.extend([puresubsonic_mach, puresubsonic_mach + delta])
            y_cla.extend([a_subsonic(machsub=x_mach[0]), a_subsonic(machsub=x_mach[1])])

            # Lift-slope peak transition points
            x_mach.extend([mach_apk - 2 * delta, mach_apk + 2 * delta])
            y_cla.extend([0.95 * cla_son, 0.95 * cla_son])

            # Supersonic transition points
            x_mach.extend([puresupsonic_mach - delta, puresupsonic_mach])
            y_cla.extend([a_supersonic(machsuper=x_mach[-2]), a_supersonic(machsuper=x_mach[-1])])

            # Recast lists as arrays
            x_mach = np.array(x_mach)
            y_cla = np.array(y_cla)

            interpf = interp1d(x_mach, y_cla, kind='cubic')
            a_transonic = interpf(mach_inf)

            # The slope is weighted either as pure subsonic, subsonic, pure transonic, supersonic, or pure supersonic
            if mach_inf < 1:
                weight_sub = np.interp(mach_inf, [puresubsonic_mach, lowertranson_mach], [1, 0])
                weight_sup = 0
            else:
                weight_sub = 0
                weight_sup = np.interp(mach_inf, [uppertranson_mach, puresupsonic_mach], [0, 1])

            liftslope_prad = 0
            if puresubsonic_mach < mach_inf < puresupsonic_mach:
                liftslope_prad += (1 - weight_sub - weight_sup) * a_transonic
            if mach_inf < lowertranson_mach:
                liftslope_prad += weight_sub * a_subsonic(machsub=mach_inf)
            elif mach_inf > uppertranson_mach:
                liftslope_prad += weight_sup * a_supersonic(machsuper=mach_inf)

        # To be implemented later: weighted averages for the subsonic and supersonic regimes
        # Lift slope values: Straight Wing (Highest a values) > Delta Wing > Swept Wing (Lowest a values)
        # Lift slope values: High aspect ratio (Highest a values) > Low aspect ratio (Lowest a values)

        return liftslope_prad

    def induceddragfact_lesm(self, wingloading_pa=None, cl_real=None, mach_inf=None):
        """Lift induced drag factor k estimate (Cd = Cd0 + k.Cl^2), from LE suction theory, for aircraft
        capable of supersonic flight.

        **Parameters:**

        wingloading_pa
            float or array, list of wing-loading values in Pa. Optional, provided that an
            aircraft weight and wing area are specified in the design definitions dictionary.

        cl_real
            float or array, the coefficient of lift demanded to perform a maneuver. Optional,
            defaults to cl at cruise.

        mach_inf
            float, Mach number at which the Oswald efficiency factor is to be estimated, required
            to evaluate compressibility effects. Optional, defaults to 0.3 (incompressible flow).

        **Outputs:**

        k
            float, predicted lift-induced drag factor K, as used in (Cd = Cd0 + k.Cl^2)

        **Note**

        This method does not contain provisions for 'wing-in-ground-effect' factors.

        """

        if mach_inf is None:
            argmsg = 'Mach number unspecified, defaulting to incompressible flow condition.'
            warnings.warn(argmsg, RuntimeWarning)
            mach_inf = 0.3

        if cl_real is None:
            argmsg = 'Coefficient of lift attained unspecified, defaulting to cruise lift coefficient.'
            warnings.warn(argmsg, RuntimeWarning)

        if wingloading_pa is None:
            if self.weight_n is False:
                designmsg = 'Maximmum take-off weight not specified in the design definitions dictionary.'
                raise ValueError(designmsg)
            if self.wingarea_m2 is False:
                designmsg = 'Wing area not specified in the design definitions dictionary.'
                raise ValueError(designmsg)
            wingloading_pa = self.weight_n / self.wingarea_m2
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        if (self.cruisespeed_ktas is False) or (self.cruisealt_m is False):
            cruisemsg = 'Cruise Cl could not be determined (missing cruise speed/altitude' \
                        ' in the designbrief dictionary), defaulting to 0.6.'
            warnings.warn(cruisemsg, RuntimeWarning)
            cl_cruise = 0.6
        else:
            cruisespeed_mps = co.kts2mps(self.cruisespeed_ktas)
            qcruise_pa = self.designatm.dynamicpressure_pa(cruisespeed_mps, self.cruisealt_m)
            cl_cruise = wingloading_pa * self.cruise_weight_fraction / qcruise_pa

        aspectr = self.aspectratio
        sweep_le_rad = self.sweep_le_rad
        machstar_le = 1.0 / math.cos(sweep_le_rad)  # Free-stream mach number required for sonic LE condition

        # Estimate subsonic k with the oswald factor
        k_oswald = self.induceddragfact(whichoswald=24, mach_inf=min(mach_inf, 0.6))

        # Estimate full regime k from Leading-Edge-Suction method (Aircraft Design, Daniel P. Raymer)

        # Zero-suction case
        k_0 = 1 / self.liftslope_prad(mach_inf=mach_inf)

        # Non-zero-suction case (This function for k_100 produces a messy curve, this needs smoothing somehow)
        if mach_inf < 1:  # Aircraft free-stream mach number is subsonic

            k_100 = 1.0 / (math.pi * aspectr)  # Full-suction case, oswald e = 1

        elif mach_inf < machstar_le:  # Free-stream is (super)sonic, but wing leading edge sees subsonic flow

            # Boundary conditions
            x1, x2 = 1.0, machstar_le
            y1, y2 = 1.0 / (math.pi * aspectr), 1 / self.liftslope_prad(mach_inf=machstar_le)
            m1 = 0

            # Solve simultaneous equations
            mat_a = np.array([[x1 ** 2, x1, 1],
                              [x2 ** 2, x2, 1],
                              [2 * x1, 1, 0]])
            mat_b = np.array([y1, y2, m1])
            mat_x = np.linalg.solve(mat_a, mat_b)

            # The polynomial describing the suction case between sonic freestream and sonic leading edge mach
            k_100 = 0
            order = 2
            for index in range(order + 1):
                k_100 += (mat_x[index] * mach_inf ** (order - index))

        else:  # Aircraft is fully supersonic

            k_100 = k_0  # Suction can not take place, therefore k_100 = k_0

        # Find the leading edge suction factor S(from model of Raymer data, assuming max suction ~93 %)

        # Suction model
        def y_suction(cl_delta, cl_crs, a, c, r):
            k = (-0.5 * cl_crs ** 2) - (0.25 * cl_crs) - 0.22
            b = 1 + r * k
            x = cl_delta
            y = a * (x - b) * np.exp(-c * (x - 0.1)) * -np.tan(0.1 * (x - k))
            return y

        if (cl_real is None) or (cl_cruise is None):
            cl_diff = 0
        else:
            cl_diff = cl_real - cl_cruise

        # Suction model for design Cl=0.3 and Cl=0.8
        y_03 = y_suction(cl_delta=cl_diff, cl_crs=0.3, a=22.5, c=1.95, r=0)
        y_08 = y_suction(cl_delta=cl_diff, cl_crs=0.8, a=5.77, c=1, r=-1.29)

        # Find suction at actual cl as a weight of the two sample curves
        weight = np.interp(cl_cruise, [0.3, 0.8], [1, 0])
        suctionfactor = weight * y_03 + (1 - weight) * y_08

        k_suction = suctionfactor * k_100 + (1 - suctionfactor) * k_0

        # Take k to be the weighted average between a subsonic oswald, and supersonic suction prediction
        weight = np.interp(mach_inf, [0, machstar_le], [1, 0]) ** 0.2

        k_predicted = (k_oswald * weight + k_suction * (1 - weight))

        return k_predicted

    def bestclimbspeedprop(self, wingloading_pa, altitude_m):
        """The best rate of climb speed for a propeller aircraft"""

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        dragfactor = np.sqrt(self.induceddragfact() / (3 * self.cdminclean))
        densfactor = 2 / self.designatm.airdens_kgpm3(altitude_m)

        # Gudmundsson, eq. (18-27)
        bestspeed_mps = np.sqrt(densfactor * wingloading_pa * dragfactor)

        if len(bestspeed_mps) == 1:
            return bestspeed_mps[0]

        return bestspeed_mps

    def _propulsion_slcorr(self, atmosphere_obj, airspeed_mpstas, altitude_m):
        """Altitude corrections, depending on propulsion system (or propulsion system type)"""

        propulsion = self.propulsion
        if self.propulsion is False:
            propulsionmsg = 'A valid propulsion system type was not specified in the "propulsion" arg instantiating ' \
                            'this class object. Defaulting to piston engine.'
            warnings.warn(propulsionmsg, RuntimeWarning)

            if self.bpr >= 0:
                propulsion = 'jet'
            elif self.bpr == - 1:
                propulsion = 'piston'
            elif self.bpr == -2:
                propulsion = 'turboprop'
            elif self.bpr == -3:
                propulsion = 'electric'
            elif -3 <= self.bpr < 0:
                designmsg = 'Specifying propulsion system type with "bpr" in the design dictionary is deprecated, ' \
                            'please consider using the "propulsion" argument when instantiating objects of this class.'
                warnings.warn(designmsg, FutureWarning)
            else:
                designmsg = 'Invalid bypass ratio bpr = {0} specified.'.format(str(self.bpr))
                raise ValueError(designmsg)

        mach = atmosphere_obj.mach(airspeed_mps=airspeed_mpstas, altitude_m=altitude_m)
        tcorr = np.ones(len(mach))  # Default correction value (No correction required) for Thrust
        pcorr = np.ones(len(mach))  # Default correction value (No correction required) for Power

        # If the propulsion type is generic, identified by a string input
        if isinstance(propulsion, str):
            temp_c = atmosphere_obj.airtemp_c(altitude_m)
            pressure_pa = atmosphere_obj.airpress_pa(altitude_m)
            density_kgpm3 = atmosphere_obj.airdens_kgpm3(altitude_m)

            for i in range(len(mach)):

                if propulsion == 'turboprop':
                    # J. D. Mattingly (full reference in atmospheres module)
                    tcorr[i] = at.turbopropthrustfactor(temp_c, pressure_pa, mach[i], self.throttle_r)

                elif propulsion == 'piston':
                    # J. D. Mattingly (full reference in atmospheres module)
                    tcorr[i] = at.pistonpowerfactor(density_kgpm3)

                elif propulsion == 'electric':
                    # No altitude corrections required for electric propulsion
                    pass

                elif propulsion == 'jet':
                    # J. D. Mattingly (full reference in atmospheres module)
                    if self.bpr == 0:
                        tcorr[i] = at.turbojetthrustfactor(temp_c, pressure_pa, mach[i], self.throttle_r, False)
                    elif 0 < self.bpr < 5:
                        tcorr[i] = at.turbofanthrustfactor(temp_c, pressure_pa, mach[i], self.throttle_r, "lowbpr")
                    elif 5 <= self.bpr:
                        tcorr[i] = at.turbofanthrustfactor(temp_c, pressure_pa, mach[i], self.throttle_r, "highbpr")
                    else:
                        propulsionmsg = 'Was not expecting negative "self.bpr" for "jet" propulsion system type!'
                        raise ValueError(propulsionmsg)

                else:
                    propulsionmsg = 'Propulsion system identifier "{0}" was not recognised amongst an accepted ' \
                                    'list of inputs.'.format(str(self.propulsion))
                    warnings.warn(propulsionmsg, RuntimeWarning)

        # If the propulsion type is specified by the contents of a tuple
        elif isinstance(propulsion, tuple):

            engine_obj, _ = self.propulsion

            if isinstance(engine_obj, pdecks.TurbopropDeck):
                pcorr = engine_obj.sl_powercorr(mach, altitude_m)

            elif isinstance(engine_obj, pdecks.PistonDeck):
                # What is the engine shaft RPM that produces the greatest shaft power for a given altitude?
                minrpm, maxrpm = min(engine_obj.pwr_data[0]), max(engine_obj.pwr_data[0])
                speed_rpm_array = np.arange(minrpm, maxrpm)
                power_w_array = engine_obj.shaftpower(speed_rpm_array, altitude_m)
                maxpwrrpm_idx = np.where(speed_rpm_array == np.nanmax(speed_rpm_array))
                bestpwrspeed_rpm = speed_rpm_array[maxpwrrpm_idx]

                pcorr = engine_obj.sl_powercorr(bestpwrspeed_rpm, altitude_m)

            elif isinstance(engine_obj, pdecks.ElectricDeck):
                # No altitude corrections required for electric propulsion
                tcorr = 1
                pcorr = 1

            elif isinstance(engine_obj, pdecks.JetDeck):
                tcorr = engine_obj.sl_thrustcorr(mach, altitude_m)

        # Check if thrust or power corrections were defined - if not, the map doesnt exist
        if 'tcorr' not in locals():
            tcorr = 1
            tcorrmsg = 'Could not find sea-level thrust mapping for specified propulsion type.'
            warnings.warn(tcorrmsg, RuntimeWarning)
        if 'pcorr' not in locals():
            pcorr = 1
            pcorrmsg = 'Could not find sea-level power mapping for specified propulsion type.'
            warnings.warn(pcorrmsg, RuntimeWarning)

        return tcorr, pcorr

    def twrequired_clm(self, wingloading_pa, map2sl=True):
        """Calculates the T/W required for climbing for a range of wing loadings.

        **Parameters:**

        wingloading_pa
            float or array, list of wing-loading values in Pa.

        map2sl
            boolean, specifies if the result in the conditions specified, should be mapped
            to sea level equivalents. Optional, defaults to True.

        **Outputs:**

        twratio
            array, thrust to weight ratio required for the given wing loadings.

        **See also** ``twrequired``

        **Notes**

        1. Use `twrequired` if a full constraint analysis is desired, as this integrates
        the take-off, turn, climb, cruise, and service ceiling constraints, as well as
        computing the combined constraint boundary.

        2. The calculation currently approximates climb performance on the constant TAS
        assumption (though note that the design brief dictionary variable must specify the
        climb speed as IAS, which is the operationally relevant figure) - a future version
        of the code will remove this approximation and assume constant IAS.

        **Example**

        Given a climb rate (in feet per minute) and a climb speed (KIAS), as well as an
        altitude (in a given atmosphere) where these must be achieved, as well as
        a set of basic geometrical and aerodynamic performance parameters, compute the
        necessary T/W ratio to hold the specified climb rate.

        ::

            from ADRpy import atmospheres as at
            from ADRpy import constraintanalysis as ca

            designbrief = {'climbalt_m': 0, 'climbspeed_kias': 101,
                        'climbrate_fpm': 1398}

            etap = {'climb': 0.8}

            designperformance = {'CDminclean': 0.0254, 'etaprop' :etap}

            designdef = {'aspectratio': 10.12, 'sweep_le_deg': 2,
                        'sweep_mt_deg': 0, 'bpr': -1}

            TOW_kg = 1542.0

            designatm = at.Atmosphere()

            concept = ca.AircraftConcept(designbrief, designdef,
                                        designperformance, designatm)

            wingloadinglist_pa = [1250, 1500, 1750]

            twratio = concept.twrequired_clm(wingloadinglist_pa)

            print('T/W: ', twratio)

        Output: ::

            T/W:  [ 0.20249491  0.2033384   0.20578177]

        """

        if self.climbspeed_kias is False:
            turnmsg = 'Climb speed not specified in the designbrief dictionary.'
            raise ValueError(turnmsg)
        climbspeed_mpsias = co.kts2mps(self.climbspeed_kias)

        # Assuming that the climb rate is 'indicated'
        if self.climbrate_fpm is False:
            turnmsg = 'Climb rate not specified in the designbrief dictionary.'
            raise ValueError(turnmsg)
        climbrate_mps = co.fpm2mps(self.climbrate_fpm)

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        climbspeed_mpstas = self.designatm.eas2tas(climbspeed_mpsias, self.climbalt_m)
        climbrate_mpstroc = self.designatm.eas2tas(climbrate_mps, self.climbalt_m)

        # What SL T/W will yield the required T/W at the actual altitude?
        tcorr, _ = self._propulsion_slcorr(self.designatm, climbspeed_mpstas, self.climbalt_m)

        # W/S at the start of the specified climb segment may be less than MTOW/S
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        wsclimb_pa = wingloading_pa * self.climb_weight_fraction

        qclimb_pa = self.designatm.dynamicpressure_pa(climbspeed_mpstas, self.climbalt_m)
        cl_climb = wsclimb_pa / qclimb_pa
        mach = self.designatm.mach(climbspeed_mpstas, self.climbalt_m)
        inddragfact = self.induceddragfact_lesm(wingloading_pa=wingloading_pa, cl_real=cl_climb, mach_inf=mach)
        cos_sq_theta = (1 - (climbrate_mpstroc / climbspeed_mpstas) ** 2)

        # To be implemented, as 1 + (V/g)*(dV/dh)
        accel_fact = 1.0

        twratio = accel_fact * climbrate_mpstroc / climbspeed_mpstas + (
                1 / wsclimb_pa) * qclimb_pa * self.cdminclean + (
                          inddragfact / qclimb_pa) * wsclimb_pa * cos_sq_theta

        if map2sl:
            twratio = twratio / tcorr

        # Map back to T/MTOW if climb start weight is less than MTOW
        twratio = twratio * self.climb_weight_fraction

        if len(twratio) == 1:
            return twratio[0]

        return twratio

    def twrequired_crs(self, wingloading_pa, map2sl=True):
        """Calculate the T/W required for cruise for a range of wing loadings

        **Parameters:**

        wingloading_pa
            float or array, list of wing-loading values in Pa.

        map2sl
            boolean, specifies if the result in the conditions specified, should be mapped
            to sea level equivalents. Optional, defaults to True.

        **Outputs:**

        twratio
            array, thrust to weight ratio required for the given wing loadings.

        **See also** ``twrequired``
        """

        if self.cruisespeed_ktas is False:
            cruisemsg = 'Cruise speed not specified in the designbrief dictionary.'
            raise ValueError(cruisemsg)
        cruisespeed_mpstas = co.kts2mps(self.cruisespeed_ktas)

        if self.cruisealt_m is False:
            cruisemsg = 'Cruise altitude not specified in the designbrief dictionary.'
            raise ValueError(cruisemsg)

        # What SL T/W will yield the required T/W at the actual altitude?
        tcorr, _ = self._propulsion_slcorr(self.designatm, cruisespeed_mpstas, self.cruisealt_m)

        # W/S at the start of the cruise may be less than MTOW/S
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        wscruise_pa = wingloading_pa * self.cruise_weight_fraction

        qcruise_pa = self.designatm.dynamicpressure_pa(cruisespeed_mpstas, self.cruisealt_m)
        cl_cruise = wscruise_pa / qcruise_pa
        mach = self.designatm.mach(cruisespeed_mpstas, self.cruisealt_m)
        inddragfact = self.induceddragfact_lesm(wingloading_pa=wingloading_pa, cl_real=cl_cruise, mach_inf=mach)

        twratio = (1 / wscruise_pa) * qcruise_pa * self.cdminclean + (inddragfact / qcruise_pa) * wscruise_pa

        if map2sl:
            twratio = twratio / tcorr

        # Map back to T/MTOW if cruise start weight is less than MTOW
        twratio = twratio * self.cruise_weight_fraction

        twratio = twratio * (1 / self.cruisethrustfact)

        if len(twratio) == 1:
            return twratio[0]

        return twratio

    def twrequired_sec(self, wingloading_pa, map2sl=True):
        """T/W required for a service ceiling for a range of wing loadings

        **Parameters:**

        wingloading_pa
            float or array, list of wing-loading values in Pa.

        map2sl
            boolean, specifies if the result in the conditions specified, should be mapped
            to sea level equivalents. Optional, defaults to True.

        **Outputs:**

        twratio
            array, thrust to weight ratio required for the given wing loadings.

        **See also** ``twrequired``
        """

        if self.servceil_m is False:
            secmsg = 'Climb rate not specified in the designbrief dictionary.'
            raise ValueError(secmsg)

        if self.secclimbspd_kias is False:
            secmsg = 'Best climb speed not specified in the designbrief dictionary.'
            raise ValueError(secmsg)

        secclimbspeed_mpsias = co.kts2mps(self.secclimbspd_kias)
        secclimbspeed_mpstas = self.designatm.eas2tas(secclimbspeed_mpsias, self.servceil_m)

        # What SL T/W will yield the required T/W at the actual altitude?
        tcorr, _ = self._propulsion_slcorr(self.designatm, secclimbspeed_mpstas, self.servceil_m)

        # W/S at the start of the service ceiling test point may be less than MTOW/S
        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        wsservceil_pa = wingloading_pa * self.sec_weight_fraction

        qservceil_pa = self.designatm.dynamicpressure_pa(secclimbspeed_mpstas, self.servceil_m)
        cl_servceil = wsservceil_pa / qservceil_pa
        mach = self.designatm.mach(secclimbspeed_mpstas, self.servceil_m)
        inddragfact = self.induceddragfact_lesm(wingloading_pa=wingloading_pa, cl_real=cl_servceil, mach_inf=mach)

        # Service ceiling typically defined in terms of climb rate (at best climb speed) of
        # dropping to 100feet/min ~ 0.508m/s
        climbrate_mps = co.fpm2mps(100)

        # What true climb rate does 100 feet/minute correspond to?
        climbrate_mpstroc = self.designatm.eas2tas(climbrate_mps, self.servceil_m)

        twratio = climbrate_mpstroc / secclimbspeed_mpstas + (1 / wsservceil_pa) * qservceil_pa * self.cdminclean + (
                inddragfact / qservceil_pa) * wsservceil_pa

        if map2sl:
            twratio = twratio / tcorr

        # Map back to T/MTOW if service ceiling test start weight is less than MTOW
        twratio = twratio * self.sec_weight_fraction

        if len(twratio) == 1:
            return twratio[0]

        return twratio

    def thrusttoweight_takeoff(self, wingloading_pa):
        """The thrust to weight ratio required for take-off. This function is an
        implementation of the following simple, analytical model:

        .. math::

            \\frac{\\overline{T}}{W} = 1.21\\frac{W/S}{\\rho C_\\mathrm{Lmax}^\\mathrm{TO}gd_
            \\mathrm{G}}+\\frac{1}{2}\\frac{C_\\mathrm{D}^\\mathrm{TO}}{C_\\mathrm{L}^\\mathrm{TO}}
            +\\frac{1}{2}\\mu_\\mathrm{R}

        where :math:`\\overline{T}` is the average thrust during the take-off run,
        :math:`W/S` is the wing loading, :math:`d_\\mathrm{G}` is the required ground
        roll, :math:`C_\\mathrm{D}^\\mathrm{TO}` and :math:`C_\\mathrm{L}^\\mathrm{TO}`
        are the 'all wheels on the runway' drag and lift coefficient respectively
        in the take-off configuration, :math:`C_\\mathrm{Lmax}^\\mathrm{TO}` is the maximum
        lift coefficient achieved during the take-off run (during rotation), :math:`\\rho`
        is the ambient density and :math:`\\mu_\\mathrm{R}` is the coefficient of rolling
        resistance on the wheels.

        This is a function exposed to the user for clarity and added flexibility.
        If you need to calculate the thrust to weight ratio required for take-off, use
        ``twrequired_to``. This corrects the output of this function to account for the
        environmental conditions (including their impact on engine performance) and includes
        a mapping to static thrust. ``thrusttoweight_takeoff`` should only be used if you
        would like to perform these corrections in a different way than implemented in
        ``twrequired_to``.

        If a full constraint analysis is required, ``twrequired`` should be used.
        A similar 'full constraint set' function is available for calculating the
        power demanded of the engine or electric motor of a propeller-driven aircraft
        (to satisfy the constraint set) - this is called ``powerrequired``.
        """

        groundrun_m = self.groundrun_m

        # Assuming that the lift-off speed is equal to VR, which we estimate at 1.1VS1(T/O)
        density_kgpm3 = self.designatm.airdens_kgpm3(self.rwyelevation_m)

        vs1to_mps = np.sqrt((2 * wingloading_pa) / (density_kgpm3 * self.clmaxto))

        liftoffspeed_mpstas = 1.1 * vs1to_mps

        thrusttoweightreqd = (liftoffspeed_mpstas ** 2) / (
                2 * constants.g * groundrun_m) + 0.5 * self.cdto / self.clto + 0.5 * self.mu_r

        return thrusttoweightreqd, liftoffspeed_mpstas

    def twrequired_to(self, wingloading_pa, map2sl=True):
        """Calculate the T/W required for take-off for a range of wing loadings

        **Parameters:**

        wingloading_pa
            float or array, list of wing-loading values in Pa.

        map2sl
            boolean, specifies if the result in the conditions specified, should be mapped
            to sea level equivalents. Optional, defaults to True.

        **Outputs:**

        twratio
            array, thrust to weight ratio required for the given wing loadings.

        liftoffspeed_mpstas
            array, liftoff speeds (TAS - true airspeed) in m/s.

        avspeed_mps
            average speed (TAS) during the take-off run, in m/s.

        **See also** ``twrequired``

        **Notes**

        1. The calculations here assume a 'no wind' take-off, conflating ground speed (GS) and
        true airspeed (TAS).

        2. Use `twrequired` if a full constraint analysis is desired, as this integrates
        the take-off, turn, climb, cruise, and service ceiling constraints, as well as
        computing the combined constraint boundary.

        **Example** ::

            from ADRpy import atmospheres as at
            from ADRpy import constraintanalysis as ca

            designbrief = {'rwyelevation_m':1000, 'groundrun_m':1200}
            designdefinition = {'aspectratio':7.3, 'bpr':3.9, 'tr':1.05}
            designperformance = {'CDTO':0.04, 'CLTO':0.9, 'CLmaxTO':1.6, 'mu_R':0.02}

            wingloadinglist_pa = [2000, 3000, 4000, 5000]

            atm = at.Atmosphere()
            concept = ca.AircraftConcept(designbrief, designdefinition,
                                        designperformance, atm)

            tw_sl, liftoffspeed_mpstas, _ = concept.twrequired_to(wingloadinglist_pa)

            print(tw_sl)
            print(liftoffspeed_mpstas)

        Output: ::

            [ 0.19397876  0.26758006  0.33994772  0.41110154]
            [ 52.16511207  63.88895348  73.77260898  82.48028428]

        """
        if self.groundrun_m is False:
            tomsg = 'Ground run not specified in the designbrief dictionary.'
            raise ValueError(tomsg)

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        twratio, liftoffspeed_mpstas = self.thrusttoweight_takeoff(wingloading_pa)

        # What does this required T/W mean in terms of static T/W required?
        twratio = self.map2static() * twratio

        # What SL T/W will yield the required T/W at the actual altitude?
        if map2sl:
            for i, los_mpstas in enumerate(liftoffspeed_mpstas):
                tcorr, _ = self._propulsion_slcorr(self.designatm, los_mpstas, self.rwyelevation_m)

                twratio[i] = twratio[i] / tcorr

        avspeed_mpstas = liftoffspeed_mpstas / np.sqrt(2)

        if len(twratio) == 1:
            return twratio[0], liftoffspeed_mpstas[0], avspeed_mpstas[0]

        return twratio, liftoffspeed_mpstas, avspeed_mpstas

    def bank2turnradius(self, bankangle_deg):
        """Calculates the turn radius in m, given the turn TAS and the bank angle"""

        bankangle_rad = math.radians(bankangle_deg)
        v_mps = co.kts2mps(self.turnspeed_ktas)

        r_m = (v_mps ** 2) / (constants.g * math.tan(bankangle_rad))

        return r_m

    def thrusttoweight_sustainedturn(self, wingloading_pa):
        """Baseline T/W req'd for sustaining a given load factor at a certain altitude"""

        cdmin = self.cdminclean
        nturn = self.stloadfactor
        turnalt_m = self.turnalt_m
        turnspeed_mpstas = co.kts2mps(self.turnspeed_ktas)

        mach = self.designatm.mach(turnspeed_mpstas, turnalt_m)
        wsclimb_pa = wingloading_pa * self.climb_weight_fraction

        qturn = self.designatm.dynamicpressure_pa(airspeed_mps=turnspeed_mpstas, altitudes_m=turnalt_m)
        cl_turn = wsclimb_pa * nturn / qturn
        inddragfact = self.induceddragfact_lesm(wingloading_pa=wingloading_pa, cl_real=cl_turn, mach_inf=mach)

        twreqtrn = qturn * (cdmin / wingloading_pa + inddragfact * ((nturn / qturn) ** 2) * wsclimb_pa)

        return twreqtrn, cl_turn

    def twrequired_trn(self, wingloading_pa, map2sl=True):
        """Calculates the T/W required for turning for a range of wing loadings

        **Parameters:**

        wingloading_pa
            float or array, list of wing-loading values in Pa.

        map2sl
            boolean, specifies if the result in the conditions specified, should be mapped
            to sea level equivalents. Optional, defaults to True.

        **Outputs:**

        twratio
            array, thrust to weight ratio required for the given wing loadings.

        clrequired
            array, lift coefficient values required for the turn (see notes).

        feasibletw
            as twratio, but contains NaNs in lieu of unachievable (CLmax exceeded) values.

        **See also** ``twrequired``

        **Notes**

        1. Use `twrequired` if a full constraint analysis is desired, as this integrates
        the take-off, turn, climb, cruise, and service ceiling constraints, as well as
        computing the combined constraint boundary.

        2. At the higher end of the wing loading range (low wing area values) the CL required
        to achieve the required turn rate may exceed the maximum clean CL (as specified in the
        `CLmaxclean` entry in the `performance` dictionary argument of the `AircraftConcept`
        class object being used). This means that, whatever the T/W ratio, the wings will stall
        at this point. The basic T/W value will still be returned in `twratio`, but there is
        another output, `feasibletw`, which is an array of the same T/W values, with those
        values blanked out (replaced with NaN) that cannot be achieved due to CL exceeding
        the maximum clean lift coefficient.

        **Example**

        Given a load factor, an altitude (in a given atmosphere) and a true airspeed, as well as
        a set of basic geometrical and aerodynamic performance parameters, compute the necessary
        T/W ratio to hold that load factor in the turn.

        ::

            from ADRpy import atmospheres as at
            from ADRpy import constraintanalysis as ca
            from ADRpy import unitconversions as co

            designbrief = {'stloadfactor': 2, 'turnalt_m': co.feet2m(10000),
                        'turnspeed_ktas': 140}

            etap = {'turn': 0.85}

            designperformance = {'CLmaxclean': 1.45, 'CDminclean':0.02541,
                                'etaprop': etap}

            designdef = {'aspectratio': 10.12, 'sweep_le_deg': 2,
                        'sweep_mt_deg': 0, 'bpr': -1}

            designatm = at.Atmosphere()

            concept = ca.AircraftConcept(designbrief, designdef,
            designperformance, designatm)

            wingloadinglist_pa = [1250, 1500, 1750]

            twratio, clrequired, feasibletw = concept.twrequired_trn(wingloadinglist_pa)

            print('T/W:               ', twratio)
            print('Only feasible T/Ws:', feasibletw)
            print('CL required:       ', clrequired)
            print('CLmax clean:       ', designperformance['CLmaxclean'])

        Output:

        ::

            T/W:                [ 0.19920641  0.21420513  0.23243016]
            Only feasible T/Ws: [ 0.19920641  0.21420513         nan]
            CL required:        [ 1.06552292  1.2786275   1.49173209]
            CLmax clean:        1.45

        """

        if self.turnspeed_ktas is False:
            turnmsg = 'Turn speed not specified in the designbrief dictionary.'
            raise ValueError(turnmsg)

        if self.stloadfactor is False:
            turnmsg = 'Turn load factor not specified in the designbrief dictionary.'
            raise ValueError(turnmsg)

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        # W/S at the start of the specified turn test may be less than MTOW/S
        wsturn_pa = wingloading_pa * self.turn_weight_fraction

        twratio, clrequired = self.thrusttoweight_sustainedturn(wsturn_pa)

        # What SL T/W will yield the required T/W at the actual altitude?
        turnspeed_mpstas = co.kts2mps(self.turnspeed_ktas)
        tcorr, _ = self._propulsion_slcorr(self.designatm, turnspeed_mpstas, self.turnalt_m)

        if map2sl:
            twratio = twratio / tcorr

        # Map back to T/MTOW if turn start weight is less than MTOW
        twratio = twratio * self.turn_weight_fraction

        # Which of these points is actually reachable given the clean CLmax?
        feasibletw = np.copy(twratio)
        for idx, val in enumerate(clrequired):
            if val > self.clmaxclean:
                feasibletw[idx] = np.nan

        if len(twratio) == 1:
            return twratio[0], clrequired[0], feasibletw[0]

        return twratio, clrequired, feasibletw

    def twrequired(self, wingloading_pa, feasibleonly=True, map2sl=True):
        """Calculate the T/W required for t/o, trn, clm, crs, sec.

        This method integrates the full set of constraints and it gives the user a
        compact way of performing a full constraint analysis. If a specific constraint
        is required only, the individual methods can be called separately:
        :code:`twrequired_to` (take-off), :code:`twrequired_trn` (turn),
        :code:`twrequired_clm` (climb), :code:`twrequired_trn` (turn),
        :code:`twrequired_crs` (cruise), :code:`twrequired_sec` (service ceiling).

        **Parameters:**

        wingloading_pa
            float or array, list of wing-loading values in Pa.

        map2sl
            boolean, specifies if the result in the conditions specified, should be mapped
            to sea level equivalents. Optional, defaults to True.

        **Outputs:**

        twreq
            dictionary variable, wherein each entry contains vectors
            related to one of the constraints: :code:`twreq['take-off']`
            (T/W required for take-off), :code:`twreq['liftoffspeed_mps']`
            (liftoff speed in m/s), :code:`twreq['avspeed_mps']` (average
            speed of the take-off run, in m/s), :code:`twreq['turn']`
            (T/W required for the turn), :code:`twreq['turnfeasible']` (same as
            :code:`twreq['turn']`, but with *NaN* where the maximum lift
            coefficient is exceeded), :code:`twreq['turncl']` (lift
            coefficient required in the turn), :code:`twreq['climb']`
            (T/W required for climb), :code:`twreq['cruise']` (T/W required
            for cruise), :code:`twreq['servceil']` (T/W required for the
            service ceiling constraint), :code:`twreq['combined']` (the
            T/W required to meet all of the above).

        """

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        tw_to, liftoffspeed_mpstas, avspeed_mpstas = self.twrequired_to(wingloading_pa, map2sl)
        tw_trn, clrequired, feasibletw_trn = self.twrequired_trn(wingloading_pa, map2sl)
        tw_clm = self.twrequired_clm(wingloading_pa, map2sl)
        tw_crs = self.twrequired_crs(wingloading_pa, map2sl)
        tw_sec = self.twrequired_sec(wingloading_pa, map2sl)

        if feasibleonly:
            tw_combined = np.amax([tw_to, feasibletw_trn, tw_clm, tw_crs, tw_sec], 0)
        else:
            tw_combined = np.max([tw_to, tw_trn, tw_clm, tw_crs, tw_sec], 0)

        twreq = {
            'take-off': tw_to,
            'liftoffspeed_mpstas': liftoffspeed_mpstas,
            'avspeed_mpstas': avspeed_mpstas,
            'turn': tw_trn,
            'turnfeasible': feasibletw_trn,
            'turncl': clrequired,
            'climb': tw_clm,
            'cruise': tw_crs,
            'servceil': tw_sec,
            'combined': tw_combined}

        return twreq

    def powerrequired(self, wingloading_pa, tow_kg, feasibleonly=True, map2sl=True):
        """Calculate the power (in HP) required for t/o, trn, clm, crs, sec.

        **Parameters:**

        wingloading_pa
            float or array, list of wing-loading values in Pa.

        tow_kg
            float, maximum take-off weight of the aircraft.

        map2sl
            boolean, specifies if the result in the conditions specified, should be mapped
            to sea level equivalents. Optional, defaults to True.

        **Outputs:**

        preq_hp
            dictionary, power (in horsepower) required for the given wing loadings."""

        if self.etadefaultflag > 0:
            etamsg = str(self.etadefaultflag) + ' prop etas set to defaults.'
            warnings.warn(etamsg, RuntimeWarning)

        # The T/W should not be mapped to SL before the conversion to P/W.
        twreq = self.twrequired(wingloading_pa, feasibleonly, map2sl=False)

        # Take-off power required
        pw_to_wpn = tw2pw(twreq['take-off'], twreq['liftoffspeed_mpstas'], self.etaprop_to)
        pw_to_hpkg = co.wn2hpkg(pw_to_wpn)
        p_to_hp = pw_to_hpkg * tow_kg
        if map2sl:
            _, pcorr = self._propulsion_slcorr(self.designatm, twreq['liftoffspeed_mpstas'], self.rwyelevation_m)
            p_to_hp = p_to_hp / pcorr

        # Turn power required
        trnspeed_mpstas = co.kts2mps(self.turnspeed_ktas)
        # Feasible turn power
        feasiblepw_trn_wpn = tw2pw(twreq['turnfeasible'], trnspeed_mpstas, self.etaprop_turn)
        if np.all(np.isnan(feasiblepw_trn_wpn)):
            nanmsg = 'All turns are infeasible for the given load factor, speed, and wing loadings.'
            warnings.warn(nanmsg, RuntimeWarning)
        feasiblepw_trn_hpkg = co.wn2hpkg(feasiblepw_trn_wpn)
        feasiblep_trn_hp = feasiblepw_trn_hpkg * tow_kg
        # Turn power, feasible and infeasible
        pw_trn_wpn = tw2pw(twreq['turn'], trnspeed_mpstas, self.etaprop_turn)
        pw_trn_hpkg = co.wn2hpkg(pw_trn_wpn)
        p_trn_hp = pw_trn_hpkg * tow_kg
        if map2sl:
            _, pcorr = self._propulsion_slcorr(self.designatm, trnspeed_mpstas, self.turnalt_m)
            feasiblep_trn_hp = feasiblep_trn_hp / pcorr
            p_trn_hp = p_trn_hp / pcorr

        # Climb power
        # Conversion to TAS, IAS and EAS conflated, safe for typical prop speeds
        climbspeed_ktas = self.designatm.eas2tas(self.climbspeed_kias, self.climbalt_m)
        clmspeed_mpstas = co.kts2mps(climbspeed_ktas)
        pw_clm_wpn = tw2pw(twreq['climb'], clmspeed_mpstas, self.etaprop_climb)
        pw_clm_hpkg = co.wn2hpkg(pw_clm_wpn)
        p_clm_hp = pw_clm_hpkg * tow_kg
        if map2sl:
            _, pcorr = self._propulsion_slcorr(self.designatm, clmspeed_mpstas, self.climbalt_m)
            p_clm_hp = p_clm_hp / pcorr

        # Power for cruise
        crsspeed_mpstas = co.kts2mps(self.cruisespeed_ktas)
        pw_crs_wpn = tw2pw(twreq['cruise'], crsspeed_mpstas, self.etaprop_cruise)
        pw_crs_hpkg = co.wn2hpkg(pw_crs_wpn)
        p_crs_hp = pw_crs_hpkg * tow_kg
        if map2sl:
            _, pcorr = self._propulsion_slcorr(self.designatm, crsspeed_mpstas, self.cruisealt_m)
            p_crs_hp = p_crs_hp / pcorr

        # Power for service ceiling
        # Conversion to TAS, IAS and EAS conflated, safe for typical prop speeds
        secclmbspeed_ktas = self.designatm.eas2tas(self.secclimbspd_kias, self.servceil_m)
        secclmspeed_mpstas = co.kts2mps(secclmbspeed_ktas)
        pw_sec_wpn = tw2pw(twreq['servceil'], secclmspeed_mpstas, self.etaprop_sec)
        pw_sec_hpkg = co.wn2hpkg(pw_sec_wpn)
        p_sec_hp = pw_sec_hpkg * tow_kg
        if map2sl:
            _, pcorr = self._propulsion_slcorr(self.designatm, secclmspeed_mpstas, self.servceil_m)
            p_sec_hp = p_sec_hp / pcorr

        if feasibleonly:
            p_combined_hp = np.max([p_to_hp, feasiblep_trn_hp, p_clm_hp, p_crs_hp, p_sec_hp], 0)
        else:
            p_combined_hp = np.max([p_to_hp, p_trn_hp, p_clm_hp, p_crs_hp, p_sec_hp], 0)

        preq_hp = {
            'take-off': p_to_hp,
            'liftoffspeed_mpstas': twreq['liftoffspeed_mpstas'],
            'avspeed_mpstas': twreq['avspeed_mpstas'],
            'turn': p_trn_hp,
            'turnfeasible': feasiblep_trn_hp,
            'turncl': twreq['turncl'],
            'climb': p_clm_hp,
            'cruise': p_crs_hp,
            'servceil': p_sec_hp,
            'combined': p_combined_hp}

        return preq_hp

    def propulsionsensitivity_monothetic(self, wingloading_pa, y_var='tw', y_lim=None, x_var='ws_pa', customlabels=None,
                                         show=True, maskbool=False, textsize=None, figsize_in=None):
        """Constraint analysis in the wing loading (or wing area) - T/W ratio (or power) space.
        The method generates a plot of the combined constraint diagram, with optional sensitivity
        diagrams for individual constraints. These are based on a One-Factor-at-a-Time analysis
        of the local sensitivities of the constraints (required T/W or power values) with respect
        to the variables that define the aircraft concept. The sensitivity charts show the
        relative proportions of these local sensitivities, referred to as 'relative sensitivities'.
        Sensitivities are computed for those inputs that are specified as a range (a [min, max] list)
        instead of a single scalar value and the sensitivity is estimated across this range, with the
        midpoint taken as the nominal value (see more details in `this notebook <https://github.com/sobester/ADRpy/blob/master/docs/ADRpy/notebooks/Constraint%20analysis%20of%20a%20single%20engine%20piston%20prop.ipynb>`_).

        Sensitivities can be computed with respect to components of the design brief, as well as
        aerodynamic parameter estimates or geometrical parameters.

        The example below can serve as a template for setting up a sensitivity study; further
        examples can be found in the notebook.

        This is a higher level wrapper of :code:`twrequired` - please consult its documentation
        entry for details on the individual constraints and their required inputs.

        **Parameters:**

        wingloading_pa
            array, list of wing loading values in Pa.

        y_var
            string, specifies the quantity to be plotted along the y-axis of the combined
            constraint diagram. Set to 'tw' for dimensionless thrust-to-weight required, or 'p_hp'
            for the power required (in horsepower); sea level standard day values in both cases.
            Optional, defaults to 'tw'.

        y_lim
            float, used to define the plot y-limit. Optional, defaults to 105% of the maximum
            value across all constraint curves.

        x_var
            string, specifies the quantity to be plotted along the x-axis of the combined
            constraint diagram. Set to 'ws_pa' for wing loading in Pa, or 's_m2' for wing area
            in metres squared. Optional, defaults to 'ws_pa'.

        customlabels
            dictionary, used to remap design definition parameter keys to labels better suited
            for plot labelling. Optional, defaults to None. See example below for usage.

        show
            boolean/string, used to indicate the type of plot required. Available arguments:
            :code:`True`, :code:`False`, 'combined', 'climb', 'cruise', 'servceil',
            'take-off', and 'turn'. Optional, defaults to True. 'combined' will generate
            the classic combined constraint diagram on its own.

        maskbool
            boolean, used to indicate whether or not constraints that do not affect the
            combined minimum propulsion sizing requirement should be obscured. Optional,
            defaults to False.

        textsize
            integer, sets a representative reference fontsize for the text on the plots.
            Optional, defaults to 10 for multi-subplot figures, and to 14 for singles.

        figsize_in
            list, used to specify custom dimensions of the output plot in inches. Image width
            must be specified as a float in the first entry of a two-item list, with height as
            the second item. Optional, defaults to 14.1 inches wide by 10 inches tall.

        **See also** ``twrequired``

        **Notes**

        1. This is a plotting routine that wraps the various constraint models implemented in ADRpy.
        If specific constraint data is required, use :code:`twrequired`.

        2. Investigating sensitivities of design parameters embedded within the aircraft concept
        definition dictionaries, such as weight fractions or propeller efficiencies for various
        constraints, is not currently supported. Similarly, it uses the atmosphere provided in
        the class argument 'designatm'; the computation of sensitivities with respect to
        atmosphere choice is not supported.

        3. The sensitivities are computed numerically.

        **Example** ::

            import numpy as np

            from ADRpy import atmospheres as at
            from ADRpy import constraintanalysis as ca

            designbrief = {'rwyelevation_m': 0, 'groundrun_m': 313,
                            'stloadfactor': [1.5, 1.65], 'turnalt_m': [1000, 1075], 'turnspeed_ktas': [100, 110],
                            'climbalt_m': 0, 'climbspeed_kias': 101, 'climbrate_fpm': 1398,
                            'cruisealt_m': [2900, 3200], 'cruisespeed_ktas': [170, 175], 'cruisethrustfact': 1.0,
                            'servceil_m': [6500, 6650], 'secclimbspd_kias': 92,
                            'vstallclean_kcas': 69}
            designdefinition = {'aspectratio': [10, 11], 'sweep_le_deg': 2, 'sweep_25_deg': 0, 'bpr': -1,
                                'wingarea_m2': 13.46, 'weight_n': 15000,
                                'weightfractions': {'turn': 1.0, 'climb': 1.0, 'cruise': 0.853, 'servceil': 1.0}}
            designperformance = {'CDTO': 0.0414, 'CLTO': 0.59, 'CLmaxTO': 1.69, 'CLmaxclean': 1.45, 'mu_R': 0.02,
                                 'CDminclean': [0.0254, 0.026], 'etaprop': {'take-off': 0.65, 'climb': 0.8,
                                                                            'cruise': 0.85, 'turn': 0.85,
                                                                            'servceil': 0.8}}

            wingloadinglist_pa = np.arange(700, 2500, 5)
            customlabelling = {'aspectratio': 'AR',
                               'sweep_le_deg': '$\\Lambda_{LE}$',
                               'sweep_mt_deg': '$\\Lambda_{MT}$'}

            atm = at.Atmosphere()
            concept = ca.AircraftConcept(designbrief, designdefinition, designperformance, atm)

            concept.propulsionsensitivity_monothetic(wingloading_pa=wingloadinglist_pa, y_var='p_hp', x_var='s_m2',
                                                 customlabels=customlabelling)

        """

        y_types_list = ['tw', 'p_hp']
        if y_var not in y_types_list:
            argmsg = 'Unsupported y-axis variable specified "{0}", using default "tw".'.format(str(y_var))
            warnings.warn(argmsg, RuntimeWarning)
            y_var = 'tw'

        if y_lim:
            if (type(y_lim) == float) or (type(y_lim) == int):
                pass
            else:
                argmsg = 'Unsupported plot y-limit specified "{0}", using default.'.format(str(y_lim))
                warnings.warn(argmsg, RuntimeWarning)
                y_lim = None

        x_types_list = ['ws_pa', 's_m2']
        if x_var not in x_types_list:
            argmsg = 'Unsupported x-axis variable specified "{0}", using default "ws_pa".'.format(str(x_var))
            warnings.warn(argmsg, RuntimeWarning)
            x_var = 'ws_pa'

        if customlabels is None:
            # There is no need to throw a warning if the following method arguments are left unspecified.
            customlabels = {}

        if textsize is None:
            if show is True:
                textsize = 10
            else:
                textsize = 14

        default_figsize_in = [14.1, 10]
        if figsize_in is None:
            figsize_in = default_figsize_in
        elif type(figsize_in) == list:
            if len(figsize_in) != 2:
                argmsg = 'Unsupported figure size, should be length 2, found {0} instead - using default parameters.' \
                    .format(len(figsize_in))
                warnings.warn(argmsg, RuntimeWarning)
                figsize_in = default_figsize_in

        if self.weight_n is False:
            defmsg = 'Maximum take-off weight was not specified in the aircraft design definitions dictionary.'
            raise ValueError(defmsg)

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        # Colour/alpha dictionary
        style = {
            'focusmask': {'colour': 'white', 'alpha': 0.70},
            'inv_soln': {'colour': 'crimson', 'alpha': 0.10}
        }
        # Pick a colour (red)
        # Step clockwise on the colour wheel and go darker (darkviolet)
        # Use the complementary colour and go lighter (yellowgreen)
        # Step clockwise on the colour wheel and go darker (olive)
        # Use the complementary colour and go lighter (mediumslateblue)
        # etc... until you end up at the start colour. Bright colours on even index, dark colours on odd
        clr_list = ['limegreen', 'olivedrab', 'darkorchid', 'indigo', 'yellow', 'darkgoldenrod',
                    'royalblue', 'darkslategrey', 'orange', 'sienna', 'darkturquoise', 'forestgreen',
                    'red', 'darkviolet', 'yellowgreen', 'olive', 'mediumslateblue', 'navy',
                    'gold', 'chocolate', 'dodgerblue', 'teal', 'lightcoral', 'darkred']
        clr_dict = {}

        # Potential design space and nominal design state, design dictionaries stored in lists
        designspace_list = self.designspace
        designstate_list = self.designstate
        designatmosphere = self.designatm
        mass_kg = self.weight_n / constants.g

        # If a design can take a range of values, its value is bounded by the max and min value items of the list
        sensitivityplots_list = ['climb', 'cruise', 'servceil', 'take-off', 'turn']
        propulsionreqprime = dict(zip(sensitivityplots_list, [{} for _ in range(len(sensitivityplots_list))]))

        # It's probably not necessary to have these all as functions since they are static - needs future optimisation
        def y_function(aircraft_object, y_type):
            if y_type == 'p_hp':  # If horsepower is to be plotted on the y-axis
                propulsionrequirement = aircraft_object.powerrequired(wingloading_pa=wingloading_pa, tow_kg=mass_kg)
            else:  # else default to T/W plotting on the y-axis
                propulsionrequirement = aircraft_object.twrequired(wingloading_pa=wingloading_pa)
            return propulsionrequirement

        def x_function(x_type):
            if x_type == 's_m2':  # If wing area is to be plotted on the x-axis
                plt_x_axis = self.weight_n / wingloading_pa
            else:  # else default to W/S plotting on the x-axis
                plt_x_axis = wingloading_pa
            return plt_x_axis

        def y_labelling(y_type):
            if y_type == 'p_hp':  # Horsepower is to be plotted on the y-axis
                ylabel = 'Power Required [hp]'
            else:  # Else default to T/W plotting on the y-axis
                ylabel = 'Thrust-to-Weight [-]'
            return ylabel

        def x_labelling(x_type):
            if x_type == 's_m2':  # If wing-area is to be plotted on the x-axis
                xlabel = 'Wing Area [m$^2$]'
            else:  # Else default to W/S plotting on the x-axis
                xlabel = 'Wing Loading [Pa]'
            return xlabel

        def wherecleanstall(x_type):
            if x_type == 's_m2':  # If wing-area is to be plotted on the x-axis
                x_stall = self.smincleanstall_m2(mass_kg)
            else:  # Else default to W/S plotting on the x-axis
                x_stall = self.wsmaxcleanstall_pa()
            return x_stall

        # Perform OFAT monothetic analysis
        for dictionary_i in range(len(designspace_list)):

            for _, (dp_k, dp_v) in enumerate(designspace_list[dictionary_i].items()):

                # If a list was found, create two temporary dictionaries with the maximum and minimum bounded values
                if type(dp_v) == list:

                    # Create copies of the nominal design state, amending the key of interest with the list extremes
                    temp_designstatemax = deepcopy(designstate_list)
                    temp_designstatemax[dictionary_i][dp_k] = max(dp_v)

                    temp_designstatemin = deepcopy(designstate_list)
                    temp_designstatemin[dictionary_i][dp_k] = min(dp_v)

                    # Evaluate the dictionaries as aircraft concepts
                    briefmax, briefmin = temp_designstatemax[0], temp_designstatemin[0]
                    designdefmax, designdefmin = temp_designstatemax[1], temp_designstatemin[1]
                    performancemax, performancemin = temp_designstatemax[2], temp_designstatemin[2]

                    # Evaluate T/W or P
                    acmax = AircraftConcept(briefmax, designdefmax, performancemax, designatmosphere, self.propulsion)
                    acmin = AircraftConcept(briefmin, designdefmin, performancemin, designatmosphere, self.propulsion)
                    propulsionreqmax = y_function(aircraft_object=acmax, y_type=y_var)
                    propulsionreqmin = y_function(aircraft_object=acmin, y_type=y_var)

                    # If after evaluating OFAT there was a change in T/W for a constraint, record magnitude of the range
                    for constraint in sensitivityplots_list:
                        propulsionreq_range = abs(propulsionreqmax[constraint] - propulsionreqmin[constraint])
                        # If the range is non-zero at any point, then the OFAT parameter had an impact on the constraint
                        if propulsionreq_range.all() != np.zeros(len(propulsionreq_range)).all():
                            propulsionreqprime[constraint].update({dp_k: propulsionreq_range})
                            # If parameter impacting the constraint was not previously recorded, assign a unique colour
                            if dp_k not in clr_dict:
                                clr_dict.update({dp_k: clr_list[len(clr_dict)]})

        # Produce data for the combined constraint
        brief, designdef, performance = designstate_list[0], designstate_list[1], designstate_list[2]
        acmed = AircraftConcept(brief, designdef, performance, designatmosphere, self.propulsion)
        propulsionreqmed = y_function(aircraft_object=acmed, y_type=y_var)
        # Refactor the x-axis
        x_axis = x_function(x_type=x_var)

        # If a stall constraint exists, create plot data
        if self.vstallclean_kcas and self.clmaxclean:  # If the stall condition is available to plot
            xcrit_stall = wherecleanstall(x_type=x_var)
        else:
            xcrit_stall = None

        # Determine the upper y-limit of the plots
        if y_lim is None:  # If the user did not specify a y_limit using the y_lim argument
            ylim_hi = []
            for sensitivityplot in sensitivityplots_list:
                ylim_hi.append(max(propulsionreqmed[sensitivityplot]))
            ylim_hi = max(ylim_hi) * 1.05
        else:
            ylim_hi = y_lim

        # Find the indices on the x-axis, where the propulsion constraint is feasible (sustained turn does not stall)
        propfeasindex = np.where(np.isfinite(propulsionreqmed['combined']))[0]
        # (Set of x-axis indices) - (set of feasible x-axis indices) = (set of infeasible x-axis indices)
        infeas_x_axis = list(set(x_axis) - set(x_axis[propfeasindex[0: -1]]))
        # x-values that bound the feasible propulsion region
        x2_infeascl = max(infeas_x_axis)
        x1_infeascl = min(infeas_x_axis)

        # GRAPH PLOTTING

        predefinedlabels = {'climb': "Climb", 'cruise': "Cruise", 'servceil': "Service Ceiling",
                            'take-off': "Take-off Ground Roll", 'turn': "Sustained Turn"}

        fontsize_title = 1.20 * textsize
        fontsize_label = 1.05 * textsize
        fontsize_legnd = 1.00 * textsize
        fontsize_tick = 0.90 * textsize

        def sensitivityplots(whichconstraint, ax_sens=None):
            if ax_sens is None:
                ax_sens = plt.gca()

            # Find the sum of all derivatives for a constraint, at every given wing-loading
            primesum = np.zeros(len(wingloading_pa))
            for _, (param_k, param_v) in enumerate(propulsionreqprime[whichconstraint].items()):
                primesum += param_v

            # Find the proportions of unity each parameter contributes to a constraint, and arrange as a list of arrays
            stackplot = []
            parameters_list = []
            keyclrs_list = []
            for _, (param_k, param_v) in enumerate(propulsionreqprime[whichconstraint].items()):
                stackplot.append(param_v / primesum)
                # Use custom labels if they exist
                if param_k in customlabels:
                    parameters_list.append(customlabels[param_k])
                else:
                    parameters_list.append(param_k)
                # Assign a key in the stackplot, its unique colour
                keyclrs_list.append(clr_dict[param_k])

            if len(stackplot) == 0:  # If no parameters could be added, populate stackplot with an empty filler
                stackplot.append([0.] * len(wingloading_pa))
                parameters_list.append("N/A")
                # Also draw a nice red 'x' to clearly identify the graph
                ax_sens.plot([min(x_axis), max(x_axis)], [0, 1], ls='-', color='r')
                ax_sens.plot([min(x_axis), max(x_axis)], [1, 0], ls='-', color='r')
                # The colour list should also be populated with a dummy colour
                keyclrs_list.append('red')
            else:
                pass

            # For the constraint being processed, generate stacked plot from the list of arrays
            ax_sens.stackplot(x_function(x_type=x_var), stackplot, labels=parameters_list, colors=keyclrs_list)
            ax_sens.set_title(predefinedlabels[whichconstraint], size=fontsize_title)
            ax_sens.set_xlim(min(x_axis), max(x_axis))
            ax_sens.set_ylim(0, 1)
            ax_sens.set_xlabel(xlabel=x_labelling(x_type=x_var), fontsize=fontsize_label)
            ax_sens.set_ylabel(ylabel=('Rel. Sensitivity of ' + y_var.split('_')[0].upper()), fontsize=fontsize_label)
            ax_sens.tick_params(axis='x', labelsize=fontsize_tick)
            ax_sens.tick_params(axis='y', labelsize=fontsize_tick)
            # The legend list must be reversed to make sure the legend displays in the same order the plot is stacked
            handles, labels = ax_sens.get_legend_handles_labels()
            ax_sens.legend(reversed(handles), reversed(labels), title='Design Variables', loc='center left',
                           bbox_to_anchor=(1, 0.5), prop={'size': fontsize_legnd}, title_fontsize=fontsize_legnd)

            if maskbool:
                # For sensitivity plots, focus user attention with transparent masks
                propreqcomb = np.nan_to_num(propulsionreqmed['combined'], copy=True)
                maskindex = np.where(propulsionreqmed[whichconstraint] < propreqcomb)[0]  # Mask where constraint < comb

                # Find discontinuities in maskindex, since we want to mask wherever maskindex is counting consecutively
                masksindex_list = np.split(maskindex, np.where(np.diff(maskindex) != 1)[0] + 1)

                for consecregion_index in range(len(masksindex_list)):
                    x2_clmask = x_axis[max(masksindex_list[consecregion_index])]
                    x1_clmask = x_axis[min(masksindex_list[consecregion_index])]

                    # If the cl mask is at the max feasible index, then draw from the min index to the max of the x-axis
                    if x2_clmask == x_axis[max(propfeasindex)]:
                        x2_clmask = x_axis[-1]

                    ax_sens.fill([x1_clmask, x2_clmask, x2_clmask, x1_clmask], [0, 0, 1, 1],
                                 color=style['focusmask']['colour'], alpha=style['focusmask']['alpha'])

            return None

        def combinedplot(ax_comb=None):
            if ax_comb is None:
                ax_comb = plt.gca()

            ax_comb.plot(x_axis, propulsionreqmed['combined'], lw=3.5, color='k', label="Feasible Turn $C_L$")
            # Aggregate the propulsion constraints onto the combined diagram
            for item in sensitivityplots_list:
                ax_comb.plot(x_axis, propulsionreqmed[item], label=predefinedlabels[item], lw=2.0, ls='--',
                             color=clr_list[sensitivityplots_list.index(item) * 2 + 6])

            # If the code could figure out where the clean stall takes place, plot it
            if xcrit_stall:
                if min(x_axis) < xcrit_stall < max(x_axis):
                    ax_comb.plot([xcrit_stall, xcrit_stall], [0, ylim_hi], label="$V_{stall}$ SL")

            # If the code could figure out where the turn stall takes place, plot it
            if len(propfeasindex) > 0:
                xturn_stall = x_axis[propfeasindex[-1]]
                if min(x_axis) < xturn_stall < max(x_axis):
                    ax_comb.plot([xturn_stall, xturn_stall], [0, ylim_hi], label="Sus. Turn stall")

            ax_comb.set_title('Aggregated Propulsion Constraints', size=fontsize_title)
            ax_comb.set_xlim(min(x_axis), max(x_axis))
            ax_comb.set_ylim(0, ylim_hi)
            ax_comb.set_xlabel(xlabel=x_labelling(x_type=x_var), fontsize=fontsize_label)
            ax_comb.set_ylabel(ylabel=y_labelling(y_type=y_var), fontsize=fontsize_label)
            ax_comb.tick_params(axis='x', labelsize=fontsize_tick)
            ax_comb.tick_params(axis='y', labelsize=fontsize_tick)
            ax_comb.legend(title='Constraints', loc='center left', bbox_to_anchor=(1, 0.5),
                           prop={'size': fontsize_legnd}, title_fontsize=fontsize_legnd)
            ax_comb.grid(True)

            # For the combined plot, obscure region of unattainable performance due to infeasible CL requirements
            if xcrit_stall:  # If the stall constraint is given, the CL mask should account for turn/stall constraints
                x2_clmask = max(xcrit_stall, x2_infeascl)
                x1_clmask = min(xcrit_stall, x1_infeascl)
            else:  # Else the CL mask should only evaluate the turn constraint CL
                x2_clmask = x2_infeascl
                x1_clmask = x1_infeascl
            # If a non-zero-thickness area was found in the "combined" plot region for which cl is invalid, mask it
            if x2_infeascl != x1_infeascl:
                ax_comb.fill([x1_clmask, x2_clmask, x2_clmask, x1_clmask], [0, 0, ylim_hi, ylim_hi],
                             color=style['inv_soln']['colour'], alpha=style['inv_soln']['alpha'])

            # Produce coordinates that describe the lower bound of the feasible region
            solnfeasindex = np.append(np.where(x2_clmask < x_axis)[0], np.where(x1_clmask > x_axis)[0])
            # Obscure the remaining region in which the minimum combined constraint is not satisfied
            if len(solnfeasindex) < 1:
                solnfeasindex = [0]
            x_inv = np.append(x_axis[solnfeasindex], [x_axis[solnfeasindex][-1], x_axis[solnfeasindex][0]])
            y_inv = np.append(propulsionreqmed['combined'][solnfeasindex], [0, 0])
            ax_comb.fill(x_inv, y_inv, color=style['inv_soln']['colour'], alpha=style['inv_soln']['alpha'])
            xfeas, yfeas = x_inv[0:-2], y_inv[0:-2]  # Map infeasible region coords, to a lower bound for feas T/W
            return xfeas, yfeas

        # Show the plot if specified to do so by method argument, then clear the plot and figure
        fig = False
        plots_list = ['climb', 'cruise', 'servceil', 'take-off', 'turn', 'combined']
        suptitle = {'t': "OFAT Sensitivity of Propulsion System Constraints (" + y_labelling(y_type=y_var) + ")",
                    'size': textsize * 1.4}

        if show is True:
            # Plotting setup, arrangement of 6 windows
            fig, axs = plt.subplots(3, 2, figsize=figsize_in,
                                    gridspec_kw={'hspace': 0.4, 'wspace': 0.8}, sharex='all')
            fig.canvas.set_window_title('ADRpy constraintanalysis.py')
            fig.subplots_adjust(left=0.1, bottom=None, right=0.82, top=None, wspace=None, hspace=None)
            fig.suptitle(suptitle['t'], size=suptitle['size'])

            axs_dict = dict(zip(plots_list, [axs[0, 0], axs[1, 0], axs[2, 0], axs[0, 1], axs[1, 1], axs[2, 1]]))

            # Plot INDIVIDUAL constraint sensitivity diagrams
            for sensitivityplottype in sensitivityplots_list:
                sensitivityplots(whichconstraint=sensitivityplottype, ax_sens=axs_dict[sensitivityplottype])
            # Plot COMBINED constraint diagram
            combinedplot(ax_comb=axs_dict['combined'])

        elif show in plots_list:
            # Plotting setup, single window
            fig, ax = plt.subplots(1, 1, figsize=figsize_in,
                                   gridspec_kw={'hspace': 0.4, 'wspace': 0.8}, sharex='all')
            fig.canvas.set_window_title('ADRpy constraintanalysis.py')
            fig.subplots_adjust(left=0.1, bottom=None, right=0.78, top=None, wspace=None, hspace=None)

            if show in sensitivityplots_list:
                # Plot INDIVIDUAL constraint sensitivity diagram
                fig.suptitle(suptitle['t'], size=suptitle['size'])
                sensitivityplots(whichconstraint=show, ax_sens=ax)
            else:
                # Plot COMBINED constraint diagram
                fig.suptitle("Combined View of Propulsion System Requirements", size=suptitle['size'])
                combinedplot(ax_comb=ax)

        if show:
            plt.show()
            plt.close(fig=fig)

        return None

    def vstall_kias(self, wingloading_pa, clmax):
        """Calculates the stall speed (indicated) for a given wing loading
        in a specified cofiguration.

        **Parameters:**

        wingloading_pa
            float or array, list of wing-loading values in Pa.

        clmax
            maximum lift coefficient (float) or the name of a standard configuration
            (string) for which a maximum lift coefficient was specified in the
            :code:`performance` dictionary (currently implemented: 'take-off').

        **Outputs:**

        vs_keas
            float or array, stall speed in knots.

        **Note**

        The calculation is performed assuming standard day ISA sea level
        conditions (not in the conditions specified in the atmosphere used
        when instantiating the :code:`AircraftConcept` object!) so the
        speed returned is an indicated (IAS) / calibrated (CAS) value.

        **Example**::

            from ADRpy import constraintanalysis as ca

            designperformance = {'CLmaxTO':1.6}

            concept = ca.AircraftConcept({}, {}, designperformance, {})

            wingloading_pa = 3500

            print("VS_to:", concept.vstall_kias(wingloading_pa, 'take-off'))

        """

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        isa = at.Atmosphere()
        rho0_kgpm3 = isa.airdens_kgpm3()

        if clmax == 'take-off':
            clmax = self.clmaxto

        vs_mps = math.sqrt(2 * wingloading_pa / (rho0_kgpm3 * clmax))

        return co.mps2kts(vs_mps)

    def wsmaxcleanstall_pa(self):
        """Maximum wing loading defined by the clean stall Clmax"""

        # (W/S)_max = q_vstall * CLmaxclean

        if self.clmaxclean is False:
            clmaxmsg = 'CLmaxclean must be specified in the performance dictionary.'
            raise ValueError(clmaxmsg)

        if self.vstallclean_kcas is False:
            vstallmsg = 'Clean stall speed must be specified in the design brief dictionary.'
            raise ValueError(vstallmsg)

        # We do the q calculation at SL conditions, TAS ~= EAS ~= CAS
        # (conflating CAS and EAS on the basis that the stall Mach number is likely v small)
        stallspeed_mpstas = co.kts2mps(self.vstallclean_kcas)

        q_pa = self.designatm.dynamicpressure_pa(stallspeed_mpstas, 0)
        return q_pa * self.clmaxclean

    def smincleanstall_m2(self, weight_kg):
        """Minimum wing area defined by the clean stall CLmax and the weight"""

        wsmax = self.wsmaxcleanstall_pa()
        return co.kg2n(weight_kg) / wsmax

    def map2static(self):
        """Maps the average take-off thrust to static thrust. If a bypass ratio
        is not specified, it returns a value of 1.
        """
        if self.bpr > 1:
            return (4 / 3) * (4 + self.bpr) / (5 + self.bpr)

        return 1.0

    def wigfactor(self):
        """Wing-in-ground-effect factor to account for the change in
        induced drag as a result of the wing being in close proximity
        of the ground. Specify the entry `wingheightratio` in the
        `design` dictionary variable you instantiated the `AircraftConcept`
        object with in order to compute this - if unspecified, a call to this
        method will result in a value practically equal to 1 being returned.

        The factor, following McCormick ("Aerodynamics, Aeronautics, and Flight
        Mechanics", Wiley, 1979) and Gudmundsson (2013) is calculated as:

        .. math::

            \\Phi = \\frac{(16\\,h/b)^2}{1+(16\\,h/b)^2}

        where :math:`h/b` is `design['wingheightratio']`: the ratio of the height
        of the wing above the ground (when the aircraft is on the runway) and the
        span of the main wing.

        The induced drag coefficient adjusted for ground effect thus becomes:

        .. math::

            C_\\mathrm{Di} = \\Phi C_\\mathrm{Di}^\\mathrm{oge},

        where the 'oge' superscript denotes the 'out of ground effect' value.

        **Example**
        ::

            import math
            from ADRpy import constraintanalysis as co

            designdef = {'aspectratio':8}
            wingarea_m2 = 10
            wingspan_m = math.sqrt(designdef['aspectratio'] * wingarea_m2)

            for wingheight_m in [0.6, 0.8, 1.0]:

                designdef['wingheightratio'] = wingheight_m / wingspan_m

                aircraft = co.AircraftConcept({}, designdef, {}, {})

                print('h/b: ', designdef['wingheightratio'],
                      'Phi: ', aircraft.wigfactor())

        Output::

            h/b:  0.06708203932499368  Phi:  0.5353159851301115
            h/b:  0.08944271909999159  Phi:  0.6719160104986877
            h/b:  0.11180339887498948  Phi:  0.761904761904762

        """
        return _wig(self.wingheightratio)


def _wig(h_over_b):
    return ((16 * h_over_b) ** 2) / (1 + (16 * h_over_b) ** 2)


def tw2pw(thrusttoweight, speed, etap):
    """Converts thrust to weight to power to weight (propeller-driven aircraft)

    **Parameters:**

    thrusttoweight
        thrust to weight ratio (non-dimensional)

    speed
        ground speed (in m/s if output in Watts / Newton is required)

    etap
        propeller efficiency (non-dimensional), float

    **Outputs:**

        power to weight ratio (in W/N if speed is in m/s)

    **See also** ``powerrequired``

    **Notes**

    1. A note on units. If the input speed is in m/s, the other two inputs being
    non-dimensional, the output product is also in m/s, which is equal to W/N
    (W / N = (J/s) / N = (Nm/s) / N = m/s).

    2. The speed input is a kinematic quantity, not an airspeed, so it is generally
    a ground speed (GS) or a true airspeed (TAS) if we are assuming zero wind.

    3. The inputs to the function are scalars or a mix of scalars and `numpy`
    arrays.

    **Example**::

        from ADRpy import constraintanalysis as ca
        from ADRpy import atmospheres as at
        from ADRpy import unitconversions as co

        designbrief = {'stloadfactor': 2, 'turnalt_m': 3050, 'turnspeed_ktas': 140}

        etap = {'turn': 0.85}

        designperformance = {'CLmaxclean': 1.45, 'CDminclean': 0.02541,
                                'etaprop': etap}

        designdef = {'aspectratio': 10, 'sweep_le_deg': 2,
                        'sweep_mt_deg': 0, 'bpr': -1}

        TOW_kg = 1500

        designatm = at.Atmosphere()
        concept = ca.AircraftConcept(designbrief, designdef,
                                        designperformance, designatm)

        wingloading_pa = 1000

        twreq, _, _ = concept.twrequired_trn(wingloading_pa)

        turnspeed_mpstas = co.kts2mps(designbrief['turnspeed_ktas'])

        pw_trn_wpn = ca.tw2pw(twreq, turnspeed_mpstas, etap['turn'])
        pw_trn_hpkg = co.wn2hpkg(pw_trn_wpn)
        p_trn_hp = pw_trn_hpkg * TOW_kg

        print(p_trn_hp)

    Output::

        318.691213406

    """
    return thrusttoweight * speed / etap

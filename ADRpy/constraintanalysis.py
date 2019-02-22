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

# pylint: disable=locally-disabled, too-many-instance-attributes
# pylint: disable=locally-disabled, too-many-branches
# pylint: disable=locally-disabled, too-many-statements
# pylint: disable=locally-disabled, too-many-locals
# pylint: disable=locally-disabled, too-many-lines

import math
import warnings
from scipy import constants
import numpy as np
from ADRpy import atmospheres as at
from ADRpy import unitconversions as co
from ADRpy import mtools4acdc as actools


class AircraftConcept:
    """Definition of a basic aircraft concept. An aircraft concept object must
    be defined for all constraint analysis functions.
    """

    def __init__(self, brief, design, performance, designatm):

        # Assign a default, if needed, to the atmosphere
        if not designatm:
            designatm = at.Atmosphere()
        self.designatm = designatm

        # Unpick the design brief dictionary first:

        if 'groundrun_m' in brief:
            self.groundrun_m = brief['groundrun_m']
        else:
            # Flag if not specified, error thrown by t/o constraint
            self.groundrun_m = -1

        if 'rwyelevation_m' in brief:
            self.rwyelevation_m = brief['rwyelevation_m']
        else:
            # Assign sea level, if not specified
            self.rwyelevation_m = 0

        if 'turnalt_m' in brief:
            self.turnalt_m = brief['turnalt_m']
        else:
            # Assign sea level, if not specified
            self.turnalt_m = 0

        if 'turnspeed_ktas' in brief:
            self.turnspeed_ktas = brief['turnspeed_ktas']
        else:
            # Flag if not specified, error thrown by turn constraint
            self.turnspeed_ktas = -1

        if 'stloadfactor' in brief:
            self.stloadfactor = brief['stloadfactor']
        else:
            # Flag if not specified, error thrown by climb constraint
            self.stloadfactor = -1

        if 'climbalt_m' in brief:
            self.climbalt_m = brief['climbalt_m']
        else:
            # Assign sea level, if not specified
            self.climbalt_m = 0

        if 'climbspeed_kias' in brief:
            self.climbspeed_kias = brief['climbspeed_kias']
        else:
            # Flag if not specified, error thrown by climb constraint
            self.climbspeed_kias = -1

        if 'climbrate_fpm' in brief:
            self.climbrate_fpm = brief['climbrate_fpm']
        else:
            # Flag if not specified, error thrown by climb constraint
            self.climbrate_fpm = -1

        if 'cruisealt_m' in brief:
            self.cruisealt_m = brief['cruisealt_m']
        else:
            # Flag if not specified, error thrown by cruise constraint
            self.cruisealt_m = -1

        if 'cruisespeed_ktas' in brief: # Option to specify Mach number instead coming soon
            self.cruisespeed_ktas = brief['cruisespeed_ktas']
        else:
            # Flag if not specified, error thrown by cruise constraint
            self.cruisespeed_ktas = -1

        if 'cruisethrustfact' in brief:
            self.cruisethrustfact = brief['cruisethrustfact']
        else:
            # Assume 100% throttle in cruise
            self.cruisethrustfact = 1.0

        if 'servceil_m' in brief:
            self.servceil_m = brief['servceil_m']
        else:
            # Flag if not specified, error thrown by cruise constraint
            self.servceil_m = -1

        if 'secclimbspd_kias' in brief:
            self.secclimbspd_kias = brief['secclimbspd_kias']
        else:
            # Flag if not specified, error thrown by cruise constraint
            self.secclimbspd_kias = -1

        if 'vstallclean_kcas' in brief:
            self.vstallclean_kcas = brief['vstallclean_kcas']
        else:
            # Flag if not specified, error thrown by cruise constraint
            self.vstallclean_kcas = -1


        # Unpick the design dictionary next:

        if 'aspectratio' in design:
            self.aspectratio = design['aspectratio']
        else:
            self.aspectratio = 8

        if 'bpr' in design:
            self.bpr = design['bpr']
        else:
            # Piston engine
            self.bpr = -1

        if 'tr' in design:
            self.throttle_r = design['tr']
        else:
            self.throttle_r = 1.07

        if 'sweep_le_deg' in design:
            self.sweep_le_deg = design['sweep_le_deg']
            self.sweep_le_rad = math.radians(self.sweep_le_deg)
        else:
            self.sweep_le_deg = 0
            self.sweep_le_rad = 0

        if 'sweep_mt_deg' in design:
            self.sweep_mt_deg = design['sweep_mt_deg']
            self.sweep_mt_rad = math.radians(self.sweep_mt_deg)
        else:
            self.sweep_mt_deg = self.sweep_le_deg
            self.sweep_mt_rad = self.sweep_le_rad

        if 'weightfractions' in design:
            if 'cruise' in design['weightfractions']:
                self.cruise_weight_fraction = design['weightfractions']['cruise']
            else:
                self.cruise_weight_fraction = 1.0
            if 'servceil' in design['weightfractions']:
                self.sec_weight_fraction = design['weightfractions']['servceil']
            else:
                self.sec_weight_fraction = 1.0
            if 'turn' in design['weightfractions']:
                self.turn_weight_fraction = design['weightfractions']['turn']
            else:
                self.turn_weight_fraction = 1.0
            if 'climb' in design['weightfractions']:
                self.climb_weight_fraction = design['weightfractions']['climb']
            else:
                self.climb_weight_fraction = 1.0
        else:
            # Assume all constraints at same weight (e.g., electrically powered a/c)
            self.cruise_weight_fraction = 1.0
            self.sec_weight_fraction = 1.0
            self.turn_weight_fraction = 1.0
            self.climb_weight_fraction = 1.0

        # Next, unpick the performance dictionary

        if 'CDTO' in performance:
            self.cdto = performance['CDTO']
        else:
            self.cdto = 0.09

        if 'CDminclean' in performance:
            self.cdminclean = performance['CDminclean']
        else:
            self.cdminclean = 0.03

        if 'mu_R' in performance:
            self.mu_r = performance['mu_R']
        else:
            self.mu_r = 0.03

        if 'CLTO' in performance:
            self.clto = performance['CLTO']
        else:
            self.clto = 0.95

        if 'CLmaxTO' in performance:
            self.clmaxto = performance['CLmaxTO']
        else:
            self.clmaxto = 1.5

        if 'etaprop' in performance:
            self.etaprop = performance['etaprop']
        else:
            self.etaprop = -1

        if 'CLmaxclean' in performance:
            self.clmaxclean = performance['CLmaxclean']
        else:
            self.clmaxclean = -1

        self.etadefaultflag = 0
        if 'etaprop' in performance:
            if 'take-off' in performance['etaprop']:
                self.etaprop_to = performance['etaprop']['take-off']
            else:
                self.etaprop_to = 0.45
                self.etadefaultflag += 1
            if 'cruise' in performance['etaprop']:
                self.etaprop_cruise = performance['etaprop']['cruise']
            else:
                self.etaprop_cruise = 0.85
                self.etadefaultflag += 1
            if 'servceil' in performance['etaprop']:
                self.etaprop_sec = performance['etaprop']['servceil']
            else:
                self.etaprop_sec = 0.65
                self.etadefaultflag += 1
            if 'turn' in performance['etaprop']:
                self.etaprop_turn = performance['etaprop']['turn']
            else:
                self.etaprop_turn = 0.85
                self.etadefaultflag += 1
            if 'climb' in performance['etaprop']:
                self.etaprop_climb = performance['etaprop']['climb']
            else:
                self.etaprop_climb = 0.75
                self.etadefaultflag += 1
        else:
            self.etaprop_to = 0.45
            self.etaprop_cruise = 0.85
            self.etaprop_sec = 0.65
            self.etaprop_climb = 0.75
            self.etaprop_turn = 0.85
            self.etadefaultflag = 5

    # Three different estimates the Oswald efficieny factor:

    def oswaldspaneff1(self):
        """Raymer's Oswald span efficiency estimate, sweep < 30, moderate AR"""
        return 1.78 * (1 - 0.045 * (self.aspectratio ** 0.68)) - 0.64

    def oswaldspaneff2(self):
        """Oswald span efficiency estimate due to Brandt et al."""
        sqrtterm = 4 + self.aspectratio ** 2 * (1 + (math.tan(self.sweep_mt_rad)) ** 2)
        return 2/(2 - self.aspectratio + math.sqrt(sqrtterm))

    def oswaldspaneff3(self):
        """Raymer's Oswald span efficiency estimate, swept wings"""
        return 4.61 * (1 - 0.045 * (self.aspectratio ** 0.68)) * \
        ((math.cos(self.sweep_le_rad)) ** 0.15) - 3.1


    def induceddragfact(self, whichoswald=1):
        """Lift induced drag factor k estimate (Cd = Cd0 + k.Cl^2)"""

        # k = 1 / pi.AR.e
        if whichoswald == 1:
            oswaldspaneff = self.oswaldspaneff1()
        elif whichoswald == 2:
            oswaldspaneff = self.oswaldspaneff2()
        elif whichoswald == 3:
            oswaldspaneff = self.oswaldspaneff3()
        elif whichoswald == 23:
            oswaldspaneff = 0.5 * (self.oswaldspaneff2() + self.oswaldspaneff3())
        elif whichoswald == 123:
            oswaldspaneff = (self.oswaldspaneff1() + self.oswaldspaneff2() + \
            self.oswaldspaneff3()) / 3.0
        return 1.0 / (math.pi * self.aspectratio * oswaldspaneff)


    def bestclimbspeedprop(self, wingloading_pa, altitude_m):
        """The best rate of climb speed for a propeller aircraft"""

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)
        dragfactor = np.sqrt(self.induceddragfact(123) / (3 * self.cdminclean))
        densfactor = 2 / self.designatm.airdens_kgpm3(altitude_m)

        # Gudmundsson, eq. (18-27)
        bestspeed_mps = np.sqrt(densfactor * wingloading_pa * dragfactor)

        if len(bestspeed_mps) == 1:
            return bestspeed_mps[0]

        return bestspeed_mps


    def thrusttoweight_takeoff(self, wingloading_pa):
        """The thrust to weight ratio required for take-off"""

        groundrun_m = self.groundrun_m

        # Assuming that the lift-off speed is equal to VR, which we estimate at 1.1VS1(T/O)
        density_kgpm3 = self.designatm.airdens_kgpm3(self.rwyelevation_m)

        vs1to_mps = np.sqrt((2 * wingloading_pa) / (density_kgpm3 * self.clmaxto))

        liftoffspeed_mps = 1.1 * vs1to_mps

        thrusttoweightreqd = (liftoffspeed_mps ** 2) / (2 * constants.g * groundrun_m) + \
        0.5 * self.cdto / self.clto + \
        0.5 * self.mu_r

        return thrusttoweightreqd, liftoffspeed_mps


    def thrusttoweight_sustainedturn(self, wingloading_pa):
        """Baseline T/W req'd for sustaining a given load factor at a certain altitude"""

        nturn = self.stloadfactor
        turnalt_m = self.turnalt_m
        turnspeed_mps = co.kts2mps(self.turnspeed_ktas)

        qturn = self.designatm.dynamicpressure_pa(airspeed_mps=turnspeed_mps, altitudes_m=turnalt_m)

        inddragfact = self.induceddragfact(whichoswald=123)

        cdmin = self.cdminclean

        twreqtrn = qturn * \
        (cdmin / wingloading_pa + inddragfact * ((nturn / qturn) ** 2) * wingloading_pa)

        # What cl is required to actually reach the target load factor
        clrequired = nturn * wingloading_pa / qturn

        return twreqtrn, clrequired


    def _altcorr(self, temp_c, pressure_pa, mach, density_kgpm3):
        """Altitude corrections, depending on propulsion system type"""
        if self.bpr == -1:
            twratio_altcorr = at.pistonpowerfactor(density_kgpm3)
        elif self.bpr == -2:
            twratio_altcorr = at.turbopropthrustfactor(temp_c, pressure_pa, mach, \
            self.throttle_r)
        elif self.bpr == -3: # no correction required
            twratio_altcorr = 1
        elif self.bpr == 0:
            twratio_altcorr = at.turbojetthrustfactor(temp_c, pressure_pa, mach, \
            self.throttle_r, False)
        elif self.bpr < 5:
            twratio_altcorr = at.turbofanthrustfactor(temp_c, pressure_pa, mach, \
            self.throttle_r, "lowbpr")
        else:
            twratio_altcorr = at.turbofanthrustfactor(temp_c, pressure_pa, mach, \
            self.throttle_r, "highbpr")
        return twratio_altcorr


    def twrequired_to(self, wingloading_pa):
        """Calculate the T/W required for take-off for a range of wing loadings

        **Parameters**

        wingloading_pa
            float or numpy array, list of wing loading values in Pa.

        **Returns**

        twratio
            array, thrust to weight ratio required for the given wing loadings.

        liftoffspeed_mps
            array, liftoff speeds (TAS) in m/s.

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

            tw_sl, liftoffspeed_mps, _ = concept.twrequired_to(wingloadinglist_pa)

            print(tw_sl)
            print(liftoffspeed_mps)

        Output: ::

            [ 0.19397876  0.26758006  0.33994772  0.41110154]
            [ 52.16511207  63.88895348  73.77260898  82.48028428]

        """
        if self.groundrun_m == -1:
            tomsg = "Ground run not specified in the designbrief dictionary."
            raise ValueError(tomsg)

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        twratio, liftoffspeed_mps = self.thrusttoweight_takeoff(wingloading_pa)

        # What does this required T/W mean in terms of static T/W required?
        twratio = self.map2static() * twratio

        # What SL T/W will yield the required T/W at the actual altitude?
        temp_c = self.designatm.airtemp_c(self.rwyelevation_m)
        pressure_pa = self.designatm.airpress_pa(self.rwyelevation_m)
        density_kgpm3 = self.designatm.airdens_kgpm3(self.rwyelevation_m)

        for i, los_mps in enumerate(liftoffspeed_mps):
            mach = self.designatm.mach(los_mps, self.rwyelevation_m)
            corr = self._altcorr(temp_c, pressure_pa, mach, density_kgpm3)
            twratio[i] = twratio[i] / corr

        avspeed_mps = liftoffspeed_mps / np.sqrt(2)

        if len(twratio) == 1:
            return twratio[0], liftoffspeed_mps[0], avspeed_mps[0]

        return twratio, liftoffspeed_mps, avspeed_mps


    def bank2turnradius(self, bankangle_deg):
        """Calculates the turn radius in m, given the turn TAS and the bank angle"""

        bankangle_rad = math.radians(bankangle_deg)
        v_mps = co.kts2mps(self.turnspeed_ktas)

        r_m = (v_mps ** 2) / (constants.g * math.tan(bankangle_rad))

        return r_m


    def twrequired_trn(self, wingloading_pa):
        """Calculates the T/W required for turning for a range of wing loadings

        **Parameters**

        wingloading_pa
            float or numpy array, list of wing loading values in Pa.

        **Returns**

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

        if self.turnspeed_ktas == -1:
            turnmsg = "Turn speed not specified in the designbrief dictionary."
            raise ValueError(turnmsg)

        if self.stloadfactor == -1:
            turnmsg = "Turn load factor not specified in the designbrief dictionary."
            raise ValueError(turnmsg)

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        # W/S at the start of the specified turn test may be less than MTOW/S
        wingloading_pa = wingloading_pa * self.turn_weight_fraction

        twratio, clrequired = self.thrusttoweight_sustainedturn(wingloading_pa)

        # What SL T/W will yield the required T/W at the actual altitude?
        temp_c = self.designatm.airtemp_c(self.turnalt_m)
        pressure_pa = self.designatm.airpress_pa(self.turnalt_m)
        density_kgpm3 = self.designatm.airdens_kgpm3(self.turnalt_m)
        turnspeed_mps = co.kts2mps(self.turnspeed_ktas)
        mach = self.designatm.mach(turnspeed_mps, self.turnalt_m)
        corr = self._altcorr(temp_c, pressure_pa, mach, density_kgpm3)

        twratio = twratio / corr

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


    def twrequired_clm(self, wingloading_pa):
        """Calculates the T/W required for climbing for a range of wing loadings.

        **Parameters**

        wingloading_pa
            float or numpy array, list of wing loading values in Pa.

        **Returns**

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
        a set of basic geometrical and aerodynamic performance parameters, compute the necessary
        T/W ratio to hold the specified climb rate.

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

        if self.climbspeed_kias == -1:
            turnmsg = "Climb speed not specified in the designbrief dictionary."
            raise ValueError(turnmsg)
        climbspeed_mpsias = co.kts2mps(self.climbspeed_kias)

        # Assuming that the climb rate is 'indicated'
        if self.climbrate_fpm == -1:
            turnmsg = "Climb rate not specified in the designbrief dictionary."
            raise ValueError(turnmsg)
        climbrate_mps = co.fpm2mps(self.climbrate_fpm)

        climbspeed_mpstas = self.designatm.eas2tas(climbspeed_mpsias, self.servceil_m)
        climbrate_mpstroc = self.designatm.eas2tas(climbrate_mps, self.servceil_m)

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        # W/S at the start of the specified climb segment may be less than MTOW/S
        wingloading_pa = wingloading_pa * self.climb_weight_fraction

        inddragfact = self.induceddragfact(whichoswald=123)
        qclimb_pa = self.designatm.dynamicpressure_pa(climbspeed_mpstas, self.climbalt_m)

        cos_sq_theta = (1 - (climbrate_mpstroc / climbspeed_mpstas) ** 2)

        # To be implemented, as 1 + (V/g)*(dV/dh)
        accel_fact = 1.0

        twratio = accel_fact * climbrate_mpstroc / climbspeed_mpstas + \
        (1 / wingloading_pa) * qclimb_pa * self.cdminclean + \
        (inddragfact / qclimb_pa) * wingloading_pa * cos_sq_theta

        # What SL T/W will yield the required T/W at the actual altitude?
        temp_c = self.designatm.airtemp_c(self.climbalt_m)
        pressure_pa = self.designatm.airpress_pa(self.climbalt_m)
        density_kgpm3 = self.designatm.airdens_kgpm3(self.climbalt_m)
        mach = self.designatm.mach(climbspeed_mpstas, self.climbalt_m)
        corr = self._altcorr(temp_c, pressure_pa, mach, density_kgpm3)

        twratio = twratio / corr

        # Map back to T/MTOW if climb start weight is less than MTOW
        twratio = twratio * self.climb_weight_fraction

        if len(twratio) == 1:
            return twratio[0]

        return twratio


    def twrequired_sec(self, wingloading_pa):
        """T/W required for a service ceiling for a range of wing loadings"""

        if self.servceil_m == -1:
            secmsg = "Climb rate not specified in the designbrief dictionary."
            raise ValueError(secmsg)

        if self.secclimbspd_kias == -1:
            secmsg = "Best climb speed not specified in the designbrief dictionary."
            raise ValueError(secmsg)

        secclimbspeed_mpsias = co.kts2mps(self.secclimbspd_kias)
        secclimbspeed_mpstas = self.designatm.eas2tas(secclimbspeed_mpsias, self.servceil_m)

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        # W/S at the start of the service ceiling test point may be less than MTOW/S
        wingloading_pa = wingloading_pa * self.sec_weight_fraction

        inddragfact = self.induceddragfact(whichoswald=123)
        qclimb_pa = self.designatm.dynamicpressure_pa(secclimbspeed_mpstas, self.servceil_m)

        # Service ceiling typically defined in terms of climb rate (at best climb speed) of
        # dropping to 100feet/min ~ 0.508m/s
        climbrate_mps = co.fpm2mps(100)

        # What true climb rate does 100 feet/minute correspond to?
        climbrate_mpstroc = self.designatm.eas2tas(climbrate_mps, self.servceil_m)

        twratio = climbrate_mpstroc / secclimbspeed_mpstas + \
        (1 / wingloading_pa) * qclimb_pa * self.cdminclean + \
        (inddragfact / qclimb_pa) * wingloading_pa

        # What SL T/W will yield the required T/W at the actual altitude?
        temp_c = self.designatm.airtemp_c(self.servceil_m)
        pressure_pa = self.designatm.airpress_pa(self.servceil_m)
        density_kgpm3 = self.designatm.airdens_kgpm3(self.servceil_m)
        mach = self.designatm.mach(secclimbspeed_mpstas, self.servceil_m)
        corr = self._altcorr(temp_c, pressure_pa, mach, density_kgpm3)

        twratio = twratio / corr

        # Map back to T/MTOW if service ceiling test start weight is less than MTOW
        twratio = twratio * self.sec_weight_fraction

        if len(twratio) == 1:
            return twratio[0]

        return twratio




    def twrequired_crs(self, wingloading_pa):
        """Calculate the T/W required for cruise for a range of wing loadings"""

        if self.cruisespeed_ktas == -1:
            cruisemsg = "Cruise speed not specified in the designbrief dictionary."
            raise ValueError(cruisemsg)
        cruisespeed_mps = co.kts2mps(self.cruisespeed_ktas)

        if self.cruisealt_m == -1:
            cruisemsg = "Cruise altitude not specified in the designbrief dictionary."
            raise ValueError(cruisemsg)

        wingloading_pa = actools.recastasnpfloatarray(wingloading_pa)

        # W/S at the start of the cruise may be less than MTOW/S
        wingloading_pa = wingloading_pa * self.cruise_weight_fraction

        inddragfact = self.induceddragfact(whichoswald=123)
        qcruise_pa = self.designatm.dynamicpressure_pa(cruisespeed_mps, self.cruisealt_m)

        twratio = (1 / wingloading_pa) * qcruise_pa * self.cdminclean + \
        (inddragfact / qcruise_pa) * wingloading_pa

        # What SL T/W will yield the required T/W at the actual altitude?
        temp_c = self.designatm.airtemp_c(self.cruisealt_m)
        pressure_pa = self.designatm.airpress_pa(self.cruisealt_m)
        density_kgpm3 = self.designatm.airdens_kgpm3(self.cruisealt_m)

        mach = self.designatm.mach(cruisespeed_mps, self.cruisealt_m)

        corr = self._altcorr(temp_c, pressure_pa, mach, density_kgpm3)

        twratio = twratio / corr

        # Map back to T/MTOW if cruise start weight is less than MTOW
        twratio = twratio * self.cruise_weight_fraction

        twratio = twratio * (1 / self.cruisethrustfact)

        if len(twratio) == 1:
            return twratio[0]

        return twratio


    def twrequired(self, wingloadinglist_pa, feasibleonly=True):
        """Calculate the T/W required for t/o, trn, clm, crs, sec.

        This method integrates the full set of constraints and it gives the user a
        compact way of performing a full constraint analysis.

        **Parameters**

        wingloading_pa
            float or numpy array, list of wing loading values in Pa.

        **Returns**

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

        tw_to, liftoffspeed_mps, avspeed_mps = self.twrequired_to(wingloadinglist_pa)
        tw_trn, clrequired, feasibletw_trn = self.twrequired_trn(wingloadinglist_pa)
        tw_clm = self.twrequired_clm(wingloadinglist_pa)
        tw_crs = self.twrequired_crs(wingloadinglist_pa)
        tw_sec = self.twrequired_sec(wingloadinglist_pa)

        if feasibleonly:
            tw_combined = np.amax([tw_to, feasibletw_trn, tw_clm, tw_crs, tw_sec], 0)
        else:
            tw_combined = np.max([tw_to, tw_trn, tw_clm, tw_crs, tw_sec], 0)

        twreq = {
            'take-off': tw_to,
            'liftoffspeed_mps': liftoffspeed_mps,
            'avspeed_mps': avspeed_mps,
            'turn': tw_trn,
            'turnfeasible': feasibletw_trn,
            'turncl': clrequired,
            'climb': tw_clm,
            'cruise': tw_crs,
            'servceil': tw_sec,
            'combined': tw_combined}

        return twreq


    def powerrequired(self, wingloadinglist_pa, tow_kg, feasibleonly=True):
        """Calculate the power required for t/o, trn, clm, crs, sec."""

        if self.etadefaultflag > 0:
            etamsg = str(self.etadefaultflag) + " prop etas set to defaults."
            warnings.warn(etamsg, RuntimeWarning)

        twreq = self.twrequired(wingloadinglist_pa, feasibleonly)

        # Take-off power required
        pw_to_wpn = tw2pw(twreq['take-off'], twreq['avspeed_mps'], self.etaprop_to)
        pw_to_hpkg = co.wn2hpkg(pw_to_wpn)
        p_to_hp = pw_to_hpkg * tow_kg

        # Turn power required
        trnspeed_mpstas = co.kts2mps(self.turnspeed_ktas)
        if feasibleonly:
            pw_trn_wpn = tw2pw(twreq['turnfeasible'], trnspeed_mpstas, self.etaprop_turn)
        else:
            pw_trn_wpn = tw2pw(twreq['turn'], trnspeed_mpstas, self.etaprop_turn)
        pw_trn_hpkg = co.wn2hpkg(pw_trn_wpn)
        p_trn_hp = pw_trn_hpkg * tow_kg

        # Climb power
        # Conversion to TAS, IAS and EAS conflated, safe for typical prop speeds
        climbspeed_ktas = self.designatm.eas2tas(self.climbspeed_kias, self.climbalt_m)
        clmspeed_mpstas = co.kts2mps(climbspeed_ktas)
        pw_clm_wpn = tw2pw(twreq['climb'], clmspeed_mpstas, self.etaprop_climb)
        pw_clm_hpkg = co.wn2hpkg(pw_clm_wpn)
        p_clm_hp = pw_clm_hpkg * tow_kg

        # Power for cruise
        crsspeed_mpstas = co.kts2mps(self.cruisespeed_ktas)
        pw_crs_wpn = tw2pw(twreq['cruise'], crsspeed_mpstas, self.etaprop_cruise)
        pw_crs_hpkg = co.wn2hpkg(pw_crs_wpn)
        p_crs_hp = pw_crs_hpkg * tow_kg

        # Power for service ceiling
        # Conversion to TAS, IAS and EAS conflated, safe for typical prop speeds
        secclmbspeed_ktas = self.designatm.eas2tas(self.secclimbspd_kias, self.servceil_m)
        secclmspeed_mpstas = co.kts2mps(secclmbspeed_ktas)
        pw_sec_wpn = tw2pw(twreq['servceil'], secclmspeed_mpstas, self.etaprop_sec)
        pw_sec_hpkg = co.wn2hpkg(pw_sec_wpn)
        p_sec_hp = pw_sec_hpkg * tow_kg

        p_combined_hp = np.amax([p_to_hp, p_trn_hp, p_clm_hp, p_crs_hp, p_sec_hp], 0)

        preq_hp = {
            'take-off': p_to_hp,
            'liftoffspeed_mps': twreq['liftoffspeed_mps'],
            'avspeed_mps': twreq['avspeed_mps'],
            'turn': p_trn_hp,
            'turncl': twreq['turncl'],
            'climb': p_clm_hp,
            'cruise': p_crs_hp,
            'servceil': p_sec_hp,
            'combined': p_combined_hp}

        return preq_hp


    def wsmaxcleanstall_pa(self):
        """Maximum wing loading defined by the clean stall Clmax"""

        # (W/S)_max = q_vstall * CLmaxclean

        if self.clmaxclean == -1:
            clmaxmsg = "CLmaxclean must be specified in the performance dictionary."
            raise ValueError(clmaxmsg)

        if self.vstallclean_kcas == -1:
            vstallmsg = "Clean stall speed must be specified in the design brief dictionary."
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
        """Maps the average take-off thrust to static thrust"""
        if self.bpr > 1:
            return (4 / 3) * (4 + self.bpr) / (5 + self.bpr)

        return 1.0


def tw2pw(thrusttoweight, speed, etap):
    """Converts thrust to weight to power to weight (propeller-driven aircraft)

    **Parameters**

    thrusttoweight
        thrust to weight ratio (non-dimensional)

    speed
        speed (in m/s if output in Watts / Newton is required)

    etap
        propeller efficiency (non-dimensional), float

    **Returns**

        power to weight ratio (in W/N if speed is in m/s)

    **See also** ``powerrequired``

    **Notes**

    A note on units. If the input speed is in m/s, the other two inputs being
    non-dimensional, the output product is also in m/s, which is equal to W/N
    (W / N = (J/s) / N = (Nm/s) / N = m/s).

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

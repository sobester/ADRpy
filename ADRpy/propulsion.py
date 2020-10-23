# -*- coding: utf-8 -*-
"""
Engine Decks
--------
Provides an engine deck based on engine data for the various known engines.
A list of these engines can be printed by the local_data function.
"""

import os
import warnings

import csv
import numpy.polynomial.polynomial as np_poly
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata

from ADRpy import atmospheres as at
from ADRpy import mtools4acdc as actools

__author__ = "Samuel Pearson"
# Other contributors: Yaseen Reza

def local_data(deck_type, printdata=False):
    """
    **Parameters:**

        deck_type
            string, for specifying the category of engine that data should be returned for.

        printdata
            boolean, specify True if to return data in the console in human-friendly reading form.
            Optional, defaults to False.

    **Returns**

        engine_data
            dictionary, of all the engines available under the engine category specified in the
            :code:`deck_type` argument.

    **Example**

    ::

        local_data("Jet", printdata=False)

    Output: ::

        {'ATF3-6A': {
                    'available_data': [
                        'ATF3-6A Sea level thrust polynomial.csv',
                        'ATF3-6A Thrust data.csv',
                        'ATF3-6A TSFC data.csv'
                        ],
                    'engine_notes':
                        ' Take-off/cruise Thrust',
                    'reference':
                        ' Oates, GC., "Aerothermodynamics of Gas Turbine and Rocket Propulsion", 3rd Edition, AIAA,
                         Fig 5.11 , 5.12, pp.129, https://app.knovel.com/web/toc.v/cid:kpAGTRPE01/'
                    },
         'F404-400': {'available_data': ... },
         ... }
    """
    # List of types available as an input.
    enginetypes_list = ["turboprop", "jet", "electric", "piston"]
    deck_type = deck_type.lower()
    if deck_type not in enginetypes_list:
        raise LookupError("Engine type specified not valid, included types are: " + ", ".join(enginetypes_list[:-1])
                          + " and " + enginetypes_list[-1])

    # Produce a nested list of available engines, notes on the data obtained, and relevant references
    enginecsvs_path = os.path.join(os.path.dirname(__file__), "data", "engine_data", deck_type.capitalize() + "_CSVs")
    availablecsvs_list = os.listdir(enginecsvs_path)
    with open(os.path.join(enginecsvs_path, '_' + deck_type + '_metadata.csv'), 'r', encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        engine_metadata_list = list(reader)

    engine_data = {}
    for engine_metadata in engine_metadata_list:
        # Decomposes the metadata
        engine_name, engine_notes, engine_reference = engine_metadata

        # Replaces Data names for turboprop.
        file_name_replace = {}
        if deck_type == "turboprop":
            file_name_replace = {"Thrust_data": "Core/hot thrust (N) versus Mach Number and altitude (m)",
                                 "Power_data": "Shaft power (W) versus Mach Number and altitude (m)",
                                 "BSFC_data": "BSFC (g/(kWh)) versus Mach Number and altitude (m)"}
        # Replaces Data names for jet.
        elif deck_type == "jet":
            file_name_replace = {"Thrust_SL_TO_data": "Take-off thrust (N) versus Mach number at SL",
                                 "Thrust_data": "Thrust (N) versus Mach number and altitude (m)",
                                 "TSFC_data": "TSFC (g/(kNs)) versus Mach number and engine thrust (N)"}
        # Replaces Data names for electric motors.
        elif deck_type == "electric":
            file_name_replace = {"Efficiency_data": "Efficiency versus Engine speed (RPM) and torque (Nm)"}
        # Replaces Data names for piston engines.
        elif deck_type == "piston":
            file_name_replace = {"Power_data": "Shaft power (W) versus Engine Speed (RPM) and altitude (m)",
                                 "BSFC_data": "BSFC (g/(kWh)) Engine Speed (RPM) and shaft power (W)"}

        # Data relevant to the engine being examined, is given in a list
        available_enginecsvs_list = [item for item in availablecsvs_list if engine_name in item]

        parsed_enginecsvs_list = []
        for _, (k, v) in enumerate(file_name_replace.items()):
            for enginecsv in available_enginecsvs_list:
                parsed_enginecsvs_list.append(v) if k in enginecsv else None

        available_data_parsed = [item.replace(".csv", "") for item in parsed_enginecsvs_list]

        # Checks to see if data is to be printed.
        if printdata is True:
            print(engine_name + " (" + deck_type.capitalize() + " Deck)")
            print("Ground-Truth Data Available:\n >", "\n > ".join(available_data_parsed))
            print("Engine Notes:\n", engine_notes if len(engine_notes) != 0 else '-')
            print("Reference:\n", engine_reference if len(engine_reference) != 0 else '-', "\n" * 2)

        engine_data[engine_name] = {'available_data': available_enginecsvs_list,
                                    'engine_notes': engine_notes,
                                    'reference': engine_reference}

    return engine_data


class TurbopropDeck:
    """An object of this class contains a model of the engine core/hot thrust, power, and fuel-economy
    decks for turboprop engines, provided the performance data is available to the module. This class is
    intended for use in conjunction with the class :code:`PropellerDeck`, so that shaft power developed
    may be evaluated as cold thrust.

    **Parameters**

    engine_name
        String. Identifier of a known engine. Engine names available to the class can be found
        using the function :code:`local_data("turboprop")`.
    """

    def __init__(self, engine_name):

        # Sets class-wide parameters
        self.engine = engine_name
        self.isa = at.Atmosphere()

        # Uses the _setup function to find the relevant data for the engine
        # and then returns pandas dataframes and polynomial types as required.
        folderpath = os.path.join(os.path.dirname(__file__), "data", "engine_data", "Turboprop_CSVs")
        data, self.data_available = _setup(engine_name, folderpath, ["thrust", "power"], ["bsfc"])
        # Creates thrust dataframe variable if data is available.
        if self.data_available[0][0] is True:
            # Assigns the dataframe to variable
            thrust_df = data[0][0]
            # For the thrust data, a numpy arrays containing the Mach number,
            # altitude and thrust data is created for later use.
            ias_thr_mps = pd.DataFrame.to_numpy(thrust_df["Mach Number"])
            alt_thr_m = pd.DataFrame.to_numpy(thrust_df["Altitude (m)"])
            thr_thr_n = pd.DataFrame.to_numpy(thrust_df["Thrust (N)"])
            # Thrust data list.
            self.thr_data = np.array([ias_thr_mps, alt_thr_m, thr_thr_n])
        if self.data_available[0][1] is True:
            # Assigns the dataframe to variable
            power_df = data[0][1]
            # For the power data, a numpy arrays containing the Mach number,
            # altitude and power data is created for later use.
            ias_pwr_mps = pd.DataFrame.to_numpy(power_df["Mach Number"])
            alt_pwr_m = pd.DataFrame.to_numpy(power_df["Altitude (m)"])
            pwr_pwr_w = pd.DataFrame.to_numpy(power_df["Power (W)"])
            # Power data list.
            self.pwr_data = np.array([ias_pwr_mps, alt_pwr_m, pwr_pwr_w])
        # Creates BSFC dataframe variable.
        if self.data_available[1][0] is True:
            # Assigns the dataframe to variable
            bsfc_df = data[1][0]
            # For the BSFC data, a numpy array containing the Mach number,
            # thrust and BSFC data is created for later use.
            mach_bsfc = pd.DataFrame.to_numpy(bsfc_df["Mach Number"])
            thr_bsfc_n = pd.DataFrame.to_numpy(bsfc_df["Altitude (m)"])
            bsfc_bsfc_gpknps = pd.DataFrame.to_numpy(bsfc_df["BSFC (g/(kWh))"])
            # BSFC data list.
            self.bsfc_data = np.array([mach_bsfc, thr_bsfc_n, bsfc_bsfc_gpknps])

    def hotthrust(self, mach, altitude_m):
        """Uses a cubic interpolation technique to produce a continuous, interpolated data set of
        core/hot thrust (N) with free-stream mach number and ISA altitude (m), given sample engine data.
        Shape of arrays 'mach' and 'altitude_m' should be the same, or unity.

        **Parameters**

        mach
            float or array, free-stream flight Mach number.

        altitude_m
            float or array, the ISA Altitude in metres.

        **Returns**

        thrust_n
            array, interpolated estimate of the engine core/hot thrust in Newtons.

        **Example**

        Given the flight Mach number and ISA altitude, the thrust in Newtons can be estimated.

        ::

            t56 = TurbopropDeck("T56-A")
            print('Thrust (N):', t56.thrust(0.6, 2000))

        Output: ::

            Thrust (N): 1198.0750676220805

        """
        # Checks to see if the data is available for this function.
        if self.data_available[0][0] is False:
            datamsg = "No Hot thrust data available. Call local_data(\"turboprop\") for a list of available data."
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Uses the _griddata_interpolate function to interpolate data.
        # thr_data[0] is the Mach Number, thr_data[1] is the altitude (m)
        # and thr_data[2] is thrust (N).
        hotthrust_n = _griddata_interpolate(self.thr_data[0], self.thr_data[1], self.thr_data[2], mach, altitude_m)
        return hotthrust_n

    def shaftpower(self, mach, altitude_m):
        """Uses a cubic interpolation technique to produce a continuous, interpolated data set of
        shaft power (W) with free-stream mach number and ISA altitude (m), given sample engine data.
        Shape of arrays 'mach' and 'altitude_m' should be the same, or unity.

        **Parameters**

        mach
            float or array, free-stream flight Mach number.

        altitude_m
            float or array, the ISA Altitude in metres.

        **Returns**

        shaftpower_w
            array, returns interpolated estimate of the engine shaft power in Watts.
        """

        # Checks to see if the data is available for this function.
        if self.data_available[0][1] is False:
            datamsg = 'No power data available. Call local_data("turboprop") for a list of available data.'
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Uses the _griddata_interpolate function to interpolate data.
        # pwr_data[0] is the mach, pwr_data[1] is the altitude (m)
        # and pwr_data[2] is engine shaft power (W).
        shaftpower_w = _griddata_interpolate(self.pwr_data[0], self.pwr_data[1], self.pwr_data[2], mach, altitude_m)
        return shaftpower_w

    def bsfc(self, mach, altitude_m):
        """Uses a cubic interpolation technique to produce a continuous, interpolated data set of
        Brake-Specific Fuel Consumption (BSFC) (g/kWh) with free-stream mach number and ISA altitude
        (m), given sample engine data. Shape of arrays 'mach' and 'altitude_m' should be the same, or unity.

        **Parameters**

        mach
            float or array, free-stream flight Mach number.

        altitude_m
            float or array, the ISA Altitude in metres.

        **Returns**

        bsfc_gpkwh
            array, interpolated estimate of the engine fuel economy in grams per kilowatt-hour.
        """
        # Checks to see if the data is available for this function.
        if self.data_available[1][0] is False:
            datamsg = 'No TSFC available. Call local_data("turboprop") for a list of available data.'
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Removes any nan values from input data. As in cases where Mach data
        # has been used instead of Mach number, the thrust data was used to
        # find the Mach data, this results in a few gaps where the thrust data
        # did not go far enough into the range of the Mach data.
        bsfc_data = self.bsfc_data[:, ~np.any(np.isnan(self.bsfc_data), axis=0)]
        # Uses the _griddata_interpolate function to interpolate data.
        # tsfc_
        bsfc_gpkwh = _griddata_interpolate(bsfc_data[0], bsfc_data[1], bsfc_data[2], mach, altitude_m)
        return bsfc_gpkwh

    def coldthrust(self, mach, altitude_m, prop_rpm, propeller_obj=None):
        """Use shaftpower and propeller efficiency to find the cold thrust from the propeller.
        Shape of arrays 'mach' and 'altitude_m' should be the same, or unity.

        **Parameters**

        mach
            float, free-stream flight Mach number.

        altitude_m
            float, the ISA Altitude in metres.

        prop_rpm
            float, the number of full revolutions the propeller shaft makes in a minute.

        propeller_obj
            ADRpy.propulsion.PropellerDeck, a fully defined propeller object. Optional, generates
            a potentially infeasible propeller, for performance prediction only.

        **Returns**

        thrust_n
            array, interpolated estimate of the fan bypass/cold thrust in Newtons.

        """
        mach = actools.recastasnpfloatarray(mach)
        altitude_m = actools.recastasnpfloatarray(altitude_m)
        power_w = self.shaftpower(mach, altitude_m)

        if propeller_obj is None:
            propeller_obj = PropellerDeck()
            propeller_obj.ansatzprop(mach, altitude_m, prop_rpm, power_w)
            argmsg = 'The propeller_obj argument was not passed, using Ansatz method in its place!'
            warnings.warn(argmsg, RuntimeWarning)
        else:
            if not isinstance(propeller_obj, PropellerDeck):
                typemsg = 'Incorrect type "{0}" specified for propeller_obj, must be an object of the PropellerDeck ' \
                          'class.'.format(str(type(propeller_obj)))
                raise TypeError(typemsg)

        etaprop = propeller_obj.efficiency(mach=mach, altitude_m=altitude_m, shaftpower_w=power_w, prop_rpm=prop_rpm)
        machstar_mps = self.isa.vsound_mps(altitudes_m=altitude_m)
        airspeed_mpstas = mach * machstar_mps

        thrust_n = etaprop * power_w / airspeed_mpstas

        return thrust_n

    def thrust(self, mach, altitude_m, prop_rpm, propeller_obj=None):
        """Combine the hot and cold thrust values, assuming both act in the same axis, to estimate
        the total thrust developed by the turboprop assembly. Shape of arrays 'mach' and 'altitude_m'
        should be the same, or unity.

        **Parameters**

        mach
            float, free-stream flight Mach number.

        altitude_m
            float, the ISA Altitude in metres.

        prop_rpm
            float, the number of full revolutions the propeller shaft makes in a minute.

        propeller_obj
            ADRpy.propulsion.PropellerDeck, a fully defined propeller object. Optional, generates
            a potentially infeasible propeller, for performance prediction only.

        **Returns**

        thrust_n
            array, interpolated estimate of the total engine thrust in Newtons."""

        mach = actools.recastasnpfloatarray(mach)
        altitude_m = actools.recastasnpfloatarray(altitude_m)

        hotthrust_n = self.hotthrust(mach=mach, altitude_m=altitude_m)
        coldthrust_n = self.coldthrust(mach=mach, altitude_m=altitude_m, prop_rpm=prop_rpm, propeller_obj=propeller_obj)
        thrust_n = hotthrust_n + coldthrust_n

        return thrust_n

    def sl_powercorr(self, mach, altitude_m):
        """Multiply by SL maximum available shaft power to find power at the specified conditions.

        This method returns the ratio of power available in the flight conditions specified, to
        the sea-level maximum available power produced by the engine. By non-dimensionalising the
        power with sea level conditions, the engine can effectively be scaled in performance for any
        power requirement, given a feasible flight Mach number and altitude. Shape of arrays 'mach'
        and 'altitude_m' should be the same, or unity.

        **Parameters**

        mach
            float or array, free-stream flight Mach number.

        altitude_m
            float or array, the ISA Altitude in metres.

        **Returns**

        powercorr
            array, interpolated estimate of the total engine power normalised to sea-level maximum power.
        """
        mach = actools.recastasnpfloatarray(mach)
        altitude_m = actools.recastasnpfloatarray(altitude_m)
        shaftpower_w = self.shaftpower(mach=mach, altitude_m=altitude_m)

        power_sl = max(self.pwr_data[2]) # The maximum possible engine power is assumed to occur at SL
        powercorr = shaftpower_w / power_sl

        return powercorr

    def tsfc(self, mach, altitude_m, prop_rpm, propeller_obj=None):
        """First uses a cubic interpolation technique to determine brake-specific fuel consumption from
        sample engine data, which is combined with propeller efficiency data to produce an estimate for
        the thrust-specific fuel consumption (TSFC) (g/kNs). Shape of arrays 'mach' and 'altitude_m'
        should be the same, or unity.

        **Parameters**

        mach
            float, free-stream flight Mach number.

        altitude_m
            float, the ISA Altitude in metres.

        prop_rpm
            float, the number of full revolutions the propeller shaft makes in a minute.

        propeller_obj
            ADRpy.propulsion.PropellerDeck, a fully defined propeller object. Optional, generates
            a potentially infeasible propeller, for performance prediction only.

        **Returns**

        tsfc_gpkns
            array, interpolated estimate of the engine fuel economy in grams per kilonewton-second.

        """

        mach = actools.recastasnpfloatarray(mach)
        altitude_m = actools.recastasnpfloatarray(altitude_m)
        power_w = self.shaftpower(mach, altitude_m)

        if propeller_obj is None:
            propeller_obj = PropellerDeck()
            propeller_obj.ansatzprop(mach, altitude_m, prop_rpm, power_w)
            argmsg = 'The propeller_obj argument was not passed, using Ansatz method in its place!'
            warnings.warn(argmsg, RuntimeWarning)
        else:
            if not isinstance(propeller_obj, PropellerDeck):
                typemsg = 'Incorrect type "{0}" specified for propeller_obj, must be an object of the PropellerDeck ' \
                          'class.'.format(str(type(propeller_obj)))
                raise TypeError(typemsg)

        etaprop = propeller_obj.efficiency(mach=mach, altitude_m=altitude_m, shaftpower_w=power_w, prop_rpm=prop_rpm)
        machstar_mps = self.isa.vsound_mps(altitudes_m=altitude_m)
        airspeed_mpstas = mach * machstar_mps

        # Grams of fuel burnt to produce energy equivalent to the power developed in the shaft for 1 hour
        bsfc_gpkwh = self.bsfc(mach=mach, altitude_m=altitude_m)
        # Less fuel is burned in a second, than in an hour
        bsfc_gpkws = bsfc_gpkwh / 3600
        # F = eta * P / V applies here
        tsfc_gpkns = bsfc_gpkws * airspeed_mpstas / etaprop

        return tsfc_gpkns

    def demoplot_thrust(self):
        """This function produces a colour-map of engine thrust against Mach number and ISA altitude.

        **Parameters**

        None

        **Returns**

        None
        """
        # Checks to see if the data is available for this plot.
        if self.data_available[0] is False:
            datamsg = 'This engine has no core/hot thrust data available and so it is not possible to plot this.'
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Finds minimum and maximum values for Mach and altitude (m) data
        min_x = min(self.thr_data[0])
        max_x = max(self.thr_data[0])
        min_y = min(self.thr_data[1])
        max_y = max(self.thr_data[1])
        # Labels the x and y axis; colour-bar and adds title
        x_label = "Mach Number"
        y_label = "Altitude (m)"
        z_label = "Core/hot Thrust (N)"
        plt_title = self.engine + " core/hot thrust (N) for Mach number and altitude (m)"
        # Applies demo plot function to produce plot.
        _demo_plot(min_x, max_x, min_y, max_y, self.hotthrust, x_label, y_label, z_label, plt_title)

    def demoplot_shaftpower(self):
        """This function produces a colour-map of engine shaft power against Mach number and ISA altitude.

        **Parameters**

        None

        **Returns**

        None
        """
        # Checks to see if the data is available for this plot.
        if self.data_available[0] is False:
            datamsg = "This engine has no power data available and so it is not possible to plot this."
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Finds minimum and maximum values for Mach and altitude (m) data
        min_x = min(self.pwr_data[0])
        max_x = max(self.pwr_data[0])
        min_y = min(self.pwr_data[1])
        max_y = max(self.pwr_data[1])
        # Labels the x and y axis; colour-bar and adds title
        x_label = "Mach Number"
        y_label = "Altitude (m)"
        z_label = "Power (W)"
        plt_title = self.engine + " shaft power (W) for Mach Number and altitude (m)"
        # Applies demo plot function to produce plot.
        _demo_plot(min_x, max_x, min_y, max_y, self.shaftpower, x_label, y_label, z_label, plt_title)

    def demoplot_bsfc(self):
        """This function produces a colour-map of engine brake-specific fuel consumption against
        Mach number and ISA altitude.

        **Parameters**

        None

        **Returns**

        None
        """
        # Checks to see if the data is available for this plot.
        if self.data_available[0] is False:
            datamsg = "This engine has no BSFC data available and so it is not possible to plot this."
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Finds minimum and maximum values for Mach and altitude (m) data
        min_x = min(self.bsfc_data[0])
        max_x = max(self.bsfc_data[0])
        min_y = min(self.bsfc_data[1])
        max_y = max(self.bsfc_data[1])
        # Labels the x and y axis; colour-bar and adds title
        x_label = "Mach Number"
        y_label = "Altitude (m)"
        z_label = "BSFC (g/(kWh))"
        plt_title = self.engine + " BSFC (g/(kWh)) for Mach Number and altitude (m)"
        # Applies demo plot function to produce plot.
        _demo_plot(min_x, max_x, min_y, max_y, self.bsfc, x_label, y_label, z_label, plt_title)


class PistonDeck:
    """An object of this class contains a model of the power, and fuel-economy decks for
    piston engines, provided the performance data is available to the module.

    **Parameters**

    engine_name
        String. Identifier of a known engine. Engine names available to the class can be found
        using the function :code:`local_data("piston")`.

    """

    def __init__(self, engine_name, ):

        # Sets class-wide parameters
        self.engine = engine_name
        self.isa = at.Atmosphere()

        # Uses the _setup function to find the relevant data for the engine
        # and then returns pandas dataframes and polynomial types as required.
        folderpath = os.path.join(os.path.dirname(__file__), "data", "engine_data", "Piston_CSVs")
        data, self.data_available = _setup(engine_name, folderpath, ["power"], ["bsfc", "bsfc_best_power"])
        # Creates thrust dataframe variable if data is available.
        if self.data_available[0][0] is True:
            # Assigns the dataframe to variable
            power_df = data[0][0]
            # For the thrust data, a numpy arrays containing the Mach number,
            # altitude and thrust data is created for later use.
            spd_pwr = pd.DataFrame.to_numpy(power_df["Speed (RPM)"])
            alt_pwr_m = pd.DataFrame.to_numpy(power_df["Altitude (m)"])
            pwr_pwr_n = pd.DataFrame.to_numpy(power_df["Power (W)"])
            # Power data list.
            self.pwr_data = np.array([spd_pwr, alt_pwr_m, pwr_pwr_n])
        # Creates BSFC dataframe variable.
        if self.data_available[1][0] is True:
            # Assigns the dataframe to variable
            bsfc_df = data[1][0]
            # For the TSFC data, a numpy array containing the Mach number,
            # thrust and TSFC data is created for later use.
            spd_bsfc = pd.DataFrame.to_numpy(bsfc_df["Speed (RPM)"])
            pwr_bsfc_w = pd.DataFrame.to_numpy(bsfc_df["Power (W)"])
            bsfc_bsfc_gpkwph = pd.DataFrame.to_numpy(bsfc_df["BSFC (g/(kWh))"])
            # BSFC for best power data list.
            self.bsfc_data = np.array([spd_bsfc, pwr_bsfc_w, bsfc_bsfc_gpkwph])
        if self.data_available[1][1] is True:
            # Assigns the dataframe to variable
            bsfc_pwr_df = data[1][1]
            # For the BSFC data, a numpy array containing the Mach number,
            # thrust and BSFC for best power data is created for later use.
            spd_bsfc_pwr = pd.DataFrame.to_numpy(bsfc_pwr_df["Speed (RPM)"])
            pwr_bsfc_pwr_w = pd.DataFrame.to_numpy(bsfc_pwr_df["Power (W)"])
            bsfc_bsfc_pwr_gpkwph = pd.DataFrame.to_numpy(bsfc_pwr_df["BSFC (g/(kWh))"])
            # TSFC data list.
            self.bsfc_pwr_data = np.array([spd_bsfc_pwr, pwr_bsfc_pwr_w, bsfc_bsfc_pwr_gpkwph])

    def shaftpower(self, shaft_rpm, altitude_m):
        """Uses a cubic interpolation technique to produce a continuous, interpolated data set of
        shaft power (W) with engine shaft RPM and ISA altitude (m), given sample engine data. Shape
        of arrays 'shaft_rpm' and 'altitude_m' should be the same, or unity.

        **Parameters**

        shaft_rpm
            float or array, the number of revolutions the engine shaft makes in a minute, units of RPM.

        altitude_m
            float or array, the ISA Altitude in metres.

        **Returns**

        shaftpower_w
            array, interpolated estimate of the engine shaft power in Watts.


        **Example**

        Given the engine RPM and ISA altitude, the thrust in Newtons can be estimated.

        ::

            io540 = PistonDeck("IO-540")
            print('Power (W):', io540.power(2000, 100))

        Output: ::

            Power (W): 135553.2409465

        """
        # Checks to see if the data is available for this function.
        if self.data_available[0][0] is False:
            datamsg = 'No power data available. Call local_data("piston") for a list of available data.'
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Uses the _griddata_interpolate function to interpolate data.
        # pwr_data[0] is the RPM data, pwr_data[1] is the altitude (m)
        # and pwr_data[2] is engine shaft power (W).
        shaftpower_w = _griddata_interpolate(self.pwr_data[0], self.pwr_data[1], self.pwr_data[2], shaft_rpm,
                                             altitude_m)
        return shaftpower_w

    def bsfc(self, shaft_rpm, shaftpower_w, best="power"):
        """Uses a cubic interpolation technique to produce a continuous, interpolated data set of
        Brake-Specific Fuel Consumption (BSFC) (g/kWh) with engine shaft RPM, shaft power (W), and
        operating mode (better power, or better fuel-economy) given sample engine data. In power mode,
        the maximum available engine power is greater, but the engine is not as fuel-efficient as the
        economy setting. Shape of arrays 'shaft_rpm' and 'shaftpower_w' should be the same, or unity.

        **Parameters**

        shaft_rpm
            float or array, the number of revolutions the engine shaft makes in a minute, units of RPM.

        shaftpower_w
            float or array, the engine shaft power in Watts.

        best
            string, takes arguments :code:`'economy'` and :code:`'power'`. If set to economy, returned
            bsfc is the best possible for the given RPM and power delivery when the engine is tuned for
            fuel economy. If set to power, return the same result but for an engine tuned to deliver
            a greater power range (typically a worse/greater fuel consumption). Optional, defaults to
            :code:`'power'`.

        **Returns**

        bsfc_gpkwh
            array, interpolated estimate of the engine fuel economy in grams per kilowatt-hour.

        **Example**

        Given the engine RPM, ISA altitude, and power setting, the brake-specific fuel consumption in
        grams per kilowatt-hour can be estimated. Note an RPM of 1900 is beyond the lower limit of
        RPM for this engine, and returns np.nan.

        ::

            io540 = PistonDeck("IO-540")
            print('BSFC (g/kWh):', io540.bsfc([1900, 2200, 2500], 100000, "economy"))

        Output: ::

            BSFC (g/kWh): [         nan 336.83398423 366.39597071]

        """
        if best == "economy":
            # Checks to see if the data is available for this function.
            if self.data_available[1][0] is False:
                datamsg = ("No BSFC available for best economy condition. Call local_data(\"piston\") for a list "
                           "of available data.")
                warnings.warn(datamsg, RuntimeWarning)
                return
            # Uses the _griddata_interpolate function to interpolate data.
            bsfc_gpkwh = _griddata_interpolate(self.bsfc_data[0], self.bsfc_data[1], self.bsfc_data[2], shaft_rpm,
                                               shaftpower_w)
        elif best == "power":
            # Checks to see if the data is available for this function.
            if self.data_available[1][1] is False:
                datamsg = ("No BSFC available for best power condition. Call local_data(\"piston\") for a list of "
                           "available data.")
                warnings.warn(datamsg, RuntimeWarning)
                return
            # Uses the _griddata_interpolate function to interpolate data.
            bsfc_gpkwh = _griddata_interpolate(self.bsfc_pwr_data[0], self.bsfc_pwr_data[1], self.bsfc_pwr_data[2],
                                               shaft_rpm, shaftpower_w)
        else:
            datamsg = 'Not able to match user input "{0}" with accepted inputs "economy" or "power".'.format(best)
            warnings.warn(datamsg, RuntimeWarning)
            return
        return bsfc_gpkwh

    def thrust(self, mach, altitude_m, prop_rpm, propeller_obj=None):
        """Combine the hot and cold thrust values, assuming both act in the same axis, to estimate
        the total thrust developed by the turboprop assembly. Shape of arrays 'shaftpower_w', 'mach',
        and 'altitude_m' should be the same, or unity.

        **Parameters**

        mach
            float, free-stream flight Mach number.

        altitude_m
            float, the ISA Altitude in metres.

        prop_rpm
            float, the number of full revolutions the propeller shaft makes in a minute.

        propeller_obj
            ADRpy.propulsion.PropellerDeck, a fully defined propeller object. Optional, generates
            a potentially infeasible propeller, for performance prediction only.

        **Returns**

        thrust_n
            array, interpolated estimate of the total engine thrust in Newtons."""

        mach = actools.recastasnpfloatarray(mach)
        altitude_m = actools.recastasnpfloatarray(altitude_m)
        # We assume a direct engine-to-propeller shaft, and so the propeller and shaft rpm is the same
        shaftpower_w = self.shaftpower(shaft_rpm=prop_rpm, altitude_m=altitude_m)

        if propeller_obj is None:
            propeller_obj = PropellerDeck()
            propeller_obj.ansatzprop(mach, altitude_m, prop_rpm, shaftpower_w)
            argmsg = 'The propeller_obj argument was not passed, using Ansatz method in its place!'
            warnings.warn(argmsg, RuntimeWarning)
        else:
            if not isinstance(propeller_obj, PropellerDeck):
                typemsg = 'Incorrect type "' + str(type(propeller_obj)) + \
                          '" specified for propeller_obj, must be an object of the PropellerDeck class.'
                raise TypeError(typemsg)

        etaprop = propeller_obj.efficiency(mach=mach, altitude_m=altitude_m, shaftpower_w=shaftpower_w, prop_rpm=prop_rpm)
        machstar_mps = self.isa.vsound_mps(altitudes_m=altitude_m)
        airspeed_mpstas = mach * machstar_mps

        thrust_n = etaprop * shaftpower_w / airspeed_mpstas

        return thrust_n

    def sl_powercorr(self, shaft_rpm, altitude_m):
        """Multiply by SL maximum available shaft power to find power at the specified conditions.

        This method returns the ratio of power available in the flight conditions specified, to
        the sea-level maximum available power produced by the engine. By non-dimensionalising the
        power with sea level conditions, the engine can effectively be scaled in performance for any
        power requirement, given a feasible engine speed and altitude. Shape of arrays 'shaft_rpm'
        and 'altitude_m' should be the same, or unity.

        **Parameters**

        shaft_rpm
            float or array, the number of revolutions the engine shaft makes in a minute, units of RPM.

        altitude_m
            float or array, the ISA Altitude in metres.

        **Returns**

        powercorr
            array, interpolated estimate of the total engine power normalised to sea-level maximum power.
        """
        shaft_rpm = actools.recastasnpfloatarray(shaft_rpm)
        altitude_m = actools.recastasnpfloatarray(altitude_m)
        shaftpower_w = self.shaftpower(shaft_rpm=shaft_rpm, altitude_m=altitude_m)

        power_sl = max(self.pwr_data[2])  # The maximum possible engine power is assumed to occur at SL
        powercorr = shaftpower_w / power_sl

        return powercorr

    def tsfc(self, mach, altitude_m, prop_rpm, propeller_obj=None, best='power'):
        """First uses a cubic interpolation technique to determine brake-specific fuel consumption from
        sample engine data, which is combined with propeller efficiency data to produce an estimate for
        the thrust-specific fuel consumption (TSFC) (g/kNs). Shape of arrays 'mach' and 'altitude_m'
        should be the same, or unity.

        **Parameters**

        mach
            float, free-stream flight Mach number.

        altitude_m
            float, the ISA Altitude in metres.

        prop_rpm
            float, the number of full revolutions the propeller shaft makes in a minute.

        propeller_obj
            ADRpy.propulsion.PropellerDeck, a fully defined propeller object. Optional, generates
            a potentially infeasible propeller, for performance prediction only.

        best
            string, takes arguments :code:`'economy'` and :code:`'power'`. If set to economy, returned
            bsfc is the best possible for the given RPM and power delivery when the engine is tuned for
            fuel economy. If set to power, return the same result but for an engine tuned to deliver
            a greater power range (typically a worse/greater fuel consumption). Optional, defaults to
            :code:`'power'`.

        **Returns**

        tsfc_gpkns
            array, interpolated estimate of the engine fuel economy in grams per kilonewton-second.

        """

        mach = actools.recastasnpfloatarray(mach)
        altitude_m = actools.recastasnpfloatarray(altitude_m)
        power_w = self.shaftpower(mach, altitude_m)

        if propeller_obj is None:
            propeller_obj = PropellerDeck()
            propeller_obj.ansatzprop(mach, altitude_m, prop_rpm, power_w)
            argmsg = 'The propeller_obj argument was not passed, using Ansatz method in its place!'
            warnings.warn(argmsg, RuntimeWarning)
        else:
            if not isinstance(propeller_obj, PropellerDeck):
                typemsg = 'Incorrect type "{0}" specified for propeller_obj, must be an object of the PropellerDeck ' \
                          'class.'.format(str(type(propeller_obj)))
                raise TypeError(typemsg)

        etaprop = propeller_obj.efficiency(mach=mach, altitude_m=altitude_m, shaftpower_w=power_w, prop_rpm=prop_rpm)
        machstar_mps = self.isa.vsound_mps(altitudes_m=altitude_m)
        airspeed_mpstas = mach * machstar_mps

        # Grams of fuel burnt to produce energy equivalent to the power developed in the shaft for 1 hour
        bsfc_gpkwh = self.bsfc(shaft_rpm=prop_rpm, shaftpower_w=power_w, best=best)
        # Less fuel is burned in a second, than in an hour
        bsfc_gpkws = bsfc_gpkwh / 3600
        # F = eta * P / V applies here
        tsfc_gpkns = bsfc_gpkws * airspeed_mpstas / etaprop

        return tsfc_gpkns

    def demoplot_shaftpower(self):
        """This function produces a colour-map of engine power against engine RPM and ISA altitude.

        **Parameters**

        None

        **Returns**

        None
        """
        # Checks to see if the data is available for this plot.
        if self.data_available[0] is False:
            datamsg = "This engine has no power data available and so it is not possible to plot this."
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Finds minimum and maximum values for engine speed (RPM) and
        # altitude (m) data
        min_x = min(self.pwr_data[0])
        max_x = max(self.pwr_data[0])
        min_y = min(self.pwr_data[1])
        max_y = max(self.pwr_data[1])
        # Labels the x and y axis; colour-bar and adds title
        x_label = "Engine Speed (RPM)"
        y_label = "Altitude (m)"
        z_label = "Shaft power (W)"
        plt_title = self.engine + " shaft power (W) for Engine speed (RPM) and altitude (m)"
        # Applies demo plot function to produce plot.
        _demo_plot(min_x, max_x, min_y, max_y, self.shaftpower, x_label, y_label, z_label, plt_title)

    def demoplot_bsfc(self, best="power"):
        """This function produces a colour-map of engine brake-specific fuel consumption (BSFC)
        against engine RPM and engine shaft power.

        **Parameters**

        best
            string, takes arguments :code:`'economy'` and :code:`'power'`. If set to economy, returned
            BSFC is the best possible for the given RPM and power delivery when the engine is tuned for
            fuel economy. If set to power, return the same result but for an engine tuned to deliver
            a greater power range (typically a worse/greater fuel consumption). Optional, defaults to
            :code:`'power'`.

        **Returns**

        None

        """

        # Labels the x and y axis; colour-bar and adds title
        x_label = "Engine Speed (RPM)"
        y_label = "Shaft power (W)"
        z_label = "BSFC (g/(kWh))"
        plt_title = self.engine + " BSFC (g/(kWh)) at best " + best + " for Engine speed (RPM) and shaft power (W)"
        # Checks to see which limits and errors are required
        if best == "economy":
            # Checks to see if the data is available for this plot.
            if self.data_available[1][0] is False:
                datamsg = "This engine has no best economy BSFC data available and so it is not possible to plot this."
                warnings.warn(datamsg, RuntimeWarning)
                return
            # Finds minimum and maximum values for Mach and altitude (m) data
            min_x = min(self.bsfc_data[0])
            max_x = max(self.bsfc_data[0])
            min_y = min(self.bsfc_data[1])
            max_y = max(self.bsfc_data[1])
        # Checks to see which limits to use
        elif best == "power":
            # Checks to see if the data is available for this plot.
            if self.data_available[1][1] is False:
                datamsg = "This engine has no best power BSFC data available and so it is not possible to plot this."
                warnings.warn(datamsg, RuntimeWarning)
                return
            # Finds minimum and maximum values for Mach and altitude (m) data.
            min_x = min(self.bsfc_pwr_data[0])
            max_x = max(self.bsfc_pwr_data[0])
            min_y = min(self.bsfc_pwr_data[1])
            max_y = max(self.bsfc_pwr_data[1])
        # If the best input has been incorrectly entered raise the same error
        # as for the bsfc function.
        else:
            datamsg = 'Not able to match user input: "{0}" with accepted inputs "economy" or "power".'.format(best)
            warnings.warn(datamsg, RuntimeWarning)
            return

        # Nested definition to set BSFC type.
        def bsfc_type(shaft_rpm, shaftpower_w):
            return self.bsfc(shaft_rpm, shaftpower_w, best)

        _demo_plot(min_x, max_x, min_y, max_y, bsfc_type, x_label, y_label, z_label, plt_title)


class ElectricDeck:
    """An object of this class contains a model of the efficiency decks for electric motor engines,
    provided the performance data is available to the module.

    **Parameters**

    engine_name
        String. Identifier of a known engine. Engine names available to the class can be found
        using the function :code:`local_data("electric")`.

    """

    def __init__(self, engine_name):

        # Sets class-wide parameters
        self.engine = engine_name
        self.isa = at.Atmosphere()

        # Uses the _setup function to find the relevant data for the engine
        # and then returns pandas dataframes and polynomial types as required.
        folderpath = os.path.join(os.path.dirname(__file__), "data", "engine_data", "Electric_CSVs")
        data, self.data_available = _setup(engine_name, folderpath, ["efficiency"], [""])
        # Creates Torque and power dataframe variable if data is available.
        if self.data_available[0][0] is True:
            # Assigns the dataframe to variable
            torque_df = data[0][0]
            # For the thrust data, a numpy arrays containing the Mach number,
            # altitude and thrust data is created for later use.
            spd_eta_rpm = pd.DataFrame.to_numpy(torque_df["Speed (RPM)"])
            trq_eta_nm = pd.DataFrame.to_numpy(torque_df["Torque (Nm)"])
            eta_eta = pd.DataFrame.to_numpy(torque_df["Efficiency"])
            # Finds shaft power by multiplying torque by angular frequency.
            sft_pwr_eta = trq_eta_nm * spd_eta_rpm * (2 * np.pi) / 60
            # Finds electrical power by finding shaft power by efficiency.
            elc_pwr_eta = sft_pwr_eta / eta_eta
            # Thrust data list.
            self.eta_data = np.array([spd_eta_rpm, trq_eta_nm, eta_eta, sft_pwr_eta, elc_pwr_eta])

    def efficiency(self, shaft_rpm, torque_nm):
        """Uses a cubic interpolation technique to produce a continuous, interpolated data set of
        electro-mechanical power conversion efficiency with motor RPM and torque, given sample
        electric engine data. Shape of arrays 'shaft_rpm' and 'torque_nm' should be the same, or unity.

        **Parameters**

        shaft_rpm
            float or array, the number of revolutions the motor shaft makes in a minute, units of RPM.

        torque_nm
            float or array, the torque of the motor shaft in Newton-metres.

        **Returns**

        efficiency
            array, interpolated estimate of the motor electro-mechanical power conversion efficiency.

        **Example**

        Given the motor RPM and torque, the electrical efficiency of the motor can be estimated.

        ::

            jmx57 = ElectricDeck("JMX57")
            print('Efficiency:', jmx57.efficiency([200, 600, 1000], 300))

        Output: ::

            Efficiency: [       nan 0.93630572 0.95183808]

        """
        # Checks to see if the data is available for this function.
        if self.data_available[0][0] is False:
            datamsg = 'No efficiency data available. Call local_data("electric") for a list of available data.'
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Uses the _griddata_interpolate function to interpolate data.
        # pwr_data[0] is the mach, pwr_data[1] is the altitude (m)
        # and pwr_data[2] is engine shaft power (W).
        efficiency = _griddata_interpolate(self.eta_data[0], self.eta_data[1], self.eta_data[2], shaft_rpm, torque_nm)
        return efficiency

    def bsec(self, shaft_rpm, torque_nm):
        """Uses a cubic interpolation technique to produce a continuous, interpolated data set of
        Brake-Specific Energy Consumption (BSEC), a non-dimensional product of Brake-Specific Fuel
        Consumpton (BSFC) of the engine, and the specific enthalpy of the energy source. This metric
        is analogous to the ratio of rate of electrical energy consumed (J/s) to mechanical shaft
        power developed (W), which is the inverse of motor efficiency. Shape of arrays 'shaft_rpm'
        and 'torque_nm' should be the same, or unity.

        **Parameters**

        shaft_rpm
            float or array, the number of revolutions the motor shaft makes in a minute, units of RPM.

        torque_nm
            float or array, the torque of the motor shaft in Newton-metres.

        **Returns**

        bsec
            array, interpolated estimate of the ratio of input power to output shaft power.

        """
        bsec = np.divide(1, self.efficiency(shaft_rpm=shaft_rpm, torque_nm=torque_nm))
        return bsec

    def shaftpower(self, shaft_rpm, torque_nm):
        """Shaftpower (W) of a motor is equal to the product of the rotational speed (rad/s) and the torque
        (Nm). Shape of arrays 'shaft_rpm' and 'torque_nm' should be the same, or unity.

        **Parameters**

        shaft_rpm
            float or array, the number of revolutions the motor shaft makes in a minute, units of RPM.

        torque_nm
            float or array, the torque of the motor shaft in Newton-metres.

        **Returns**

        shaftpower_w
            array, interpolated estimate of the engine shaft power in Watts.

        """
        shaft_rpm = actools.recastasnpfloatarray(shaft_rpm)
        torque_nm = actools.recastasnpfloatarray(torque_nm)

        speed_rps = shaft_rpm / 60
        speed_rads = speed_rps * np.pi
        shaftpower_w = np.multiply(speed_rads, torque_nm)

        return shaftpower_w

    def thrust(self, shaftpower_w, mach, altitude_m, prop_rpm, propeller_obj=None):
        """Combine the hot and cold thrust values, assuming both act in the same axis, to estimate
        the total thrust developed by the turboprop assembly. Shape of arrays 'shaftpower_w', 'mach',
        and 'altitude_m' should be the same, or unity.

        **Parameters**

        shaftpower_w
            float or array, the engine shaft power in Watts.

        mach
            float or array, free-stream flight Mach number.

        altitude_m
            float or array, the ISA Altitude in metres.

        prop_rpm
            float, the number of full revolutions the propeller shaft makes in a minute.

        propeller_obj
            ADRpy.propulsion.PropellerDeck, a fully defined propeller object. Optional, generates
            a potentially infeasible propeller, for performance prediction only.

        **Returns**

        thrust_n
            array, interpolated estimate of the total engine thrust in Newtons."""

        mach = actools.recastasnpfloatarray(mach)
        altitude_m = actools.recastasnpfloatarray(altitude_m)
        shaftpower_w = actools.recastasnpfloatarray(shaftpower_w)

        if propeller_obj is None:
            propeller_obj = PropellerDeck()
            propeller_obj.ansatzprop(mach, altitude_m, prop_rpm, shaftpower_w)
            argmsg = 'The propeller_obj argument was not passed, using Ansatz method in its place!'
            warnings.warn(argmsg, RuntimeWarning)
        else:
            if not isinstance(propeller_obj, PropellerDeck):
                typemsg = 'Incorrect type "{0}" specified for propeller_obj, must be an object of the PropellerDeck ' \
                          'class.'.format(str(type(propeller_obj)))
                raise TypeError(typemsg)

        etaprop = propeller_obj.efficiency(mach=mach, altitude_m=altitude_m, shaftpower_w=shaftpower_w,
                                           prop_rpm=prop_rpm)
        machstar_mps = self.isa.vsound_mps(altitudes_m=altitude_m)
        airspeed_mpstas = mach * machstar_mps

        thrust_n = etaprop * shaftpower_w / airspeed_mpstas

        return thrust_n


    def demoplot_efficiency(self):
        """This function produces a colour-map of motor efficiency against engine RPM and torque.

        **Parameters**

        None

        **Returns**

        None
        """
        # Checks to see if the data is available for this plot.
        if self.data_available[0][0] is False:
            datamsg = "This engine has no efficiency data available and so it is not possible to plot this."
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Finds minimum and maximum values for motor speed (RPM) and
        # torque (Nm)
        min_x = min(self.eta_data[0])
        max_x = max(self.eta_data[0])
        min_y = min(self.eta_data[1])
        max_y = max(self.eta_data[1])
        # Labels the x and y axis; colour-bar and adds title
        x_label = "Motor Speed (RPM)"
        y_label = "Torque (Nm)"
        z_label = "Efficiency"
        plt_title = self.engine + " efficiency for Motor speed (RPM) and torque (Nm)"
        # Applies demo plot function to produce plot.
        _demo_plot(min_x, max_x, min_y, max_y, self.efficiency, x_label, y_label, z_label, plt_title)


class JetDeck:
    """An object of this class contains a model of the thrust and fuel-economy decks for
    jet engines, provided the performance data is available to the module.

    **Parameters**

    engine_name
        String. Identifier of a known engine. Engine names available to the class can be found
        using the function :code:`local_data("jet")`.

    """

    def __init__(self, engine_name):

        # Sets class-wide parameters
        self.engine = engine_name

        # Uses the _setup function to find the relevant data for the engine
        # and then returns pandas dataframes and polynomial types as required.
        folderpath = os.path.join(os.path.dirname(__file__), "data", "engine_data", "Jet_CSVs")
        data, self.data_available = _setup(engine_name, folderpath, ["thrust"], ["tsfc"])
        # Creates thrust dataframe variable if data is available.
        if self.data_available[0][0] is True:
            # Assigns the dataframe to variable
            thrust_df = data[0][0]
            # For the thrust data, a numpy arrays containing the Mach number,
            # altitude and thrust data is created for later use.
            mach_thr = pd.DataFrame.to_numpy(thrust_df["Mach Number"])
            alt_thr_m = pd.DataFrame.to_numpy(thrust_df["Altitude (m)"])
            thr_thr_n = pd.DataFrame.to_numpy(thrust_df["Thrust (N)"])
            # Thrust data list.
            self.thr_data = np.array([mach_thr, alt_thr_m, thr_thr_n])
        # Creates TSFC dataframe variable.
        if self.data_available[1][0] is True:
            # Assigns the dataframe to variable
            tsfc_df = data[1][0]
            # For the TSFC data, a numpy array containing the Mach number,
            # thrust and TSFC data is created for later use.
            mach_tsfc = pd.DataFrame.to_numpy(tsfc_df["Mach Number"])
            thr_tsfc_n = pd.DataFrame.to_numpy(tsfc_df["Thrust (N)"])
            tsfc_tsfc_gpknps = pd.DataFrame.to_numpy(tsfc_df["TSFC (g/(kNs))"])
            # TSFC data list.
            self.tsfc_data = np.array([mach_tsfc, thr_tsfc_n, tsfc_tsfc_gpknps])
        # Sea level (SL) thrust (thr) polynomial (poly) data.
        self.sl_thr_poly = data[2]
        self.sl_poly_limits = data[3]
        # Sea level (SL) take off (TO) thrust (thr) polynomial (poly) data.
        self.sl_to_thr_poly = data[4]
        self.sl_to_poly_limits = data[5]

    def thrust(self, mach, altitude_m):
        """Uses a cubic interpolation technique to produce a continuous, interpolated data set of
        core/hot thrust (N) with free-stream mach number and ISA altitude (m), given sample engine data.
        Shape of arrays 'mach' and 'altitude_m' should be the same, or unity.

        **Parameters**

        mach
            float or array, free-stream flight Mach number.

        altitude_m
            float or array, the ISA Altitude in metres.

        **Returns**

        thrust_n
            array, interpolated estimate of the engine core/hot thrust in Newtons.

        **Example**

        Given the flight Mach number and ISA altitude, the thrust in Newtons can be estimated.

        ::

            jt8d9 = JetDeck("JT8D-9")
            print('Thrust (N):', jt8d9.thrust(0.5, 1000))

        Output: ::

            Thrust (N): 35766.25058041

        """
        # Checks to see if the data is available for this function.
        if self.data_available[0][0] is False:
            datamsg = "No Thrust data available. Call local_data(\"jet\") for a list of available data."
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Uses the _griddata_interpolate function to interpolate data.
        # thr_data[0] is the Mach number, thr_data[1] is the altitude (m)
        # and thr_data[2] is thrust (N).
        thrust_n = _griddata_interpolate(self.thr_data[0], self.thr_data[1], self.thr_data[2], mach, altitude_m)
        return thrust_n

    def sl_thrustcorr(self, mach, altitude_m):
        """Multiply by SL static thrust by this to get thrust at the specified conditions.

        This method returns the ratio of thrust available in the flight conditions specified, to
        the sea-level static thrust produced by the engine. By non-dimensionalising the thrust with
        sea level conditions, the engine can effectively be scaled in performance for any thrust
        requirement, given a feasible flight Mach number and altitude. Shape of arrays 'mach' and
        'altitude_m' should be the same, or unity.

        **Parameters**

        mach
            float, free-stream flight Mach number.

        altitude_m
            float, the ISA Altitude in metres.

        **Returns**

        thrustcorr
            array, interpolated estimate of the total engine thrust normalised to sea-level static thrust.
        """

        thrust_fl = self.thrust(mach=mach, altitude_m=altitude_m)
        # Not quite sea level, but try to find data from the lowest ISA altitude and mach number available
        thrust_sl = max(self.thr_data[2])
        thrustcorr = thrust_fl / thrust_sl

        return thrustcorr

    def tsfc(self, mach, thrust_n):
        """Uses a cubic interpolation technique to produce a continuous, interpolated data set of
        Thrust-Specific Fuel Consumption (TSFC) (g/kNs) with free-stream mach number and thrust (N),
        given sample engine data. Shape of arrays 'mach' and 'thrust_n' should be the same, or unity.

        **Parameters**

        mach
            float or array, free-stream flight Mach number.

        thrust_n
            float or array, the developed thrust in Newtons.

        **Returns**

        tsfc_gpkns
            array, interpolated estimate of the engine fuel economy in grams per kilonewton-second.

        **Example**

        Given the flight Mach number and thrust setting, the thrust-specific fuel consimption in
        grams per kilonewton-second can be estimated.

        ::

            jt8d9 = JetDeck("JT8D-9")
            print('TSFC (g/kNs):', jt8d9.tsfc(0.3, 30000))

        Output: ::

            TSFC (g/kNs): 18.71142251

        """
        # Checks to see if the data is available for this function.
        if self.data_available[1][0] is False:
            datamsg = 'No TSFC available. Call local_data("jet") for a list of available data.'
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Removes any nan values from input data. As in cases where TAS data
        # has been used instead of Mach number, the thrust data was used to
        # find the Mach data, this results in a few gaps where the thrust data
        # did not go far enough into the range of the TAS data.
        tsfc_data = self.tsfc_data[:, ~np.any(np.isnan(self.tsfc_data), axis=0)]
        # Uses the _griddata_interpolate function to interpolate data.
        tsfc_gpkns = _griddata_interpolate(tsfc_data[0], tsfc_data[1], tsfc_data[2], mach, thrust_n)
        return tsfc_gpkns

    def sl_thrust(self, mach):
        """Uses a 6th order polynomial to estimate sea level thrust.

        **Parameters**

        mach
            float, free-stream flight Mach number.

        **Returns**

        thrust_n
            float, interpolated estimate of engine thrust at sea level in Newtons.

        **Example**

        ::

            jt8d9 = JetDeck("JT8D-9")
            print('Thrust SL (N):', jt8d9.sl_thrust(0.5))

        Output: ::

            Thrust SL (N): 37412.617753750004
        """
        # Checks to see if the data is available for this function.
        if self.data_available[2] is False:
            datamsg = ("This engine has no sea level thrust data available. Call local_data(\"jet\") for a list "
                       "of available data.")
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Creates limits warning and suggestions
        high_warn = "Input Mach number too high, limit of provided data is Mach " + str(self.sl_poly_limits[-1]) + \
                    ". To resolve, use a lower Mach number."
        low_warn = "Input Mach number too low, limit of provided data is Mach " + str(self.sl_poly_limits[0]) + \
                   ". To resolve, use a higher Mach number."
        # Applies polynomial function
        thrust_n = _poly(mach, self.sl_thr_poly, self.sl_poly_limits, low_warn, high_warn)
        return thrust_n

    def sl_take_off_thrust(self, mach):
        """Uses a 6th order polynomial to estimate sea level thrust during take-off.

        **Parameters**

        mach
            float, free-stream flight Mach number.

        **Returns**

        thrust_n
            float, interpolated estimate of engine take off thrust at sea level in Newtons.

        **Example**

        ::

            jt8d9 = JetDeck("JT8D-9")
            print('Thrust SL_TO (N):', jt8d9.sl_take_off_thrust(0.4))

        Output: ::

            Thrust SL_TO (N): 53179.248666918036

        """
        # Checks to see if the data is available for this function.
        if self.data_available[3] is False:
            datamsg = ("No sea level take off thrust data available. Call local_data(\"jet\") for a list of "
                       "available data.")
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Creates limits warning and suggestions.
        high_warn = "Input Mach number too high, limit of provided data is Mach " + str(self.sl_to_poly_limits[-1]) + \
                    ". To resolve, use a lower Mach number."
        low_warn = "Input Mach number too low, limit of provided data is Mach " + str(self.sl_to_poly_limits[0]) + \
                   ". To resolve, use a higher Mach number."
        # Applies polynomial function
        thrust_n = _poly(mach, self.sl_to_thr_poly, self.sl_to_poly_limits, low_warn, high_warn)
        return thrust_n

    def demoplot_thrust(self):
        """This function produces a colour-map of engine thrust against Mach number and ISA altitude.

        **Parameters**

        None

        **Returns**

        None
        """
        # Checks to see if the data is available for this plot.
        if self.data_available[0][0] is False:
            datamsg = "This engine has no thrust data available and so it is not possible to plot this data."
            warnings.warn(datamsg, RuntimeWarning)
            return
        # Finds minimum and maximum values for mach and altitude (m) data
        min_x = min(self.thr_data[0])
        max_x = max(self.thr_data[0])
        min_y = min(self.thr_data[1])
        max_y = max(self.thr_data[1])
        # Labels the x and y axis; colour-bar and adds title
        x_label = "Mach Number"
        y_label = "Altitude (m)"
        z_label = "Thrust (N)"
        plt_title = self.engine + " thrust (N) for Mach number and altitude (m)"
        # Applies demo plot function to produce plot.
        _demo_plot(min_x, max_x, min_y, max_y, self.thrust, x_label, y_label, z_label, plt_title)

    def demoplot_tsfc(self, y_var="t_n"):
        """This function produces a colour-map of engine thrust-specific fuel consumption against
        Mach number and either ISA altitude, or thrust setting.

        **Parameters**

            y_var
                string. Used to indicate if the TSFC (g/(kNs) and Mach plot is against thrust,
                or altitude. Set to 't_n' for thrust, and 'alt_m' for altitude. Optional, defaults
                to 't_n'.

        **Returns**

            None

        """
        # Checks to see if TSFC data is available. If not then a warning
        # message is printed out and the function does not return anything.
        if self.data_available[1][0] is False:
            datamsg = "This engine has no TSFC data available and so it is not possible to plot this."
            warnings.warn(datamsg, RuntimeWarning)
            return

        if y_var == "t_n":
            # If it is then it checks to see if the Thrust data is available.
            # If not then a warning message is printed out and the function
            # does not return anything.
            if self.data_available[0][0] is False:
                datamsg = ("This engine has no thrust data available and so it is not possible to plot this. "
                           "Set y_var to 'alt_m' to see a graph of TSFC for Mach number and altitude.")
                warnings.warn(datamsg, RuntimeWarning)
                return
            # Finds minimum and maximum values for the Mach and Thrust (N)
            # data.
            min_x = min(self.tsfc_data[0])
            max_x = max(self.tsfc_data[0])
            min_y = min(self.tsfc_data[1])
            max_y = max(self.tsfc_data[1])
            # Labels the x and y axis; colour-bar and adds title
            x_label = "Mach Number"
            y_label = "Thrust (N)"
            z_label = "TSFC (g/(kNs))"
            plt_title = self.engine + " TSFC (g/(kNs)) for Mach number and Thrust (N)"
            # Applies demo plot function to produce plot.
            _demo_plot(min_x, max_x, min_y, max_y, self.tsfc, x_label, y_label, z_label, plt_title)

        elif y_var == "alt_m":

            # Finds minimum and maximum values for the Mach and altitude (m) data.
            min_x = min(self.thr_data[0])
            max_x = max(self.thr_data[0])
            min_y = min(self.thr_data[1])
            max_y = max(self.thr_data[1])
            # Labels the x and y axis; colour-bar and adds title
            x_label = "Mach Number"
            y_label = "Altitude (m)"
            z_label = "TSFC (g/(kNs))"
            plt_title = self.engine + " TSFC (g/(kNs)) for Mach number and altitude (m)"

            # Nested function to allow TSFC (g/(kNs)) to be expressed as a function
            # of Mach number and altitude (m).
            def _tsfc_fun(mach, altitude):
                return self.tsfc(mach, self.thrust(mach, altitude))

            # Applies demo plot function to produce plot. Using a lambda function
            # to provide the thrust for the tsfc function.
            _demo_plot(min_x, max_x, min_y, max_y, _tsfc_fun, x_label, y_label, z_label, plt_title)

        else:
            datamsg = ("Not able to match user input: \"" + y_var + "\" with accepted inputs \"t_n\" (Thrust) or "
                                                                    "\"alt_m\" (Altitude).")
            warnings.warn(datamsg, RuntimeWarning)

        return


class PropellerDeck:
    """An object of this class contains a model of the efficiency decks for propellers, with
    performance data being provided from semi-empirical models. A concept propeller's basic
    geometrical parameters may be defined, and promptly have its performance estimated by this
    class. This class does not yet support custom efficiency decks from csv data.

    **Parameters**

    propeller
        Dictionary. Definition of the basic parameters that will ultimately define a propeller's
        geometry. Contains the following key names:

        diameter_m
            Float. The diameter of the circle drawn by the tip of a propeller blade on a static bed.

        bladecount
            Integer. The number of blades stemming from the propeller hub.

        bladeactivityfact
            Float. Blade activity factor, a non-dimensional measure of the capacity of a propeller
            to absorb power. Heavily influenced by the radial distribution of chord along a blade.

        solidity
            Float. Ratio of propeller blade area to the area of the circle circumscribing the
            propeller.

        idesign_cl
            Float. The integrated design lift coefficient for a single blade.

    """

    def __init__(self, propeller=None):

        # Sets class-wide parameters
        if propeller is None:
            propeller = {}
        self.propeller = propeller
        self.isa = at.Atmosphere()

        # Custom propeller parameters
        if type(propeller) == dict:
            # Propeller geometry
            self.diameter_m = propeller['diameter_m'] if 'diameter_m' in propeller else False
            self.bladecount = propeller['bladecount'] if 'bladecount' in propeller else False
            self.bladeactivityfact = propeller['bladeactivityfact'] if 'bladeactivityfact' in propeller else False
            self.solidity = propeller['solidity'] if 'solidity' in propeller else False
            self.idesign_cl = propeller['idesign_cl'] if 'idesign_cl' in propeller else False
            # eta solver accuracy
            self.solve_dp = propeller['solve_dp'] if 'solve_dp' in propeller else 6
            self.solve_maxiter = propeller['solve_maxiter'] if 'solve_maxiter' in propeller else 30
        else:
            dictmsg = 'Argument of invalid type "{0}" was passed for "propeller", please use the built-in ' \
                      'dictionary type "{1}".'.format(str(type(propeller)), str(type({})))
            raise TypeError(dictmsg)

    def ansatzprop(self, mach, altitude_m, prop_rpm, shaftpower_w):
        """Using methods from P. M. Sforza, 'Theory of Aerospace Propulsion 2nd Edition', Section 10.6.
        This method is not a substitute for a carefully considered propeller design, is not always feasible,
        and is only here to produce a generic propeller for performance consideration only."""

        mach = np.average(actools.recastasnpfloatarray(mach))
        altitude_m = np.average(actools.recastasnpfloatarray(altitude_m))

        machstar_mps = self.isa.vsound_mps(altitudes_m=altitude_m)
        rho_kgm3 = self.isa.airdens_kgpm3(altitude_m)

        airspeed_mpstas = np.multiply(mach, machstar_mps)
        prop_rps = prop_rpm / 60

        shaftpower_w = np.average(actools.recastasnpfloatarray(shaftpower_w))
        thrustreq_n = shaftpower_w / airspeed_mpstas

        # Propeller Diameter
        # From Sforza, (2.4 < (J/Cpower)^(1/3) < 3.2) for best eta efficiency
        besteta_maxdiam = ((3.2 ** 3) * (shaftpower_w / (airspeed_mpstas * rho_kgm3 * prop_rps ** 2))) ** 0.25
        besteta_mindiam = ((2.4 ** 3) * (shaftpower_w / (airspeed_mpstas * rho_kgm3 * prop_rps ** 2))) ** 0.25
        # Now use the mach limit and trigonometry to find the prop diameter that produces this mach limit
        machlimit = 0.8
        tiplimit_mps = ((machlimit * machstar_mps) ** 2 - (mach * machstar_mps) ** 2) ** 0.5
        machlim_100_diam_m = tiplimit_mps / (prop_rps * np.pi / 2)
        # Take a weighted average of the besteta diameters
        self.diameter_m = min(0.2 * besteta_maxdiam + 0.8 * besteta_mindiam, machlim_100_diam_m)
        diameter_m = self.diameter_m

        # Propeller Blade Count, Activity Factor
        cpowerx = 0.3
        cpower = shaftpower_w / (rho_kgm3 * prop_rps ** 3 * diameter_m ** 5)
        x_adjustfact = cpower / cpowerx
        propactivityfact = 1000 * (x_adjustfact ** 0.8)
        # If the propeller has a total AF of less than 400, recommend a 3 blade propeller
        if propactivityfact < 400:
            self.bladecount = max(int(round(propactivityfact / 133, 0)), 2)
            self.bladeactivityfact = propactivityfact / self.bladecount
        else:
            self.bladecount = max(int(round(propactivityfact / 120, 0)), 2)
            self.bladeactivityfact = propactivityfact / self.bladecount
        bladecount = self.bladecount
        bladeactivityfact = self.bladeactivityfact

        # Propeller Solidity
        # Assume blade has constant chord, then solidity can be found analytically
        self.solidity = 128 * bladecount * bladeactivityfact / (100000 * np.pi)
        solidity = self.solidity

        # Integrated Design Lift Coefficient
        propdiscarea_m2 = np.pi * (diameter_m / 2) ** 2
        frontproparea_m2 = propdiscarea_m2 * solidity
        self.idesign_cl = thrustreq_n / (0.5 * rho_kgm3 * airspeed_mpstas ** 2 * frontproparea_m2)

    def efficiency(self, mach, altitude_m, shaftpower_w, prop_rpm):
        """With methods obtained from a paper by O. Gur:
        https://www.researchgate.net/publication/290676112_Practical_propeller_efficiency_model.
        Shape of arrays 'mach', 'altitude_m', and 'shaftpower_w', should be the same, or unity.

        **Parameters**

        mach
            float or array, free-stream flight Mach number.

        altitude_m
            float or array, the ISA Altitude in metres.

        shaftpower_w
            float or array, the engine shaft power in Watts.

        prop_rpm
            float, the number of full revolutions the propeller shaft makes in a minute.

        **Returns**

        eta_list
            array, a semi-empirical estimate of the propeller power/work efficiency.
        """

        altitude_m = actools.recastasnpfloatarray(altitude_m)
        mach = actools.recastasnpfloatarray(mach)
        shaftpower_w = actools.recastasnpfloatarray(shaftpower_w)

        # Find ISA air density
        rho_kgm3 = actools.recastasnpfloatarray(self.isa.airdens_kgpm3(altitude_m))

        # Propeller Solidity
        # If Solidity is not specified, assume blade has constant chord analytical solution and move forward
        if self.solidity is False:
            self.solidity = 128 * self.bladecount * self.bladeactivityfact / (100000 * np.pi)
        solidity = self.solidity

        eta_list = []

        for index, _ in enumerate(shaftpower_w):

            # Coefficient of Power
            diameter_m = self.diameter_m
            cpower = shaftpower_w[index] / (rho_kgm3[index] * (prop_rpm / 60) ** 3 * diameter_m ** 5)

            # Advance Ratio
            machstar_mps = self.isa.vsound_mps(altitudes_m=altitude_m[index])
            airspeed_mpstas = np.multiply(mach[index], machstar_mps)
            advratio = airspeed_mpstas / ((prop_rpm / 60) * diameter_m)

            # Average Blade Drag
            idesign_cl = self.idesign_cl
            cdbar_coeff = (idesign_cl * cpower / advratio)
            cdbar_blade = -0.136 * cdbar_coeff ** 2 + 0.116 * cdbar_coeff + 0.00627

            # Propeller Efficiency
            # Initial guess of efficiency
            eta1 = 0.98
            eta = 0.87

            def glauert_func(x):
                a = (2 + 5 * np.tan(x) ** 2) / (8 * np.cos(x))
                b = (3 / 16 * np.tan(x) ** 4) * np.log((1 - np.cos(x)) / (1 + np.cos(x)))
                return a - b

            # Propeller efficiency solver parameters
            dpmatch = self.solve_dp
            maxiterations = self.solve_maxiter
            count = 0

            # Propeller efficiency solver
            while count < maxiterations:
                count += 1

                eta2 = 1 - np.interp((4 * eta1 * cpower) / (np.pi ** 3 * advratio), [0, 1], [0, 1])
                phi1 = np.arctan(advratio / (np.pi * eta1 * eta2))
                eta3 = 1 - (np.pi ** 4 * eta2 ** 2 * solidity * cdbar_blade * glauert_func(phi1)) / (8 * cpower)
                eta1 = 1 - ((2 * cpower * eta2 * eta3 * eta1 ** 3) / (np.pi * advratio ** 3))

                if round(eta, dpmatch) == round(eta1 * eta2 * eta3, dpmatch):
                    eta = eta1 * eta2 * eta3
                    break
                else:
                    eta = eta1 * eta2 * eta3

            eta_list.append(eta)

        return np.array(eta_list)


def _setup(engine_name, csvpath, outputs, efficiencies):
    """
    **Parameters**

    engine_name
        string, name of engine.

    csvpath
        string, file path for named engine's CSVs.

    outputs
        list, containing strings of output type of engine (i.e. thrust and/or power).

    efficiency
        list, containing strings of efficiency type of engine (i.e. TSFC/BSFC).

    """
    # Creates list of stored data.
    data_list = os.listdir(csvpath)
    # Finds the data with the engine name in the title
    specific_data = [item for item in data_list if engine_name in item]
    # A variable used to check if any data is available to process.
    # First value is used for output data (such as thrust or power), second is
    # efficiency data (such as TSFC or BSFC), third is a sea level output
    # polynomial and fourth is a sea level take off output polynomial.
    data_available = [[False] * len(outputs), [False] * len(efficiencies), False,
                      False]
    # As there can be multiple output data variables, a list is created and
    # filled with the correct number of output data variables.
    # For each output None is append to the list to later be filled by the
    # data.
    output_data = [None] * len(outputs)
    # As there can be multiple output data variables, a list is created and
    # filled with the correct number of output data variables. For example a
    # turboprop engine can output both thrust and power and so a search is
    # required for both.
    eta_data = [None] * len(efficiencies)
    # Sets data to be None in case there is no data available.
    sl_output_poly = sl_limits = sl_to_output_poly = sl_to_limits = None
    # Finds the specific file names and checks each name against the names
    # being searched for as defined in the input.
    for data in specific_data:
        # Finds output data. This has a standard format of output + data.csv
        # i.e. Thrust data.csv or Power data.csv
        for index, output in enumerate(outputs):
            if output + "_data.csv" in data.lower():
                # Reads output data.
                output_data[index] = pd.read_csv(csvpath + os.sep + data)
                # Data is available and so the function can proceed.
                data_available[0][index] = True
            # Checks to see if there is a sea level (SL) polynomial for the output.
            if "sea_level_" + output + "_polynomial.csv" in data.lower():
                # Processes polynomial and returns
                sl_output_poly, sl_limits = _poly_process(csvpath + os.sep + data)
                # Data is available and so the function can proceed
                data_available[2] = True
            # Checks to see if there is a sea level (SL) take off (TO) output
            # polynomial.
            if "sea_level_take_off_" + output + "_polynomial.csv" in data.lower():
                # Processes polynomial and returns
                sl_to_output_poly, sl_to_limits = _poly_process(csvpath + os.sep + data)
                # Data is available and so the function can proceed
                data_available[3] = True
        # Finds TSFC Data
        for index, efficiency in enumerate(efficiencies):
            if efficiency + "_data.csv" in data.lower():
                eta_data[index] = pd.read_csv(csvpath + os.sep + data)
                # Data is available and so the function can proceed.
                data_available[1][index] = True
    if all(value is False for value in data_available):
        raise NameError("Engine was not found in stored files. Try checking name against list of engines "
                        "available in local_data function or if function is using non local data, set "
                        "Local_model=False.")
    # Returns data
    return [output_data, eta_data, sl_output_poly, sl_limits, sl_to_output_poly, sl_to_limits], data_available


def _poly_process(file_path):
    """
    **Parameters**

    file_path
        string, file path of csv polynomial

    **Returns**

    poly
        Power Series (numpy.polynomial.polynomial), numpy type polynomial.

    poly_limits
        list, contains two elements. First item is the lower limit, and last item is the upper limit
        for input values to the polynomial.

    **Example**

    ::
        _poly_process("Jet CSVs//J52 Sea level thrust polynomial.csv")

    Output: ::

        (Polynomial([32223.13844  ,   139.2130384,  2081.675712 , -2386.231868, -655.4083234,  -586.2035676,
           184.5337528], domain=[-1,  1], window=[-1,  1]),
        [0.0, 0.955730394])

    """
    with open(file_path, "r", encoding='utf-8-sig') as file:
        # Splits polynomial data into two new lines.
        poly_data = file.read().split("\n")
        # finds polynomial coefficients from stored CSV data.
        coeff_list = [float(coeff) for coeff in poly_data[0].split(",")]
        # Finds limits from stored CSV data.
        poly_limits = [float(limit) for limit in poly_data[1].split(",") if
                       len(limit) != 0]
        # Creates a polynomial for the sea level thrust.
        poly = np_poly.Polynomial(coeff_list)
        # Returns polynomial type and limits.
        return poly, poly_limits


def _griddata_interpolate(x_ref, y_ref, z_ref, x, y):
    """
    **Parameters**

    x_ref, y_ref
        float or array, x and y reference data

    z_ref
        float or array, corresponding z data.

    x, y
        float or array, data to be tested

    **Returns**

    z
        array, interpolated data.

    """
    # Interpolated data using the griddata function from scipy interpolate.
    z = griddata((x_ref, y_ref), z_ref, (x, y), method="cubic", rescale=True)
    # Returns data.
    return z


def _poly(x, poly, limits, low_warn, high_warn):
    """Uses a polynomial to find y from x.

    **Parameters**

    x
        float or array, input values for the polynomial function.

    poly
        Power Series (numpy.polynomial.polynomial), but may also be any function.

    limits
        list, contains two elements. First item is the lower limit, and last item is the upper
        limit for input values to the polynomial.

    low_warn
        string, argument message to return if x data is out of bounds (too low).

    high_warn
        string, argument message to return if x data is out of bounds (too high).

    **Returns**

    y
        array, corresponding y value for polynomial and input data.

    """
    # If the x value is too high print a warning and return nothing.
    if x > limits[1]:
        warnings.warn(high_warn, RuntimeWarning)
        return
    # If the x value is too low print a warning and return nothing.
    elif x < limits[0]:
        warnings.warn(low_warn, RuntimeWarning)
        return
    # Return the y data if the conditions are met.
    return poly(x)


def _demo_plot(min_x, max_x, min_y, max_y, func, x_label, y_label, z_label, plt_title):
    """Function for creating color gradient plot, with x and y data taking spatial dimensions,
    and z data plotted along a 1-D colour space.

    **Parameters**

    min_x
        float, smallest x value.

    max_y
        float, largest x value.

    min_y
        float, smallest y value.

    max_y
        float, largest y value.

    func
        function, method used to find z data, as if z = f(x,y).

    x_label
        string, chart x label (x-axis).

    y_label
        string, chart y label (y-axis).

    z_label
        string, chart z label (colour bar).

    plt_title
        string, the chart title.

    **Returns**

    None

    """
    # This value will be used to scale the range to produce increments
    frac = 1000
    # Finds increments for x and y data.
    x_inc = (max_x - min_x) / frac
    y_inc = (max_y - min_y) / frac
    # Creates a grid of x and y data to plot.
    x, y = np.mgrid[slice(min_x, max_x + x_inc, x_inc),
                    slice(min_y, max_y + y_inc, y_inc)]
    z = func(x, y)  # Applied function to x and y data
    plt.figure(figsize=(10, 10))  # plots a figure of size 10,10
    plt.pcolormesh(x, y, z, cmap="viridis", shading='auto')  # creates a colour map
    plt.xlabel(x_label)  # Labels x axis
    plt.ylabel(y_label)  # Labels y axis
    plt.title(plt_title)  # Adds title.
    plt.colorbar(label=z_label)  # Plots colour-bar with label
    plt.contour(x, y, z, 20, colors="white")
    plt.show()  # Shows plot

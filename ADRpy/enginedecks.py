# -*- coding: utf-8 -*-
"""
Engine Decks
--------
Provides an engine deck based on engine data for the various known engines.
A list of these engines can be printed by the local_engine_models function.
"""

import pandas as pd
from scipy.interpolate import griddata
import numpy.polynomial.polynomial as np_poly
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Author of module
_author_ = "Samuel Pearson"


def local_data(engine_type, return_data=False, references=True):
    """
    **Parameters** (all optional)
        engine_type
            string. This determines what engine types are to be listed.

        return_data
            bool.If Return_Data=False then the data will be printed out
            in a readable form if Return_Data is not False then the data will
            be returned as a dictionary with the value for each item being a
            list.

        references
            bool. If set to True then the references will returned or printed
            depending on the Return_Data setting.

    **Returns**
        If return_data=False then a list is printed of what engines are
        available and what data is available with each engine. If references is
        set to false then references will not be printed.

        If return_data=True then a dictionary is returned with the key which
        consists of a list. The first item in the list is a list of the names
        of the files and the second item is the notes for that specific engine.
        The third entry is a string that will contain the reference if
        if references is set to True. If references is set to False then there
        will not be a third reference.
    **Examples**
        local_engine_models("Jet", return_data=True, References=False)
    Output:
        {'ATF3-6A': [['ATF3-6A Thrust Data.csv'], ''],
         'F404-400': [['F404-400 Sea level thrust polynomial.csv',
           'F404-400 Thrust data.csv'],
          ''],
        ... }
    """
    # List of types available as an input.
    type_list = ["turboprop", "jet", "electric", "piston"]
    if engine_type.lower() not in type_list:
        raise LookupError("Engine type specified not valid, included types " +
                          "are: " + ", ".join(type_list[:-1]) + " and " +
                          type_list[-1])
    # Creates file name form data available
    file_name = "data" + os.sep + "engine data" + os.sep + engine_type + " CSVs"
    # creates file name string.
    data_list = os.listdir(file_name)
    # Opens CSV
    with open(file_name + os.sep + engine_type + ' data available.csv', 'r',
              encoding="utf-8-sig") as file:
        # Reads file
        reader = csv.reader(file)
        # converts to list.
        data = list(reader)
    # Creates an empty dictionary in case return_data is true.
    engine_data = {}
    for row in data:
        # Finds first element of list (engine type).
        engine = row[0]
        # Creates a dictionary of values to replace.
        csv_engine_replace = {".csv": "", engine + " ": ""}
        # Replaces Data names for turboprop.
        if engine_type == "turboprop":
            file_name_replace = {"Thrust data":
                                 "Hot Thrust (N) versus Mach Number and " +
                                 "altitude (m)", "Power data":
                                 "Power (W) versus Mach Number and " +
                                 "altitude (m)", "BSFC data":
                                 "BSFC (g/(kWh)) versus Mach Number and " +
                                 "altitude (m)"}
        # Replaces Data names for jet.
        if engine_type == "jet":
            file_name_replace = {"SL TO Thrust data":
                                 "", "Thrust data":
                                 "Thrust (N) versus Mach number and " +
                                 "altitude (m)",
                                 "TSFC data":
                                 "TSFC (g/(kNs)) versus Mach number and" +
                                 " engine thrust (N)."}
        # Replaces Data names for electric motors.
        if engine_type == "electric":
            file_name_replace = {"Efficiency data":
                                 "Efficinecy versus Engine Speed (RPM) and " +
                                 "Torque (Nm)"}
        # Replaces Data names for piston engines.
        if engine_type == "piston":
            file_name_replace = {"Power data":
                                 "Power (W) versus Engine Speed (RPM) and " +
                                 "altitude (m)", "BSFC data":
                                 "BSFC (g/(kWh)) Engine Speed (RPM) and " +
                                 "Power (W)"}
        # Generates a replace dictioanry for a specific type.
        replace_dict = {**csv_engine_replace, **file_name_replace}
        # If it is not possible for there to be engine notes then they will
        # then a blank string will be returned.
        engine_notes = row[1]
        # Engine notes are after first element.
        reference = row[-1]
        #  Appends relevant data to list.
        specific_data = [item for item in data_list if engine in item]
        # Checks to see if data is to be returned or printed.
        if return_data is False:
            # This goes through a dictionary of replacements and uses that to
            # Replace items.
            for a, b in replace_dict.items():
                specific_data = [item.replace(a, b) for item in specific_data]
                # Adds a new line after each entry to make it clearer.
            specific_data = [item for item in specific_data if len(item) > 1]
            # Prints engine data.
            print("Type:  " + engine)
            print("Data Available:\n", ", ".join(specific_data))
            if len(engine_notes) != 0:
                print("Engine Notes: \n", engine_notes)
            # Only prints references if reference data is requested and
            # there is reference data that can be printed. Given that the data
            # format stored is {engine name}; {engine notes}; {reference}
            # then the length of the list of stored engine data must have at
            # at least two items (engine name and engine notes) must exist for
            # there to be a reference. If there is a reference, the engine
            # notes field must always have comma even if there is no data
            # contained in it for this reason.
            if references and len(row) > 2:
                print("Reference: \n", reference)
            print("\n")
        else:
            # Checks to see if references are available and requested.
            if references and len(row) > 2:
                # Creates dictionary of data for easy searching with references
                engine_data[engine] = [specific_data, engine_notes, reference]
            else:
                # Creates dictionary of data for easy searching.
                engine_data[engine] = [specific_data, engine_notes]
    # Checks if data is to be returned, if not it will exit as the relevant.
    # data will have been printed.
    if return_data is False:
        return
    else:
        return(engine_data)


class turboprop_deck:
    """
    Engine model deck based off performance data for the various listed
    turboprop engines.
    """
    def __init__(self, engine, silent=False):
        """
        engine
            string input of known engine type. The engines to use can be found
            using local_data("turboprop")
        silent
            Boolean input. If True then warnings will be printed, if False then
            they will not be printed.
        """
        # Sets engine name
        self.engine = engine
        # Sets silent parameter
        self.silent = silent
        # Uses the _setup function to find the relevant data for the engine
        # and then returns pandas dataframes and polynomial types as required.
        data, self.data_available = _setup(engine, "data" + os.sep +
                                           "engine data" + os.sep +
                                           "Turboprop CSVs",
                                           ["thrust", "power"], ["bsfc"])
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
            bsfc_bsfc_gpknps = \
                pd.DataFrame.to_numpy(bsfc_df["BSFC (g/(kWh))"])
            # BSFC data list.
            self.bsfc_data = np.array([mach_bsfc, thr_bsfc_n,
                                       bsfc_bsfc_gpknps])

    def thrust(self, mach, altitude_m):
        """ Uses a cubic interpolation from the data points to find the thrust
        data at given ISA altitudes and Mach numbers, can accept arrays of
        Free-stream Mach (mach) and altitude (m) data as input and will
        return a thrust (N) array.

        **Parameters:**
        mach
            Free-stream Mach, Can be a float, list or numpy array.
        altitude_m
            ISA Altitude (m). Can be a float, list or numpy array.
        **Outputs:**
        Returns engine Thrust (N)

        **Example**
            t56 = turboprop_deck("T56-A")
            t56.thrust(100,2000)
        Output:
            array(2033.54823208)
        """
        # Checks to see if the data is available for this function.
        if self.data_available[0][0] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("No Thrust data available. Call " +
                      "local_data(\"turboprop\") for a list of available " +
                      "data.")
            return
        # Uses the _griddata_interpolate function to interpolate data.
        # thr_data[0] is the Mach Number, thr_data[1] is the altitude (m)
        # and thr_data[2] is thrust (N).
        thrust_n = _griddata_interpolate(self.thr_data[0], self.thr_data[1],
                                         self.thr_data[2], mach, altitude_m)
        return(thrust_n)

    def power(self, mach, altitude_m):
        """ Uses a cubic interpolation from the data points to find the power
        data at given ISA altitudes and Mach numbers, can accept arrays of Mach
        and thrust data as input and will return an array.

        **Parameters:**
        mach
            Free-stream Mach number, Can be a float, list or numpy array.
        altitude_m
            ISA Altitude (m). Can be a float, list or numpy array.
        **Outputs:**
        Returns engine Power (W)

        **Example**
            t56 = turboprop_deck("T56-A")
            t56.power(100,2000)
        Output:
            array(3153160.19842163)
        """
        # Checks to see if the data is available for this function.
        if self.data_available[0][1] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("No power data available. " +
                      "Call local_data(\"turboprop\") " +
                      "for a list of available data.")
            return
        # Uses the _griddata_interpolate function to interpolate data.
        # pwr_data[0] is the mach, pwr_data[1] is the altitude (m)
        # and pwr_data[2] is engine shaft power (W).
        power_w = _griddata_interpolate(self.pwr_data[0], self.pwr_data[1],
                                        self.pwr_data[2], mach, altitude_m)
        return(power_w)

    def bsfc(self, mach, altitude_m):
        """
        Uses a cubic interpolation from the data points to find the
        Brake-Specific Fuel Consumption (BSFC) data at given ISA altitudes and
        Mach numbers, can accept arrays of Mach and thrust data as input and
        will return an array.
        **Parameters:**
        mach
            Mach, Can be a float, list or numpy array.
        altitude_m
            ISA Altitude (m). Can be a float, list or numpy array.
        **Outputs:**
        Returns engine BSFC (g/(kWh)).

        **Example**
            t56 = turboprop_deck("T56-A")
            t56.bsfc(100,5000)
        Output:
            array(311.89328667)
        """
        # Checks to see if the data is available for this function.
        if self.data_available[1][0] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("No TSFC available. Call local_data(\"turboprop\") " +
                      "for a list of available data.")
            return
        # Removes any nan values from input data. As in cases where Mach data
        # has been used instead of Mach number, the thrust data was used to
        # find the Mach data, this results in a few gaps where the thrust data
        # did not go far enough into the range of the Mach data.
        bsfc_data = \
            self.bsfc_data[:, ~np.any(np.isnan(self.bsfc_data), axis=0)]
        # Uses the _griddata_interpolate function to interpolate data.
        # tsfc_
        bsfc_gpkwph = _griddata_interpolate(bsfc_data[0], bsfc_data[1],
                                            bsfc_data[2], mach, altitude_m)
        return(bsfc_gpkwph)

    def thrust_demo_plot(self):
        """
        **Parameters**
            None
        **Outputs:**
            Thrust (N) for Mach Number and ISA Altitude (m) shown as a
            colour-map.
        """
        # Checks to see if the data is available for this plot.
        if self.data_available[0] is False:
            # Checks to see if printing is allowed.
            if self.silent is False:
                print("This engine has no hot thrust data available and so " +
                      "it is not possible to plot this data.")
            return
        # Finds minimum and maximum values for Mach and altitude (m) data
        min_x = min(self.thr_data[0])
        max_x = max(self.thr_data[0])
        min_y = min(self.thr_data[1])
        max_y = max(self.thr_data[1])
        # Labels the x and y axis; colour-bar and adds title
        x_label = "Mach Number"
        y_label = "Altitude (m)"
        z_label = "Hot Thrust (N)"
        plt_title = self.engine + \
            " hot thrust (N) for Mach number and altitude (m)"
        # Applies demo plot function to produce plot.
        _demo_plot(min_x, max_x, min_y, max_y, self.thrust, x_label, y_label,
                   z_label, plt_title)

    def power_demo_plot(self):
        """
        **Parameters**
            None
        **Outputs:**
            Power (W) for Mach Number and ISA Altitude (m) shown as a
            colour-map.
        """
        # Checks to see if the data is available for this plot.
        if self.data_available[0] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("This engine has no power data available and so it is " +
                      "not possible to plot this data.")
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
        plt_title = self.engine + \
            " power (W) for Mach Number and altitude (m)"
        # Applies demo plot function to produce plot.
        _demo_plot(min_x, max_x, min_y, max_y, self.power, x_label, y_label,
                   z_label, plt_title)

    def bsfc_demo_plot(self):
        """
        **Parameters**
            None
        **Outputs:**
            Returns a plot of Mach Number against ISA Altitude (m) with the
            BSFC (g/(kWh)) shown as a colour-map.
        """
        # Checks to see if the data is available for this plot.
        if self.data_available[0] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("This engine has no BSFC data available and so it is " +
                      "not possible to plot this data.")
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
        plt_title = self.engine + \
            " BSFC (g/(kWh)) for Mach Number and altitude (m)"
        # Applies demo plot function to produce plot.
        _demo_plot(min_x, max_x, min_y, max_y, self.bsfc, x_label, y_label,
                   z_label, plt_title)


class piston_deck:
    """
    Engine model deck based off performance data for the various listed
    piston engines.
    """
    def __init__(self, engine, silent=False):
        """
        engine
            string input of known engine type. The engines to use can be found
            using local_data("piston")
        silent
            Boolean input. If True then warnings will be printed, if False then
            they will not be printed.
        """
        # sets engine name
        self.engine = engine
        # Sets silent parameter
        self.silent = silent
        # Uses the _setup function to find the relevant data for the engine
        # and then returns pandas dataframes and polynomial types as required.
        data, self.data_available = _setup(engine, "data" + os.sep +
                                           "engine data" + os.sep +
                                           "Piston CSVs", ["power"],
                                           ["bsfc", "bsfc best power"])
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
            bsfc_bsfc_gpkwph = \
                pd.DataFrame.to_numpy(bsfc_df["BSFC (g/(kWh))"])
            # BSFC for best power data list.
            self.bsfc_data = np.array([spd_bsfc, pwr_bsfc_w,
                                       bsfc_bsfc_gpkwph])
        if self.data_available[1][1] is True:
            # Assigns the dataframe to variable
            bsfc_pwr_df = data[1][1]
            # For the BSFC data, a numpy array containing the Mach number,
            # thrust and BSFC for best power data is created for later use.
            spd_bsfc_pwr = pd.DataFrame.to_numpy(bsfc_pwr_df["Speed (RPM)"])
            pwr_bsfc_pwr_w = pd.DataFrame.to_numpy(bsfc_pwr_df["Power (W)"])
            bsfc_bsfc_pwr_gpkwph = \
                pd.DataFrame.to_numpy(bsfc_pwr_df["BSFC (g/(kWh))"])
            # TSFC data list.
            self.bsfc_pwr_data = np.array([spd_bsfc_pwr, pwr_bsfc_pwr_w,
                                           bsfc_bsfc_pwr_gpkwph])

    def power(self, speed_rpm, altitude_m):
        """ Uses a cubic interpolation from the data points to find the power
        data at given ISA altitudes and Mach numbers, can accept arrays of Mach
        and thrust data as input and will return an array.

        **Parameters:**
        speed_rpm
            Engine RPM, Can be a float, list or numpy array.
        altitude_m
            ISA Altitude (m). Can be a float, list or numpy array.
        **Outputs:**
        Returns engine Power (W)

        **Example**
            io540 = piston_deck("IO-540")
            io540.power(2000, 100)
        Output:
            array(135553.2409465)
        """
        # Checks to see if the data is available for this function.
        if self.data_available[0][0] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("No Power data available. Call local_data(\"piston\") " +
                      "for a list of available data.")
            return
        # Uses the _griddata_interpolate function to interpolate data.
        # pwr_data[0] is the RPM data, pwr_data[1] is the altitude (m)
        # and pwr_data[2] is engine shaft power (W).
        power_w = _griddata_interpolate(self.pwr_data[0], self.pwr_data[1],
                                        self.pwr_data[2], speed_rpm,
                                        altitude_m)
        return(power_w)

    def bsfc(self, speed_rpm, power_w, best="power"):
        """
        Uses a cubic interpolation from the data points to find the most
        Brake-Specific Fuel Consumption (BSFC) when the engine is set to be
        most economical data at given engine speed (RPM) and power, can accept
        arrays of engine speed (RPM) and power data as input and will return
        an array.
        **Parameters:**
        speed_rpm
            Engine speed (RPM), Can be a float, list or numpy array..
        power_w
            Engine power (w). Can be a float, list or numpy array.
        best
            A string with default setting being "power". If set to "economy"
            the data will be the lowest bsfc achievable for a given engine
            speed (RPM). If set to "power", then the BSFC resulting from the
            highest power achievable at a given engine speed will be returned.

        **Outputs:**
        Returns engine BSFC (g/(kWh)).

        **Example**
            io540 = piston_deck("IO-540")
            io540.bsfc(2200, 100000, "economy")
        Output:
            array(336.83398423)
        """
        if best == "economy":
            # Checks to see if the data is available for this function.
            if self.data_available[1][0] is False:
                # Checks to see if printing is allowed
                if self.silent is False:
                    print("No BSFC available for best economy condition." +
                          "Call local_data(\"piston\") for a list " +
                          "of available data.")
                return
            # Uses the _griddata_interpolate function to interpolate data.
            bsfc_gpkwph = \
                _griddata_interpolate(self.bsfc_data[0], self.bsfc_data[1],
                                      self.bsfc_data[2], speed_rpm, power_w)
        elif best == "power":
            # Checks to see if the data is available for this function.
            if self.data_available[1][1] is False:
                # Checks to see if printing is allowed
                if self.silent is False:
                    print("No BSFC available for best power condition. " +
                          "Call local_data(\"piston\") for a list " +
                          "of available data.")
                return
            # Uses the _griddata_interpolate function to interpolate data.
            bsfc_gpkwph = \
                _griddata_interpolate(self.bsfc_pwr_data[0],
                                      self.bsfc_pwr_data[1],
                                      self.bsfc_pwr_data[2], speed_rpm,
                                      power_w)
        else:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("Not able to match user input: \"" + best + "\" with " +
                      "accepted inputs \"economy\" or \"power\".")
            return
        return(bsfc_gpkwph)

    def power_demo_plot(self):
        """
        **Parameters**
            None
        **Outputs:**
            Power (W) for engine speed (RPM) and ISA Altitude (m) shown as a
            colour-map.
        """
        # Checks to see if the data is available for this plot.
        if self.data_available[0] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("This engine has no power data available and so it is " +
                      "not possible to plot this data.")
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
        z_label = "Power (W)"
        plt_title = self.engine + \
            " power (W) for engine speed (RPM) and altitude (m)"
        # Applies demo plot function to produce plot.
        _demo_plot(min_x, max_x, min_y, max_y, self.power, x_label, y_label,
                   z_label, plt_title)

    def bsfc_demo_plot(self, best="economy"):
        """
        **Parameters**
            best
                String, default is "economy". The other possible input is
                "best". This input is to allow the user to select what
                performance data to display for when there is data for best
                engine power or best engine economy.
        **Outputs:**
            Returns a plot of engine speed (RPM) against Power (W) with the
            optimal BSFC (g/(kWh)) shown as a colour-map.
        """
        # Labels the x and y axis; colour-bar and adds title
        x_label = "Engine Speed (RPM)"
        y_label = "Power (W)"
        z_label = "BSFC (g/(kWh))"
        plt_title = self.engine + " BSFC (g/(kWh)) at best " + best + \
            " for engine speed (RPM) and power (W)"
        # Checks to see which limits and errors are required
        if best == "economy":
            # Checks to see if the data is available for this plot.
            if self.data_available[1][0] is False:
                # Checks to see if printing is allowed
                if self.silent is False:
                    print("This engine has no best economy BSFC data " +
                          "available and so it is not possible to plot this " +
                          "data.")
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
                # Checks to see if printing is allowed
                if self.silent is False:
                    print("This engine has no best power BSFC data available" +
                          " and so it is not possible to plot this data.")
                return
            # Finds minimum and maximum values for Mach and altitude (m) data.
            min_x = min(self.bsfc_pwr_data[0])
            max_x = max(self.bsfc_pwr_data[0])
            min_y = min(self.bsfc_pwr_data[1])
            max_y = max(self.bsfc_pwr_data[1])
        # If the best input has been incorrectly entered raise the same error
        # as for the bsfc function.
        else:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("Not able to match user input: \"" + best + "\" with " +
                      "accepted inputs \"economy\" or \"power\".")
            return

        # Nested definition to set BSFC type.
        def bsfc_type(speed_rpm, power_w):
            return(self.bsfc(speed_rpm, power_w, best))

        _demo_plot(min_x, max_x, min_y, max_y, bsfc_type, x_label, y_label,
                   z_label, plt_title)


class electric_deck:
    """
    Engine model deck based off performance data for the various listed
    electric motors.
    """
    def __init__(self, engine, silent=False):
        """
        engine
            string input of known engine type. The engines to use can be found
            using local_data("electric")
        silent
            Boolean input. If True then warnings will be printed, if False then
            they will not be printed.
        """
        # sets engine name
        self.engine = engine
        # Sets silent parameter
        self.silent = silent
        # Uses the _setup function to find the relevant data for the engine
        # and then returns pandas dataframes and polynomial types as required.
        data, self.data_available = _setup(engine, "data" + os.sep +
                                           "engine data" + os.sep +
                                           "Electric CSVs",
                                           ["efficiency"], [""])
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
            sft_pwr_eta = trq_eta_nm * spd_eta_rpm * np.pi / 30
            # Finds electrical power by finding shaft power by efficiency.
            elc_pwr_eta = sft_pwr_eta / eta_eta
            # Thrust data list.
            self.eta_data = np.array([spd_eta_rpm, trq_eta_nm, eta_eta,
                                      sft_pwr_eta, elc_pwr_eta])

    def efficiency(self, speed_rpm, torque_nm):
        """ Uses a cubic interpolation from the data points to find the power
        data at given ISA altitudes and Mach numbers, can accept arrays of
        motor speed (RPM) and torque (Nm) as input and will return an array.

        **Parameters:**
        speed_rpm
            Motor speed (RPM), Can be a float, list or numpy array.
        torque_nm
            Torque (Nm). Can be a float, list or numpy array.
        **Outputs:**
        Returns motor efficiency

        **Example**
            jmx57 = electric_deck("JMX57")
            jmx57.efficiency(1000, 300)
        Output:
            array(0.95183808)
        """
        # Checks to see if the data is available for this function.
        if self.data_available[0][0] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("No efficiency data available. Call " +
                      "local_data(\"electric\") for a list of available data.")
            return
        # Uses the _griddata_interpolate function to interpolate data.
        # pwr_data[0] is the mach, pwr_data[1] is the altitude (m)
        # and pwr_data[2] is engine shaft power (W).
        efficiency = _griddata_interpolate(self.eta_data[0], self.eta_data[1],
                                           self.eta_data[2], speed_rpm,
                                           torque_nm)
        return(efficiency)

    def efficiency_demo_plot(self):
        """
        **Parameters**
            None
        **Outputs:**
            Efficiency for Mach Number and ISA Altitude (m) with the shown as a
            colour-map.
        """
        # Checks to see if the data is available for this plot.
        if self.data_available[0][0] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("This engine has no efficiency data available and so " +
                      "it is not possible to plot this data.")
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
        plt_title = self.engine + \
            " efficiency for motor speed (RPM) and torque (Nm)"
        # Applies demo plot function to produce plot.
        _demo_plot(min_x, max_x, min_y, max_y, self.efficiency, x_label,
                   y_label, z_label, plt_title)


class jet_deck:
    """
    Engine model deck based off performance data for the various listed
    turbojet and turbofan engines.
    """
    def __init__(self, engine, silent=False):
        """
        engine
            string input of known engine type. The engines to use can be found
            using local_data("jet").
        silent
            Boolean input. If True then warnings will be printed, if False then
            they will not be printed.
        """
        # sets engine name
        self.engine = engine
        # Sets silent parameter
        self.silent = silent
        # Uses the _setup function to find the relevant data for the engine
        # and then returns pandas dataframes and polynomial types as required.
        data, self.data_available = _setup(engine, "data" + os.sep +
                                           "engine data" + os.sep +
                                           "Jet CSVs", ["thrust"],
                                           ["tsfc"])
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
            tsfc_tsfc_gpknps = \
                pd.DataFrame.to_numpy(tsfc_df["TSFC (g/(kNs))"])
            # TSFC data list.
            self.tsfc_data = np.array([mach_tsfc, thr_tsfc_n,
                                       tsfc_tsfc_gpknps])
        # Sea level (SL) thrust (thr) polynomial (poly) data.
        self.sl_thr_poly = data[2]
        self.sl_poly_limits = data[3]
        # Sea level (SL) take off (TO) thrust (thr) polynomial (poly) data.
        self.sl_to_thr_poly = data[4]
        self.sl_to_poly_limits = data[5]

    def thrust(self, mach, altitude_m):
        """
        Uses a cubic interpolation from the data points to find the thrust
        data at given ISA altitudes and Mach numbers, can accept arrays of Mach
        and altitude (m) data as input and will return a corresponding array
        of thrust (N) data.
        **Parameters:**
        mach
            Free-stream Mach number, Can be a float, list or numpy array.
        altitude_m
            ISA Altitude (m). Can be a float, list or numpy array.
        **Outputs:**
        Returns engine Thrust (N)

        **Example**
        jt8d9 = jet_deck("JT8D-9")
        jt8d9.thrust(0.5, 1000)
        Output:
            array(35766.25058041)
        """
        # Checks to see if the data is available for this function.
        if self.data_available[0][0] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("No Thrust data available. Call local_data(\"jet\") " +
                      "for a list of available data.")
            return
        # Uses the _griddata_interpolate function to interpolate data.
        # thr_data[0] is the Mach number, thr_data[1] is the altitude (m)
        # and thr_data[2] is thrust (N).
        thrust_n = _griddata_interpolate(self.thr_data[0], self.thr_data[1],
                                         self.thr_data[2], mach, altitude_m)
        return(thrust_n)

    def tsfc(self, mach, thrust_n):
        """
        Uses a cubic interpolation from the data points to find the Thrust
        Specific Fuel Consumption (TSFC) data at given ISA altitudes and Mach
        numbers, can accept arrays of Mach and thrust data as input and will
        return an array.
        **Parameters:**
        mach
            Free-stream Mach number, Can be a float, list or numpy array.
        thrust_n
            Thrust of engine in Newtons. Can be a float, an int, list or
            numpy array.
        **Outputs:**
        Returns engine TSFC (g/(kNs)).

        **Example**
        jt8d9 = jet_deck("JT8D-9")
        jt8d9.tsfc(0.3, 30000)

        Output:
            array(18.71142251)
        """
        # Checks to see if the data is available for this function.
        if self.data_available[1][0] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("No TSFC available. Call local_data(\"jet\") for a " +
                      "list of available data.")
            return
        # Removes any nan values from input data. As in cases where TAS data
        # has been used instead of Mach number, the thrust data was used to
        # find the Mach data, this results in a few gaps where the thrust data
        # did not go far enough into the range of the TAS data.
        tsfc_data = \
            self.tsfc_data[:, ~np.any(np.isnan(self.tsfc_data), axis=0)]
        # Uses the _griddata_interpolate function to interpolate data.
        tsfc_gpknps = _griddata_interpolate(tsfc_data[0], tsfc_data[1],
                                            tsfc_data[2], mach, thrust_n)
        return(tsfc_gpknps)

    def sl_thrust(self, mach):
        """ Uses 6th order polynomial to find take off thrust.

        **Parameters:**
        mach
            Free-stream Mach number, Input can be float or int.

        **Outputs:**
        Returns engine thrust at sea level in N.

        **Example**
        jt8d9 = jet_deck("JT8D-9")
        jt8d9.sl_thrust(0.5)

        Output:
            37412.617753750004
        """
        # Checks to see if the data is available for this function.
        if self.data_available[2] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("This engine has no sea level thrust data available." +
                      " Call local_data(\"jet\") for a list of available " +
                      "data.")
            return
        # Creates limits warning and suggestions
        high_warn = "Input Mach number too high, limit of provided data is:" +\
            " Mach " + str(self.sl_poly_limits[-1]) +\
            ". To resolve use a lower Mach number."
        low_warn = "Input Mach number too low, limit of provided data is: " +\
            "Mach " + str(self.sl_poly_limits[0]) +\
            ". To resolve use a higher Mach number."
        # Applies polynomial function
        thrust_n = _poly(mach, self.sl_thr_poly, self.sl_poly_limits, low_warn,
                         high_warn, self.silent)
        return(thrust_n)

    def sl_take_off_thrust(self, mach):
        """ Uses 6th order polynomial to find sea level thrust.

        **Parameters:**
        mach
            Free-stream Mach number, Input can be float or int.

        **Outputs:**
        Returns engine take off thrust at sea level in N.

        **Example**
        jt8d9 = jet_deck("JT8D-9")
        jt8d9.sl_take_off_thrust(0.4)

        Output:
            53179.24866385598
        """
        # Checks to see if the data is available for this function.
        if self.data_available[3] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("No sea level take off thrust data available. Call " +
                      "local_data(\"jet\") for a list of available data.")
            return
        # Creates limits warning and suggestions.
        high_warn = "Input Mach number too high, limit of provided data is:" +\
            " Mach " + str(self.sl_to_poly_limits[-1]) +\
            ". To resolve use a lower Mach number."
        low_warn = "Input Mach number too low, limit of provided data is:" +\
            " Mach " + str(self.sl_to_poly_limits[0]) +\
            ". To resolve use a higher Mach number."
        # Applies polynomial function
        thrust_n = _poly(mach, self.sl_to_thr_poly, self.sl_to_poly_limits,
                         low_warn, high_warn, self.silent)
        return(thrust_n)

    def thrust_demo_plot(self):
        """
        **Parameters**
            None
        **Outputs:**
            Thrust (N) for Mach Number and ISA Altitude (m) with the shown as a
            colour-map.
        """
        # Checks to see if the data is available for this plot.
        if self.data_available[0][0] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("This engine has no thrust data available and so it is" +
                      " not possible to plot this data.")
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
        plt_title = self.engine + \
            " thrust (N) for Mach number and altitude (m)"
        # Applies demo plot function to produce plot.
        _demo_plot(min_x, max_x, min_y, max_y, self.thrust, x_label, y_label,
                   z_label, plt_title)

    def tsfc_demo_plot(self, alt=False):
        """
        Plots TSFC data in a colour plot for Mach number and Altitude.
        **Parameters**
            alt
                Boolean type. If True then the TSFC (g/(kNs)) for Mach number
                and altitude (m) is plotted. If False TSFC (g/(kNs)) for Mach
                number and Thrust is plotted.
        **Outputs:**
            If alt is False then TSFC is plotted against Mach number and
            Thrust (N) as a colour-plot. If alt is True, then TSFC (g/(kNs)) is
            plotted Mach Number and altitude (m).
        """
        # Checks to see if TSFC data is available. If not then a warning
        # message is printed out and the function does not return anything.
        if self.data_available[1][0] is False:
            # Checks to see if printing is allowed
            if self.silent is False:
                print("This engine has no TSFC data available and"
                      + " so it is not possible to plot this data.")
            return
        # Checks to see if alt is True.
        if alt is False:
            # If it is then it checks to see if the Thrust data is available.
            # If not then a warning message is printed out and the function
            # does not return anything.
            if self.data_available[0][0] is False:
                # Checks to see if printing is allowed
                if self.silent is False:
                    print("This engine has no thrust data available and"
                          + " so it is not possible to plot this data." +
                          " Set alt to False to plot a graph of TSFC for" +
                          " Mach number and altitude.")
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
            plt_title = self.engine + \
                " TSFC (g/(kNs)) for Mach number and Thrust (N)"
            # Applies demo plot function to produce plot.
            _demo_plot(min_x, max_x, min_y, max_y, self.tsfc, x_label, y_label,
                       z_label, plt_title)
            return

        # Finds minimum and maximum values for the Mach and altitude (m) data.
        min_x = min(self.thr_data[0])
        max_x = max(self.thr_data[0])
        min_y = min(self.thr_data[1])
        max_y = max(self.thr_data[1])
        # Labels the x and y axis; colour-bar and adds title
        x_label = "Mach Number"
        y_label = "Altitude (m)"
        z_label = "TSFC (g/(kNs))"
        plt_title = self.engine + \
            " TSFC (g/(kNs)) for Mach number and altitude (m)"

        # Nested function to allow TSFC (g/(kNs)) to be expressed as a function
        # of Mach number and altitude (m).
        def _tsfc_fun(mach, altitude):
            return(self.tsfc(mach, self.thrust(mach, altitude)))
        # Applies demo plot function to produce plot. Using a lambda function
        # to provide the thrust for the tsfc function.
        _demo_plot(min_x, max_x, min_y, max_y, _tsfc_fun, x_label, y_label,
                   z_label, plt_title)


def _setup(engine, location, outputs, efficiencies):
    """
    **Parameters**
        engine
            string. Name of engine
        location
            string. Relevant file path of engine CSVs
        output
            list. A list containing strings of output type of engine
            (i.e. thrust and/or power)
        efficiency
            list. A list containing strings of efficiency type of engine
            (i.e. TSFC/BSFC)
    **Returns**

    """
    # Creates list of stored data.
    data_list = os.listdir(location)
    # Finds the data with the engine name in the title
    specific_data = [item for item in data_list if engine in item]
    # A variable used to check if any data is available to process.
    # First value is used for output data (such as thrust or power), second is
    # efficiency data (such as TSFC or BSFC), third is a sea level output
    # polynomial and fourth is a sea level take off output polynomial.
    data_available = [[False]*len(outputs), [False]*len(efficiencies), False,
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
            if output + " data.csv" in data.lower():
                # Reads output data.
                output_data[index] = pd.read_csv(location + os.sep + data)
                # Data is available and so the function can proceed.
                data_available[0][index] = True
        # Finds TSFC Data
        for index, efficiency in enumerate(efficiencies):
            if efficiency + " data.csv" in data.lower():
                eta_data[index] = pd.read_csv(location + os.sep + data)
                # Data is available and so the function can proceed.
                data_available[1][index] = True
        # Checks to see if there is a sea level (SL) polynomial for the output.
        if "sea level " + output + " polynomial.csv" in data.lower():
            # Processes polynomial and returns
            sl_output_poly, sl_limits = _poly_process(location + os.sep + data)
            # Data is available and so the function can proceed
            data_available[2] = True
        # Checks to see if there is a sea level (SL) take off (TO) output
        # polynomial.
        if "sea level take off " + output + " polynomial.csv" in data.lower():
            # Processes polynomial and returns
            sl_to_output_poly, sl_to_limits = _poly_process(location + os.sep
                                                            + data)
            # Data is available and so the function can proceed
            data_available[3] = True
    if all(value is False for value in data_available):
        raise NameError("Engine was not found in stored files." +
                        " Try checking name against list of engines" +
                        " available in local_engine_models function or" +
                        " if function is using non local data, set" +
                        " Local_model=False.")
    # Returns data
    return([output_data, eta_data, sl_output_poly, sl_limits,
            sl_to_output_poly, sl_to_limits], data_available)


def _poly_process(file_path):
    """
    **Parameters**
        file_path
            string. file path of csv polynomial
    **Returns**
        numpy.polynomial.polynomial.Polynomial type polynomial and a list with
        two items in, the first item is the lower limit and the last item is
        the upper limit.
    **Examples**
        _poly_process("Jet CSVs//J52 Sea level thrust polynomial.csv")

        (Polynomial([32223.13844  ,   139.2130384,  2081.675712 , -2386.231868
            , -655.4083234,  -586.2035676,   184.5337528], domain=[-1,  1],
             window=[-1,  1]),
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
        return(poly, poly_limits)


def _griddata_interpolate(x_ref, y_ref, z_ref, x, y):
    """
    **Parameters**
        x_ref, y_ref
            Can be a float, list or numpy array. x and y reference data
        z_ref
            Can be a float, list or numpy array. Corresponding z data.
        x, y
            Can be a float, list or numpy array. Data to be tested
    **Returns**
        z
            Interpolated data.
    """
    # Interpolated data using the griddata function from scipy interpolate.
    z = griddata((x_ref, y_ref), z_ref, (x, y), method="cubic", rescale=True)
    # Returns data.
    return(z)


def _poly(x, poly, limits, low_warn, high_warn, silent):
    """
    Uses polynomial to find y from x.

    **Parameters:**
        x
            Can be a float, list or numpy array. Input values into polynomial
            function.
        poly
            Typically np.polynomial.polynomial.Polynomial or any function.
        limits
            list containing lower and upper limits. First item in list is the
            lower limit and second item is the upper limit.
        low_warn
            A string to be printed if the data is below the minimum x value.
        high_warn
            A string to be printed if the data is above the maximum x value.
        silent
            Boolean command. If false, warnings will not be printed.
    **Returns**
        y
            Can be a float, list or numpy array. Corresponding y value for
            data.
    """
    # If the x value is too high print a warning and return nothing.
    if x > limits[1]:
        # Checks to see if printing is allowed
        if silent is False:
            print(high_warn)
        return
    # If the x value is too low print a warning and return nothing.
    elif x < limits[0]:
        # Checks to see if printing is allowed
        if silent is False:
            print(low_warn)
        return
    # Return the y data if the conditions are met.
    return poly(x)


def _demo_plot(min_x, max_x, min_y, max_y, func, x_label, y_label, z_label,
               plt_title):
    """
    **Parameters**
        min_x
            smallest x value
        max_y
            largest x value
        min_y
            smallest y value
        max_y
            largest y value
        func
            function to be applied to data to get z value.
        x_label
            A string containing the x label.
        y_label
            A string containing the y label.
        z_label
            A string containing the label for the colour-bar.
        plt_title
            A string containing the plot title.
    **Outputs:**
        Returns a colour-map of the values to plot.
    """
    # This value will be used to scale the range to produce increments
    frac = 1000
    # Finds increments for x and y data.
    x_inc = (max_x - min_x)/frac
    y_inc = (max_y - min_y)/frac
    # Creates a grid of x and y data to plot.
    x, y = np.mgrid[slice(min_x, max_x + x_inc, x_inc),
                    slice(min_y, max_y + y_inc, y_inc)]
    z = func(x, y)  # Applied function to x and y data
    plt.figure(figsize=(10, 10))  # plots a figure of size 10,10
    plt.pcolormesh(x, y, z, cmap="viridis")  # creates a colour map
    plt.xlabel(x_label)  # Labels x axis
    plt.ylabel(y_label)  # Labels y axis
    plt.title(plt_title)  # Adds title.
    plt.colorbar(label=z_label)  # Plots colour-bar with label
    plt.contour(x, y, z, 20, colors="white")
    plt.show()  # Shows plot

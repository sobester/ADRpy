![ADRpy](https://github.com/sobester/ADRpy/raw/master/docs/ADRpy/ADRpy_splash.png)

Aircraft Design Recipes in Python
=================================

A library of aircraft conceptual design and performance analysis tools, including
virtual (design) atmospheres, constraint analysis methods, propulsion system 
performance models, conversion functions and much else.

version number: 0.1.6

author: Andras Sobester

Installation / Usage
--------------------

On most systems you should be able to simply open an operating system terminal
and at the command prompt type

    $ pip install ADRpy
    
or

    $ python -m pip install ADRpy
    
NOTE: `pip` is a Python package; if it is not available on your system, download
[get-pip.py](https://bootstrap.pypa.io/get-pip.py) and run it in Python by entering

    $ python get-pip.py
    
at the operating system prompt.

An alternative approach to installing ADRpy is to clone the GitHub repository, by typing

    $ git clone https://github.com/sobester/ADRpy.git

at the command prompt and then executing the setup file in the same directory by entering:

    $ python setup.py install

    
A 'hello world' example: atmospheric properties
-----------------------------------------------

There are several options for running the examples shown here: you could copy and paste them 
into a `.py` file, save it and run it in Python, or you could enter the lines, in sequence,
at the prompt of a Python terminal. You could also copy and paste them into a Jupyter notebook
(`.ipynb` file) cell and execute the cell.

```python
from ADRpy import atmospheres as at
from ADRpy import unitconversions as co

# Instantiate an atmosphere object: an ISA with a +10C offset
isa = at.Atmosphere(offset_deg=10)

# Query the ambient density in this model at 41,000 feet 
print("ISA+10C density at 41,000 feet (geopotential):", 
      isa.airdens_kgpm3(co.feet2m(41000)), "kg/m^3")
```

A design example: wing/powerplant sizing for take-off
-----------------------------------------------------

```python
# Compute the thrust to weight ratio required for take-off, given
# a basic design brief, a basic design definition and a set of 
# atmospheric conditions

from ADRpy import atmospheres as at
from ADRpy import constraintanalysis as ca
from ADRpy import unitconversions as co


# The environment: 'unusually high temperature at 5km' atmosphere
# from MIL-HDBK-310. 

# Extract the relevant atmospheric profiles...
profile_ht5_1percentile, _ = at.mil_hdbk_310('high', 'temp', 5)

# ...then use them to create an atmosphere object 
m310_ht5 = at.Atmosphere(profile=profile_ht5_1percentile)

#====================================================================

# The take-off aspects of the design brief:
designbrief = {'rwyelevation_m':1000, 'groundrun_m':1200}

# Basic features of the concept:
# aspect ratio, engine bypass ratio, throttle ratio 
designdefinition = {'aspectratio':7.3, 'bpr':3.9, 'tr':1.05}

# Initial estimates of aerodynamic performance:
designperf = {'CDTO':0.04, 'CLTO':0.9, 'CLmaxTO':1.6,
              'mu_R':0.02} # ...and wheel rolling resistance coeff.

# An aircraft concept object can now be instantiated
concept = ca.AircraftConcept(designbrief, designdefinition,
                             designperf, m310_ht5)

#====================================================================

# Compute the required standard day sea level thrust/MTOW ratio reqd.
# for the target take-off performance at a range of wing loadings:
wingloadinglist_pa = [2000, 3000, 4000, 5000]

tw_sl, liftoffspeed_mpstas, _ = concept.twrequired_to(wingloadinglist_pa)

# The take-off constraint calculation also supplies an estimate of
# the lift-off speed; this is TAS (assuming zero wind) - we convert 
# it to equivalent airspeed (EAS), in m/s:
liftoffspeed_mpseas = \
m310_ht5.tas2eas(liftoffspeed_mpstas, designbrief['rwyelevation_m'])

print("Required T/W and V_liftoff under MIL-HDBK-310 conditions:")
print("\nT/W (std. day, SL, static thrust):", tw_sl)
print("\nLiftoff speed (KEAS):", co.mps2kts(liftoffspeed_mpseas))
```

A complete example: wing/powerplant sizing for a single engine prop
-------------------------------------------------------------------

[View the single engine prop example as a Jupyter notebook on nbviewer](https://nbviewer.jupyter.org/github/sobester/ADRpy/blob/master/docs/ADRpy/single_engine_prop_power_requirements.ipynb) (click on the binder icon in the top right corner for of the nbviewer page for an editable, 'live', online version of the notebook).

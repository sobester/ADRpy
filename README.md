![ADRpy](https://github.com/sobester/ADRpy/raw/master/docs/ADRpy/ADRpy_splash.png)

Aircraft Design Recipes in Python
=================================

[![PyPI version](https://badge.fury.io/py/ADRpy.svg)](https://badge.fury.io/py/ADRpy)
[![Build Status](https://travis-ci.com/sobester/ADRpy.svg?branch=master)](https://travis-ci.com/sobester/ADRpy)

A library of aircraft conceptual design and performance analysis tools, including
virtual (design) atmospheres, constraint analysis methods, propulsion system 
performance models, conversion functions and much else.

For a detailed description of the library, please consult the
[Documentation](https://adrpy.readthedocs.io/en/latest/). To get started,
follow the instructions below.

For video tutorials and explainers (a.k.a. *ADRpy Shorts*) scroll to the bottom of this page.

author: Andras Sobester

Installation / Usage
--------------------

ADRpy is written in Python 3 and tested in Python 3.5, 3.6, 3.7 and 3.8.

It is not available for Python 2.

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

You should see the following output:

    ISA+10C density at 41,000 feet (geopotential): 0.274725888531 kg/m^3

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

You should see the following output:

    Required T/W and V_liftoff under MIL-HDBK-310 conditions:

    T/W (std. day, SL, static thrust): [ 0.19618164  0.2710746   0.34472518  0.41715311]

    Liftoff speed (KEAS): [  96.99203483  118.79049722  137.1674511   153.35787248]


More extensive examples - a library of notebooks
------------------------------------------------

Click on [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/sobester/ADRpy/master?filepath=/docs/ADRpy/notebooks) to open a library of examples recorded in Jupyter notebooks. You can play
with these 'live' in Binder, or you can click File / Download as / ... to create your own local 
copy in any number of formats. [*Note: if you don't want to wait for Binder to generate the library,
you can still access the 'static' versions of the notebooks through nbviewer - click on the required
notebook in the lower half of the holding page.* ]


ADRpy Shorts - video tutorials and explainers
---------------------------------------------------------

**1. An Aircraft Engineer's Brief Introduction to Modelling the Atmosphere**

[![1. An Aircraft Engineer's Brief Introduction to Modelling the Atmosphere](http://img.youtube.com/vi/II9vuVCgV-w/0.jpg)](http://www.youtube.com/watch?v=II9vuVCgV-w)


**2. On V-n Diagrams and How to Build them in ADRpy**

[![2. On V-n Diagrams and How to Build them in ADRpy](http://img.youtube.com/vi/s-d5z-BQovY/0.jpg)](http://www.youtube.com/watch?v=s-d5z-BQovY)


**3. Speed in aviation - GS, WS, TAS, IAS, CAS and EAS**

[![3. Speed in aviation - GS, WS, TAS, IAS, CAS and EAS](http://img.youtube.com/vi/WSzDXlTlXiI/0.jpg)](http://www.youtube.com/watch?v=WSzDXlTlXiI)


More ADRpy Shorts coming soon!
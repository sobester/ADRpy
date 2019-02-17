Aircraft Design Recipes in Python
=================================

A library of aircraft conceptual design and performance codes, including virtual (design) atmospheres and constraint analysis methods.

version number: 0.0.10
author: Andras Sobester

Installation / Usage
--------------------

To install use pip:

    $ pip install ADRpy

Or clone the repo:

    $ git clone https://github.com/sobester/ADRpy.git
    $ python setup.py install
    
Example
-------
A 'hello world': convert equivalent airspeeds into calibrated in an ISA at a given altitude.

```python
import numpy as np
from ADRpy import atmospheres as at
from ADRpy import unitconversions as co

isa = at.Atmosphere()

keas = np.array([100, 200, 300])
altitude_m = co.feet2m(40000)

kcas, mach = isa.keas2kcas(keas, altitude_m)

print(kcas)
```
---

```python
[ 101.25392563  209.93839073  333.01861569]
```
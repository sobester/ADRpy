Aircraft Design Recipes in Python -- User's Guide
=================================================

**Welcome to ADRpy**, a library of aircraft design and performance tools suitable for
conceptual design calculations. The classes, methods and functions of the library fall
into three broad categories:

1. **Models of the operating environment.** These live in the :ref:`atmospheres_module`
and include virtual atmospheres (ISA, MIL-HDBK-310, etc.), runway models (suitable 
for takle-off and landing performance modelling) and models of propulsion performance
response to variations in ambient conditions.

2. **Conceptual sizing methods** for fixed wing aircraft sizing (wing area and thrust/power
requirements), given a set of constraints. These can be found in the :ref:`constraints_module`.

3. **Utilities**

Installing and running ADRpy
----------------------------

For the code and installation instructions see `the ADRpy GitHub 
repository <https://github.com/sobester/ADRpy#aircraft-design-recipes-in-python>`_.

ADRpy Modules
-------------

.. automodule:: atmospheres
    :members:

.. automodule:: constraintanalysis
    :members:

* :ref:`genindex`
* :ref:`search`


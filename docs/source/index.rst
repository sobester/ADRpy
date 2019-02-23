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

3. **Utilities**, including a module for :ref:`unitconversions_module` and a set of 
:ref:`mtools4acdc_module`.

Installing and running ADRpy
----------------------------

On most systems you should be able to simply open an operating system terminal
and at the command prompt type :code:`pip install ADRpy` or
:code:`python -m pip install ADRpy` (:code:`pip` is a Python package: 
if it is not available on your system, download `get-pip.py <https://bootstrap.pypa.io/get-pip.py>`_ 
and run it in Python by entering :code:`python get-pip.py` at the operating system prompt).

This is an open source project (released under a `GPLv3 <https://www.gnu.org/licenses/gpl-3.0.en.html>`_
license) -- for the source code (and further installation instructions)
see `the ADRpy GitHub repository <https://github.com/sobester/ADRpy#aircraft-design-recipes-in-python>`_.

ADRpy Modules
-------------

.. automodule:: atmospheres
    :members:

.. automodule:: constraintanalysis
    :members:

.. automodule:: unitconversions
    :members:

.. automodule:: mtools4acdc
    :members:

* :ref:`genindex`
* :ref:`search`


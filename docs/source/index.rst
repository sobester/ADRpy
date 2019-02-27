.. image:: https://badge.fury.io/py/ADRpy.svg
    :target: https://badge.fury.io/py/ADRpy

.. image:: https://travis-ci.com/sobester/ADRpy.svg?branch=master
    :target: https://travis-ci.com/sobester/ADRpy

Aircraft Design Recipes in Python -- User's Guide
=================================================

**Welcome to ADRpy**, a free library of aircraft design and performance analysis tools suitable for
rapid sizing calculations. The models implemented in ADRpy are largely analytical, enabling 
fast explorations of large design spaces. Most of the methods can already be used in the 
earliest phases of the design process, even before a geometry model is built. In fact, ADRpy
can serve as the basis of sensitivity analyses and uncertainty quantification (UQ) exercises
as part of the analysis of the feasibility of the design requirements.

The classes, methods and functions of the library fall into three broad categories:

1. **Models of the operating environment.** These live in the :ref:`atmospheres_module`
and include virtual atmospheres (ISA, MIL-HDBK-310, etc.), runway models (suitable 
for take-off and landing performance modelling using a 'real world' runway database)
and models of propulsion performance response to variations in ambient conditions.

2. **Conceptual sizing methods** for fixed wing aircraft sizing (wing area and thrust/power
requirements), given a set of constraints, such as take-off distance, climb rate, etc.
These can be found in the :ref:`constraints_module`.

3. **Utilities**, including a module for :ref:`unitconversions_module` and a set of 
:ref:`mtools4acdc_module`.

This document contains numerous usage examples and details on the inputs and outputs
of each class, method and function. Any problems, issues, questions, please
`raise an issue on GitHub <https://github.com/sobester/ADRpy/issues>`_. Happy designing! 

Installing and running ADRpy
----------------------------

First and foremost, you will need to have Python installed - you can get the latest version
`here <https://www.python.org/downloads/>`_ or as part of a number of alternative
`packages <https://www.python.org/download/alternatives/>`_. 

Once you have Python, on most systems you should be able to simply open an operating system terminal
and at the command prompt type :code:`python -m pip install ADRpy` or just :code:`pip install ADRpy`
(:code:`pip` is a Python package; in the unlikely event that it is not available on your system,
download `get-pip.py <https://bootstrap.pypa.io/get-pip.py>`_ and run it by 
entering :code:`python get-pip.py` at the operating system prompt).

For the ADRpy source code, license conditions (GPLv3), and alternative installation instructions
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


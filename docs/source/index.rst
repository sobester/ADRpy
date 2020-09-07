.. image:: https://badge.fury.io/py/ADRpy.svg
    :target: https://badge.fury.io/py/ADRpy

.. image:: https://travis-ci.com/sobester/ADRpy.svg?branch=master
    :target: https://travis-ci.com/sobester/ADRpy

Aircraft Design Recipes in Python -- User's Guide
=================================================

by `Andras Sobester <https://www.southampton.ac.uk/engineering/about/staff/as7.page>`_

**Welcome to ADRpy**, a free library of aircraft design and performance analysis tools suitable for
rapid sizing calculations. The models implemented in ADRpy are largely analytical, enabling 
fast explorations of large design spaces. Most of the methods can already be used in the 
earliest phases of the design process, even before a geometry model is built. In fact, ADRpy
can serve as the basis of sensitivity analyses and uncertainty quantification (UQ) exercises
as part of the analysis of the feasibility of the design requirements.

The classes, methods and functions of the library fall into these broad categories:

1. **Models of the operating environment.** These live in the :ref:`atmospheres_module`
module and include virtual atmospheres (ISA, MIL-HDBK-310, etc.), runway models (suitable 
for take-off and landing performance modelling using a 'real world' runway database)
and models of propulsion performance response to variations in ambient conditions.

2. **Sizing and performance analysis methods** for fixed wing aircraft. They can be
found in the :ref:`constraints_module`.

3. **Airworthiness**. Tools to assist in analysing the airworthiness of a design from
a certification standpoint. See the :ref:`airworthiness_module`.

4. **Utilities**, including a module for :ref:`unitconversions_module` and a set of 
:ref:`mtools4acdc_module`.

This document contains numerous usage examples and details on the inputs and outputs
of each class, method and function. You can copy and paste these into .py files and 
execute those in a development environment or from Python's command prompt, you can
copy and paste them line by line into a Python terminal or (perhaps most usefully)
into a `Jupyter notebook <https://jupyter.org>`_. Any problems, issues, questions, please
`raise an issue on GitHub <https://github.com/sobester/ADRpy/issues>`_. Happy designing! 

Installing and running ADRpy
----------------------------

ADRpy is written in Python 3 and tested in Python 3.5, 3.6, 3.7 and 3.8. It is not 
available for Python 2.

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

Notebooks
---------

ADRpy includes a library of examples recorded in Jupyter notebooks. You can play
with these 'live' in Binder:

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/sobester/ADRpy/master?filepath=/docs/ADRpy/notebooks

...or you can click File / Download as / ... to create your own local copy in any number of formats.
[*Note: if you don't want to wait for Binder to generate the library, you can still access the 
'static' versions of the notebooks through nbviewer - click on the required notebook in 
the lower half of the holding page.*]

ADRpy Modules
-------------

.. automodule:: atmospheres
    :members:

.. automodule:: constraintanalysis
    :members:

.. automodule:: airworthiness
    :members:

.. automodule:: unitconversions
    :members:

.. automodule:: mtools4acdc
    :members:

* :ref:`genindex`
* :ref:`search`


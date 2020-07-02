Aircraft Design Recipes in Python -- a Guide for Contributors
=============================================================

Thank you for considering to contribute to ADRpy! Please follow these...

General principles
------------------

1. Follow the [PEP 8 principles](https://www.python.org/dev/peps/pep-0008/) as far
as a possible, but never at the expense of readability or execution speed.

2. Variable and function/method names. If dimensional, they should have units at 
the end of the name, for example `dynamicpressure_pa(airspeed_mps=0, altitudes_m=0)`. 
Names of internal function/method names not meant to be called directly by the user
should start with an underscore, e.g., `_somefunction(x)`.

3. Each function/method exposed to the user must have a docstring containing three
main sections: Parameters, Returns and a simple Example (plus an optional Notes section
if it helps with clarity and usability, e.g., to place the item in a broader context).
Use an existing function (e.g., `twrequired_trn(wingloading_pa)` in the `constraintanalysis`
module) as a template. Include equations if appropriate (see, as an example, 
`thrusttoweight_takeoff`). Always check afterwards that your docstring has rendered 
correctly on `readthedocs`.

4. Beyond the simple example in each docstring, a more extensive case study, with
detailed, step-by-step explanations, should be included in a Jupyter notebook and
placed in ADRpy/docs/ADRpy/notebooks.

5. Every new function/method needs a test, the more extensive the better. The test
should be based on public domain data. Please include a reference to the data, going back
as close to its original source as you can get. If your new code/test has new dependencies,
remember to add these to the `requirements.txt` file at the top level of this directory structure.

6. If your code requires a unit conversion, always use the `unitconversions.py` module
and never code conversions locally into the method. If `unitconversions.py` does not
have the conversion you're after, please add it (along with an appropriate test).

7. DRY - Don't Repeat Yourself (a generalisation of 6.) - never write the same code
in multiple different places. "Every piece of knowledge must have a single, unambiguous,
authoritative representation within a system." (Hunt and Thomas)

Happy coding!
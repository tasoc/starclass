=======================================
The TASOC Stellar Classification module
=======================================
.. image:: https://travis-ci.org/tasoc/starclass.svg?branch=devel
    :target: https://travis-ci.org/tasoc/starclass
.. image:: https://img.shields.io/github/license/tasoc/starclass.svg
    :alt: license
    :target: https://github.com/tasoc/starclass/blob/devel/LICENSE

This module provides the stellar classification setup for the TESS Asteroseismic Science Operations Center (TASOC).

The overall strategy of the classification pipeline is to have different classifiers are run on the same data, and all the results from the individual classifiers are passed into an overall "meta-classifier" which will assign the final classifications based on the inputs from all classifiers.

Classification is done in two levels (1 and 2), where the first level separates stars into overall classes of stars that exhibit similar lightcurves. In level 2, these classes are further separated into the individual pulsation classes.

.. image:: docs/burger_diagram.png

The code is available through our GitHub organization (https://github.com/tasoc/starclass) and full documentation for this code can be found on https://tasoc.dk/code/.

Installation instructions
=========================
* Start by making sure that you have `Git Large File Storage (LFS) <https://git-lfs.github.com/>`_ installed. You can verify that is installed by running the command:

  >>> git lfs version

* Go to the directory where you want the Python code to be installed and simply download it or clone it via *git* as::

  >>> git clone https://github.com/tasoc/starclass.git .

* All dependencies can be installed using the following command. It is recommended to do this in a dedicated `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ or similar:

  >>> pip install -r requirements.txt

How to run tests
================
You can test your installation by going to the root directory where you cloned the repository and run the command::

>>> pytest
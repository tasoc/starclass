Common Features (``starclass.features``)
========================================
This module contains the features extracted for all targets 

Light curve
-----------
The primary feature which all classifiers will have available is the light curve of the target
in question. This is available in the features dictionary under the key ``'lightcurve'`` and
will contain a :class:`lightkurve.TessLightCurve` object.

Pre-calculated diagnostics
--------------------------
The following features are also provided to the classifiers in the features dictionary-
Some of these could be useful in, some may not, but since they are already computed by the
previous steps in the TASOC Pipeline, they can be provided to the classifiers "for free":

* TESS Magnitude (``'tmag'``).
* Variance (``'variance'``).
* RMS on 1 hour timescale (``'rms_hour'``).
* Point-to-point scatter (``'ptp'``).

Power spectrum (``starclass.features.powerspectrum``)
-----------------------------------------------------
.. automodule:: starclass.features.powerspectrum
    :members:
    :undoc-members:
    :show-inheritance:

Frequency Extraction (``starclass.features.freqextr``)
------------------------------------------------------
.. automodule:: starclass.features.freqextr
    :members:
    :undoc-members:
    :show-inheritance:
	
FliPer (``starclass.features.fliper``)
--------------------------------------
.. automodule:: starclass.features.fliper
    :members:
    :undoc-members:
    :show-inheritance:

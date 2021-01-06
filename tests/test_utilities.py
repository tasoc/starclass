#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import numpy as np
import warnings
from lightkurve import LightCurve, LightkurveWarning
import conftest # noqa: F401
from starclass.utilities import rms_timescale

#--------------------------------------------------------------------------------------------------
def test_rms_timescale():

	time = np.linspace(0, 27, 100)
	flux = np.zeros(len(time))

	rms = rms_timescale(LightCurve(time=time, flux=flux))
	print(rms)
	np.testing.assert_allclose(rms, 0)

	rms = rms_timescale(LightCurve(time=time, flux=flux*np.nan))
	print(rms)
	assert np.isnan(rms), "Should return nan on pure nan input"

	rms = rms_timescale(LightCurve(time=[], flux=[]))
	print(rms)
	assert np.isnan(rms), "Should return nan on empty input"

	# Pure nan in the time-column should raise ValueError:
	with np.testing.assert_raises(ValueError):
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore', category=LightkurveWarning, message='LightCurve object contains NaN times')
			rms = rms_timescale(LightCurve(time=time*np.nan, flux=flux))

	# Time with invalid contents (e.g. Inf) should throw an ValueError:
	time_invalid = time.copy()
	time_invalid[2] = np.inf
	with np.testing.assert_raises(ValueError):
		rms = rms_timescale(LightCurve(time=time_invalid, flux=flux))

	time_someinvalid = time.copy()
	time_someinvalid[2] = np.nan
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore', category=LightkurveWarning, message='LightCurve object contains NaN times')
		rms = rms_timescale(LightCurve(time=time_someinvalid, flux=flux))
	print(rms)
	np.testing.assert_allclose(rms, 0)

	# Test with timescale longer than timespan should return zero:
	flux = np.random.randn(1000)
	time = np.linspace(0, 27, len(flux))
	rms = rms_timescale(LightCurve(time=time, flux=flux), timescale=30.0)
	print(rms)
	np.testing.assert_allclose(rms, 0)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

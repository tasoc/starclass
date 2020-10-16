#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import numpy as np
import tempfile
import warnings
from lightkurve import LightCurve, LightkurveWarning
import conftest # noqa: F401
from starclass.utilities import savePickle, loadPickle, rms_timescale

#--------------------------------------------------------------------------------------------------
def test_pickle():

	# Pretty random object to save and restore:
	test_object = {'test': 42, 'whatever': (1,2,3,4,5432)}

	with tempfile.TemporaryDirectory() as tmpdir:
		fname = os.path.join(tmpdir, 'test.pickle')

		# Save object in file:
		savePickle(fname, test_object)
		assert os.path.exists(fname), "File does not exist"

		# Recover the object again:
		recovered_object = loadPickle(fname)
		print(recovered_object)
		assert test_object == recovered_object, "The object was not recovered"

		# Save object in file:
		savePickle(fname + ".gz", test_object)
		assert os.path.exists(fname + ".gz"), "File does not exist"

		# Recover the object again:
		recovered_object = loadPickle(fname + ".gz")
		print(recovered_object)
		assert test_object == recovered_object, "The object was not recovered"

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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import sys
import os
import numpy as np
import tempfile
from lightkurve import LightCurve
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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
		rms = rms_timescale(LightCurve(time=time*np.nan, flux=flux))

	# Test with timescale longer than timespan should return zero:
	flux = np.random.randn(1000)
	time = np.linspace(0, 27, len(flux))
	rms = rms_timescale(LightCurve(time=time, flux=flux), timescale=30.0)
	print(rms)
	np.testing.assert_allclose(rms, 0)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import tempfile
import lightkurve as lk
import conftest # noqa: F401
from starclass import io

#--------------------------------------------------------------------------------------------------
def test_pickle():

	# Pretty random object to save and restore:
	test_object = {'test': 42, 'whatever': (1,2,3,4,5432), 'something': 'string'}

	with tempfile.TemporaryDirectory() as tmpdir:
		fname = os.path.join(tmpdir, 'test.pickle')

		# Save object in file:
		io.savePickle(fname, test_object)
		assert os.path.exists(fname), "File does not exist"

		# Recover the object again:
		recovered_object = io.loadPickle(fname)
		print(recovered_object)
		assert test_object == recovered_object, "The object was not recovered"

		# Save object in file:
		io.savePickle(fname + ".gz", test_object)
		assert os.path.exists(fname + ".gz"), "File does not exist"

		# Recover the object again:
		recovered_object = io.loadPickle(fname + ".gz")
		print(recovered_object)
		assert test_object == recovered_object, "The object was not recovered"

#--------------------------------------------------------------------------------------------------
def test_json():

	# Pretty random object to save and restore:
	# NOTE: Using list instead of tuple here, since it is not preserved by JSON
	test_object = {'test': 42, 'whatever': [1,2,3,4,5432], 'something': 'string'}

	with tempfile.TemporaryDirectory() as tmpdir:
		fname = os.path.join(tmpdir, 'test.json')

		# Save object in file:
		io.saveJSON(fname, test_object)
		assert os.path.exists(fname), "File does not exist"

		# Recover the object again:
		recovered_object = io.loadJSON(fname)
		print(recovered_object)
		assert test_object == recovered_object, "The object was not recovered"

		# Save object in file:
		io.saveJSON(fname + ".gz", test_object)
		assert os.path.exists(fname + ".gz"), "File does not exist"

		# Recover the object again:
		recovered_object = io.loadJSON(fname + ".gz")
		print(recovered_object)
		assert test_object == recovered_object, "The object was not recovered"

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('fname,mission', [
	['tess00029281992-s01-c1800-dr01-v04-tasoc-cbv_lc.fits.gz', 'TESS'],
	['kplr001864183-2012179063303_llc.fits.gz', 'Kepler'],
])
def test_load_lightcurve(SHARED_INPUT_DIR, fname, mission):

	fpath = os.path.join(SHARED_INPUT_DIR, 'create_todolist', fname)
	assert os.path.isfile(fpath), "File does not exist"

	lc = io.load_lightcurve(fpath)

	assert isinstance(lc, lk.LightCurve)
	assert lc.mission == mission

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

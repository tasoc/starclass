#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import tempfile
import conftest # noqa: F401
from starclass.io import savePickle, loadPickle

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
if __name__ == '__main__':
	pytest.main([__file__])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of MetaClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os
from astropy.table import Table
import conftest # noqa: F401
from starclass import MetaClassifier, TaskManager
from starclass.training_sets.testing_tset import testing_tset

#--------------------------------------------------------------------------------------------------
def test_metaclassifier_import():
	tset = testing_tset()
	with MetaClassifier(tset=tset) as cl:
		assert(cl.__class__.__name__ == 'MetaClassifier')

#--------------------------------------------------------------------------------------------------
def test_metaclassifier_load_star(PRIVATE_INPUT_DIR):

	# Use the following training set as input:
	tset = testing_tset()

	# Set a dummy features cache inside the private input dir:
	features_cache_name = 'features_cache'
	features_cache = os.path.join(PRIVATE_INPUT_DIR, features_cache_name)
	os.makedirs(features_cache, exist_ok=True)

	with TaskManager(PRIVATE_INPUT_DIR, classes=tset.StellarClasses) as tm:
		for k in range(2): # Try loading twice - second time we should load from cache
			with MetaClassifier(tset=tset, features_cache=features_cache) as cl:
				# Check that the second time there should still be nothing in the cache:
				assert len(os.listdir(features_cache)) == 0

				task = tm.get_task(priority=17, classifier='meta')
				feat = cl.load_star(task)
				print(feat)

				# Check "transfered" features:
				assert feat['priority'] == 17
				assert feat['priority'] == task['priority']
				assert feat['starid'] == task['starid']

				# Check the complex objects:
				assert 'other_classifiers' in feat
				assert isinstance(feat['other_classifiers'], Table)

				# For the MetaClassifier these things should not be included:
				assert 'lightcurve' not in feat
				assert 'powerspectrum' not in feat
				assert 'frequencies' not in feat

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

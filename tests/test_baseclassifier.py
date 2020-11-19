#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of BaseClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
from lightkurve import TessLightCurve
from astropy.table import Table
import numpy as np
import conftest # noqa: F401
from starclass import BaseClassifier, TaskManager, get_trainingset
from starclass.features.powerspectrum import powerspectrum

#--------------------------------------------------------------------------------------------------
def test_baseclassifier_import():
	with BaseClassifier() as cl:
		assert(cl.__class__.__name__ == 'BaseClassifier')

#--------------------------------------------------------------------------------------------------
def test_baseclassifier_import_exceptions(SHARED_INPUT_DIR):

	with pytest.raises(ValueError):
		BaseClassifier(features_cache=os.path.join(SHARED_INPUT_DIR, 'does-not-exist'))

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('linfit', [False, True])
def test_baseclassifier_load_star(PRIVATE_INPUT_DIR, linfit):

	# Use the following training set as input:
	tsetclass = get_trainingset('keplerq9v3')
	tset = tsetclass(linfit=linfit)

	# Set a dummy features cache inside the private input dir:
	features_cache = os.path.join(PRIVATE_INPUT_DIR, 'features_cache')
	os.makedirs(features_cache, exist_ok=True)

	# The features cache should be empty to begin with:
	assert len(os.listdir(features_cache)) == 0

	with TaskManager(PRIVATE_INPUT_DIR) as tm:
		for k in range(2): # Try loading twice - second time we should load from cache
			with BaseClassifier(tset=tset, features_cache=features_cache) as cl:
				# Check that the second time there is something in the features cache:
				if k > 0:
					assert os.listdir(features_cache) == ['features-17.pickle']

				task = tm.get_task(priority=17)
				print(task)

				fname = os.path.join(PRIVATE_INPUT_DIR, 'tess00029281992-s01-c1800-dr01-v04-tasoc-cbv_lc.fits.gz')

				feat = cl.load_star(task, fname)
				print(feat)

				# Check the complex objects:
				assert isinstance(feat['lightcurve'], TessLightCurve)
				assert isinstance(feat['powerspectrum'], powerspectrum)
				assert isinstance(feat['frequencies'], Table)

				# Check "transfered" features:
				assert feat['priority'] == 17
				assert feat['priority'] == task['priority']
				assert feat['starid'] == task['starid']
				assert feat['tmag'] == task['tmag']
				assert feat['variance'] == task['variance']
				assert feat['rms_hour'] == task['rms_hour']
				assert feat['ptp'] == task['ptp']

				# Check FliPer:
				assert np.isfinite(feat['Fp07'])
				assert np.isfinite(feat['Fp7'])
				assert np.isfinite(feat['Fp20'])
				assert np.isfinite(feat['Fp50'])
				assert np.isfinite(feat['FpWhite'])
				assert np.isfinite(feat['Fphi'])
				assert np.isfinite(feat['Fplo'])

				# Check frequencies:
				freqtab = feat['frequencies']
				for k in np.unique(freqtab['num']):
					assert np.isfinite(feat['freq%d' % k]) or np.isnan(feat['freq%d' % k]), "Invalid frequency"
					assert np.isfinite(feat['amp%d' % k]) or np.isnan(feat['amp%d' % k]), "Invalid amplitude"
					assert np.isfinite(feat['phase%d' % k]) or np.isnan(feat['phase%d' % k]), "Invalid phase"

					peak = freqtab[(freqtab['num'] == k) & (freqtab['harmonic'] == 0)]
					np.testing.assert_allclose(feat['freq%d' % k], peak['frequency'])
					np.testing.assert_allclose(feat['amp%d' % k], peak['amplitude'])
					np.testing.assert_allclose(feat['phase%d' % k], peak['phase'])

				# Check details about lightkurve object:
				lc = feat['lightcurve']
				lc.show_properties()
				assert lc.targetid == feat['starid']
				assert lc.label == 'TIC %d' % feat['starid']
				assert lc.mission == 'TESS'
				assert lc.time_format == 'btjd'
				assert lc.time_format == 'btjd'
				assert lc.camera == 1
				assert lc.ccd == 4
				assert lc.sector == 1

				# When running with linfit enabled, the features should contain
				# an extra set of coefficients from the detrending:
				if linfit:
					assert 'detrend_coeff' in feat
					assert len(feat['detrend_coeff']) == 2
					assert np.all(np.isfinite(feat['detrend_coeff']))

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

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
from starclass import BaseClassifier, TaskManager, get_trainingset, STATUS
from starclass.StellarClasses import StellarClassesLevel1
from starclass.features.powerspectrum import powerspectrum
from starclass.plots import plt, plots_interactive
from starclass.training_sets.testing_tset import testing_tset

#--------------------------------------------------------------------------------------------------
def test_baseclassifier_import():
	tset = testing_tset() # Just to suppress warning
	with BaseClassifier(tset=tset) as cl:
		assert(cl.__class__.__name__ == 'BaseClassifier')

#--------------------------------------------------------------------------------------------------
def test_baseclassifier_import_exceptions(SHARED_INPUT_DIR):
	tset = testing_tset() # Just to suppress warning
	with pytest.raises(ValueError):
		BaseClassifier(tset=tset, features_cache=os.path.join(SHARED_INPUT_DIR, 'does-not-exist'))

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('linfit', [False, True])
@pytest.mark.parametrize('fake_metaclassifier', [False, True])
def test_baseclassifier_load_star(PRIVATE_INPUT_DIR, linfit, fake_metaclassifier):

	# Use the following training set as input:
	tsetclass = get_trainingset()
	tset = tsetclass(linfit=linfit)
	tset.fake_metaclassifier = fake_metaclassifier

	# Set a dummy features cache inside the private input dir:
	features_cache_name = 'features_cache'
	if linfit:
		features_cache_name += '_linfit'
	features_cache = os.path.join(PRIVATE_INPUT_DIR, features_cache_name)
	os.makedirs(features_cache, exist_ok=True)

	# The features cache should be empty to begin with:
	assert len(os.listdir(features_cache)) == 0

	with TaskManager(PRIVATE_INPUT_DIR, classes=StellarClassesLevel1) as tm:

		# Create fake results from all classifiers:
		# This is needed for the meta-classifier results to be returned
		if fake_metaclassifier:
			for classifier in tm.all_classifiers:
				tm.save_results({
					'priority': 17,
					'classifier': classifier,
					'status': STATUS.OK,
					'starclass_results': {
						StellarClassesLevel1.SOLARLIKE: 0.2,
						StellarClassesLevel1.DSCT_BCEP: 0.1,
						StellarClassesLevel1.ECLIPSE: 0.7
					}
				})

		for k in range(2): # Try loading twice - second time we should load from cache
			with BaseClassifier(tset=tset, features_cache=features_cache) as cl:
				# Check that the second time there is something in the features cache:
				if k > 0:
					if fake_metaclassifier:
						assert len(os.listdir(features_cache)) == 0
					else:
						assert os.listdir(features_cache) == ['features-17.pickle']

				clfier = 'meta' if fake_metaclassifier else None
				task = tm.get_task(
					priority=17,
					classifier=clfier,
					change_classifier=False,
					chunk=1)[0]
				print(task)
				assert task is not None, "Task not found"

				feat = cl.load_star(task)
				print(feat)

				# Check basic identifiers:
				assert feat['priority'] == 17
				assert feat['priority'] == task['priority']
				assert feat['starid'] == task['starid']

				if fake_metaclassifier:
					# Check the complex objects:
					assert 'lightcurve' not in feat, "lightcurve should not be available"
					assert 'powerspectrum' not in feat, "powerspectrum should not be available"
					assert 'frequencies' not in feat, "frequencies should not be available"
					assert isinstance(feat['other_classifiers'], Table), "other_classifiers should be a Table"

					# Linfit-related parameters:
					assert 'detrend_coeff' not in feat
				else:
					# Check the complex objects:
					assert isinstance(feat['lightcurve'], TessLightCurve)
					assert isinstance(feat['powerspectrum'], powerspectrum)
					assert isinstance(feat['frequencies'], Table)
					assert 'other_classifiers' not in feat, "other_classifiers not be available"

					# Check "transfered" features:
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
					else:
						assert 'detrend_coeff' not in feat

#--------------------------------------------------------------------------------------------------
def test_linfit(PRIVATE_INPUT_DIR):

	fname = os.path.join(PRIVATE_INPUT_DIR, 'tess00029281992-s01-c1800-dr01-v04-tasoc-cbv_lc.fits.gz')

	# Use the following training set as input:
	tsetclass = get_trainingset()
	tset = tsetclass(linfit=True)

	with BaseClassifier(tset=tset) as cl:
		# This is only used to easier load the original lightcurve:
		task = {
			'priority': 1,
			'starid': 29281992,
			'tmag': None,
			'variance': None,
			'rms_hour': None,
			'ptp': None,
			'other_classifiers': None,
			'lightcurve': fname
		}
		feat = cl.load_star(task)
		lc = feat['lightcurve']
		p_rem = feat['detrend_coeff']

		# Remove any trend from the lightcurve:
		indx = np.isfinite(lc.time) & np.isfinite(lc.flux) & np.isfinite(lc.flux_err)
		mintime = np.nanmin(lc.time[indx])
		lc -= np.polyval(p_rem, lc.time - mintime)

		# Insert a new known trend in the lightcurve:
		p_ins = [500, 1234]
		time_orig = lc.time
		lintrend_input = np.polyval(p_ins, lc.time - mintime)
		lc.flux += lintrend_input

		# Save the modified lightcurve to a file:
		fname_modified = fname.replace('.fits.gz', '.txt')
		with open(fname_modified, 'wt') as fid:
			for k in range(len(lc)):
				fid.write("{0:.12f}  {1:.18e}  {2:.18e}\n".format(lc.time[k], lc.flux[k], lc.flux_err[k]))

		# Now load the modified
		task['lightcurve'] = fname_modified
		feat = cl.load_star(task)
		lc = feat['lightcurve']
		psd2 = feat['powerspectrum']
		p = feat['detrend_coeff']
		print(p)

		lintrend_recovered = np.polyval(p, lc.time - mintime)

		psd = powerspectrum(lc)

		# Create debugging figure:
		fig, (ax1, ax2) = plt.subplots(2, figsize=(12,12))
		ax1.plot(lc.time, lc.flux, lw=0.5, label='Original')
		ax1.plot(time_orig, lintrend_input, lw=0.5, label='Input')
		ax1.plot(lc.time, lintrend_recovered, lw=0.5, label='Recovered')
		ax1.legend()
		ax2.plot(psd.standard[0], psd.standard[1], lw=0.5, label='Original')
		ax2.plot(psd2.standard[0], psd2.standard[1], lw=0.5, label='Detrended')
		ax2.set_yscale('log')
		ax2.legend()

		# Make sure we recover the trend that we put in:
		np.testing.assert_allclose(p, p_ins)

		# Compare the power spectra:
		np.testing.assert_allclose(psd.standard[0], psd2.standard[0])
		assert np.all(psd2.standard[1][0:2] < psd.standard[1][0:2])

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	plots_interactive()
	pytest.main([__file__])
	plt.show()

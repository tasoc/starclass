#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of BaseClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
from lightkurve import TessLightCurve
import numpy as np
import conftest # noqa: F401
from starclass import BaseClassifier, TaskManager
from starclass.features.powerspectrum import powerspectrum

#----------------------------------------------------------------------
def test_baseclassifier_import():
	with BaseClassifier() as cl:
		assert(cl.__class__.__name__ == 'BaseClassifier')

#----------------------------------------------------------------------
def test_baseclassifier_load_star(PRIVATE_INPUT_DIR):
	with TaskManager(PRIVATE_INPUT_DIR) as tm:
		with BaseClassifier() as cl:

			task = tm.get_task(priority=17)
			print(task)

			fname = os.path.join(PRIVATE_INPUT_DIR, 'tess00029281992-s01-c1800-dr01-v04-tasoc-cbv_lc.fits.gz')

			feat = cl.load_star(task, fname)
			print(feat)

			# Check the complex objects:
			assert isinstance(feat['lightcurve'], TessLightCurve)
			assert isinstance(feat['powerspectrum'], powerspectrum)

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
			for k in range(1, 7):
				assert np.isfinite(feat['freq%d' % k]) or np.isnan(feat['freq%d' % k])
				assert np.isfinite(feat['amp%d' % k]) or np.isnan(feat['amp%d' % k])
				assert np.isfinite(feat['phase%d' % k]) or np.isnan(feat['phase%d' % k])

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

#----------------------------------------------------------------------
if __name__ == '__main__':
	test_baseclassifier_import()
	test_baseclassifier_load_star()

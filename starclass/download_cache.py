#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download any missing data files to cache.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
from astropy.utils import iers
from .constants import trainingset_list
from .convenience import get_trainingset

def download_cache(trainingsets=None):
	"""
	Download any missing data files to cache.

	This will download all axillary files used by Astropy or our code itself
	to the cache. If all the necessary files already exists, nothing will be done.
	It can be a good idea to call this function before starting starclass
	in parallel on many machines sharing the same cache, in which case the processes
	will all attempt to download the cache files and may conflict with each other.

	Parameters:
		trainingsets (list, optional):

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# This will download IERS data needed for astropy.Time transformations:
	# https://docs.astropy.org/en/stable/utils/iers.html
	# Ensure that the auto_download config is enabled, otherwise nothing will be downloaded
	logger.info("Downloading IERS data...")
	oldval = iers.conf.auto_download
	try:
		iers.conf.auto_download = True
		iers.IERS_Auto().open()
	finally:
		iers.conf.auto_download = oldval

	# Download trainingsets:
	if trainingsets is None:
		download_tsets = ['keplerq9v3']
	elif trainingsets == 'all':
		download_tsets = trainingset_list
	else:
		download_tsets = trainingsets

	for tskey in download_tsets:
		logger.info("Downloading %s training set...", tskey)
		tsetclass = get_trainingset(tskey)
		tsetclass().close()

	logger.info("All cache data downloaded.")

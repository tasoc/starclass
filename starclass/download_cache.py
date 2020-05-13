#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download any missing data files to cache.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
from astropy.utils.iers import IERS_Auto
from .training_sets import keplerq9, keplerq9v2

def download_cache():
	"""
	Download any missing data files to cache.

	This will download all axillary files used by Astropy or our code itself
	to the cache. If all the necessary files already exists, nothing will be done.
	It can be a good idea to call this function before starting the photometry
	in parallel on many machines sharing the same cache, in which case the processes
	will all attempt to download the cache files and may conflict with each other.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# This will download IERS data needed for astropy.Time transformations:
	# https://docs.astropy.org/en/stable/utils/iers.html
	logger.info("Downloading IERS data...")
	IERS_Auto().open()

	logger.info("Downloading KeplerQ9 training set...")
	keplerq9()

	logger.info("Downloading KeplerQ9v2 training set...")
	keplerq9v2()

	logger.info("All cache data downloaded.")

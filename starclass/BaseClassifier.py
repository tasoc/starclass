#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The basic stellar classifier class for the TASOC pipeline.
All other specific stellar classification algorithms will inherit from BaseClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import numpy as np
import os.path
import logging
import six.moves.cPickle as pickle
from lightkurve import TessLightCurve
from astropy.stats import mad_std
from .StellarClasses import StellarClasses
from .features.freqextr import freqextr
from .features.fliper import FliPer
from .features.powerspectrum import powerspectrum

__docformat__ = 'restructuredtext'

class BaseClassifier(object):
	"""
	The basic stellar classifier class for the TASOC pipeline.
	All other specific stellar classification algorithms will inherit from BaseClassifier.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, level='L1', features_cache=None, plot=False):
		"""
		Initialize the classifier object.

		Parameters:
			plot (boolean, optional): Create plots as part of the output. Default is ``False``.

		Attributes:
			plot (boolean): Indicates wheter plotting is enabled.
			data_dir (string): Path to directory where classifiers store auxiliary data.
				Different directories will be used for each classification level.
		"""

		# Check the input:
		assert level in ('L1', 'L2'), "Invalid level"

		# Start logger:
		logger = logging.getLogger(__name__)

		# Store the input:
		self.plot = plot
		self.level = level
		self.features_cache = features_cache
		self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', level))

		if self.features_cache is not None and not os.path.exists(self.features_cache):
			raise ValueError("features_cache directory does not exists")

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()

	def close(self):
		"""Close the classifier."""
		pass

	def classify(self, features):
		"""
		Classify a star from the lightcurve and other features.

		Will run the :py:func:`do_classify` method and
		check some of the output and calculate various
		performance metrics.

		Parameters:
			features (dict): Dictionary of features, including the lightcurve itself.

		Returns:
			dict: Dictionary of classifications

		See Also:
			:py:func:`do_classify`

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		res = self.do_classify(features)
		# Check results
		for key, value in res.items():
			if key not in StellarClasses:
				raise ValueError("Classifier returned unknown stellar class.")
			if value < 0 or value > 1:
				raise ValueError("Classifier should return probability between 0 and 1.")

		return res

	def do_classify(self, features):
		"""
		This method should be overwritten by child classes.

		Parameters:
			features (dict): Dictionary of features of star, including the lightcurve itself.

		Returns:
			dict: Dictionary where the keys should be from ``StellarClasses`` and the
			corresponding values indicate the probability of the star belonging to
			that class.

		Raises:
			NotImplementedError
		"""
		raise NotImplementedError()

	def train(self, features, labels):
		"""
		Parameters:
			features (iterable of dict): Features of star, including the lightcurve itself.
			labels (ndarray, [n_objects]): labels for training set lightcurves.

		Raises:
			NotImplementedError
		"""
		raise NotImplementedError()

	def load_star(self, task, fname):
		"""Recieve a task from the TaskManager and load the lightcurve."""

		logger = logging.getLogger(__name__)

		# Load lightcurve file and create a TessLightCurve object:
		if fname.endswith('.noisy') or fname.endswith('.sysnoise'):
			data = np.loadtxt(fname)
			lightcurve = TessLightCurve(
				time=data[:,0],
				flux=data[:,1],
				flux_err=data[:,2],
				quality=np.asarray(data[:,3], dtype='int32'),
				time_format='jd',
				time_scale='tdb',
				ticid=task['starid'],
				camera=task['camera'],
				ccd=task['ccd'],
				sector=2,
				#ra=0,
				#dec=0,
				quality_bitmask=2+8+256 # lightkurve.utils.TessQualityFlags.DEFAULT_BITMASK
			)
		else:
			raise ValueError("Invalid file format")

		# Load features from cache file, or calculate them
		# and put them into cache file for other classifiers
		# to use later on:
		features = None
		loaded_from_cache = False
		if self.features_cache:
			features_file = os.path.join(self.features_cache, 'features-' + str(task['priority']) + '.pickle')
			if os.path.exists(features_file):
				loaded_from_cache = True
				with open(features_file, 'rb') as fid:
					features = pickle.load(fid)

		# No features found in cache, so calculate them:
		if features is None:
			features = self.calc_features(lightcurve)
			logger.debug(features)

		# Add the fields from the task to the list of features:
		features.update(task)

		# Save features in cache file for later use:
		if self.features_cache and not loaded_from_cache:
			with open(features_file, 'wb') as fid:
				pickle.dump(features, fid)

		return features

	def calc_features(self, lightcurve):
		"""Calculate other derived features from the lightcurve."""

		# We start out with an empty list of features:
		features = {}

		# Add the lightcurve as a seperate feature:
		features['lightcurve'] = lightcurve

		# Prepare lightcurve for power spectrum calculation:
		lc = (lightcurve.remove_nans().normalize() - 1.0) * 1e6
		#lc = lc.remove_outliers(5.0, stdfunc=mad_std) # Sigma clipping

		# Calculate power spectrum:
		psd = powerspectrum(lc)

		# Save the entire power spectrum object in the features:
		features['powerspectrum'] = psd

		# Extract primary frequencies from lightcurve and add to features:
		features.update(freqextr(lightcurve))

		# Calculate FliPer features:
		features.update(FliPer(psd))

		return features
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
from lightkurve import TessLightCurve

__docformat__ = 'restructuredtext'

class BaseClassifier(object):
	"""
	The basic stellar classifier class for the TASOC pipeline.
	All other specific stellar classification algorithms will inherit from BaseClassifier.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, plot=False):
		"""
		Initialize the classifier object.

		Parameters:
			plot (boolean, optional): Create plots as part of the output. Default is ``False``.

		Attributes:
			plot (boolean): Indicates wheter plotting is enabled.
			data_dir (string): Path to directory where classifiers store auxiliary data.
		"""

		# Store the input:
		self.plot = plot

		self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))


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
			lightcurve (``lightkurve.TessLightCurve`` object): Lightcurve.
			features (dict): Dictionary of other features.

		Returns:
			dict: Dictionary of classifications

		See Also:
			:py:func:`do_classify`

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		res = self.do_classify(features)
		# Check results
		return res

	def do_classify(self, features):
		"""
		This method should be overwritten by child classes.

		Parameters:
			features (dict): Features of star, including the lightcurve itself.

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

		# Load features from cache file, or calculate them
		# and put them into cache file for other classifiers
		# to use later on:
		features = self.calc_features(lightcurve)

		features['lightcurve'] = lightcurve

		return features

	def calc_features(self, lightcurve):
		"""Calculate other derived features from the lightcurve."""

		# TODO: This has to actually do something useful!
		return {
			'freq1': np.nan,
			'freq2': np.nan,
			'freq3': np.nan,
			'freq4': np.nan,
			'freq5': np.nan,
			'freq6': np.nan,
			'amp1': np.nan,
			'amp2': np.nan,
			'amp3': np.nan,
			'amp4': np.nan,
			'amp5': np.nan,
			'amp6': np.nan,
			'phase1': np.nan,
			'phase2': np.nan,
			'phase3': np.nan,
			'phase4': np.nan,
			'phase5': np.nan,
			'phase6': np.nan,
		}

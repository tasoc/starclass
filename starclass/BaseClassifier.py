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
		"""

		# Store the input:
		self.plot = plot

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()

	def close(self):
		"""Close the classifier."""
		pass

	def classify(self, lightcurve, features):
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
		res = self.do_classify(lightcurve, features)
		# Check results
		return res

	def do_classify(self, lightcurve, features):
		"""
		This method should be overwritten by child classes.

		Raises:
			NotImplementedError
		"""
		raise NotImplementedError()

	def train(self, lightcurves, labels):
		"""
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
		features = {}

		return lightcurve, features
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
import sqlite3
from lightkurve import TessLightCurve
from .features.freqextr import freqextr
from .features.fliper import FliPer

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

		self.conn = self.cursor = None
		if self.features_cache is not None:
			self.conn = sqlite3.connect(self.features_cache)
			self.conn.row_factory = sqlite3.Row
			self.cursor = self.conn.cursor()
			#self.cursor.execute("DROP TABLE IF EXISTS features;")
			self.cursor.execute("""CREATE TABLE IF NOT EXISTS features (
				priority INT PRIMARY KEY NOT NULL,
				freq1 REAL, freq2 REAL, freq3 REAL, freq4 REAL, freq5 REAL, freq6 REAL,
				amp1 REAL, amp2 REAL, amp3 REAL, amp4 REAL, amp5 REAL, amp6 REAL,
				phase1 REAL, phase2 REAL, phase3 REAL, phase4 REAL, phase5 REAL, phase6 REAL,
				Fp07 REAL, Fp7 REAL, Fp20 REAL, Fp50 REAL
			);""")


	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()
		if self.cursor: self.cursor.close()
		if self.conn: self.conn.close()

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
			self.cursor.execute("SELECT * FROM features WHERE priority=?;", (task['priority'],))
			features = self.cursor.fetchone()
			if features is not None:
				features = dict(features)
				loaded_from_cache = True

		# No features found in database, so calculate them:
		if features is None:
			features = self.calc_features(lightcurve)
			logger.debug(features)

		# Add the fields from the task to the list of features:
		features.update(task)

		# Save features in cache file for later use:
		if self.features_cache and not loaded_from_cache:
			# Save features in database:
			self.cursor.execute("""INSERT INTO features (
				priority,
				freq1,freq2,freq3,freq4,freq5,freq6,
				amp1,amp2,amp3,amp4,amp5,amp6,
				phase1,phase2,phase3,phase4,phase5,phase6,
				Fp07,Fp7,Fp20,Fp50
			) VALUES (
				:priority,
				:freq1,:freq2,:freq3,:freq4,:freq5,:freq6,
				:amp1,:amp2,:amp3,:amp4,:amp5,:amp6,
				:phase1,:phase2,:phase3,:phase4,:phase5,:phase6,
				:Fp07,:Fp7,:Fp20,:Fp50
			);""", features)
			self.conn.commit()

		# Add the lightcurve as a seperate feature:
		features['lightcurve'] = lightcurve

		return features

	def calc_features(self, lightcurve):
		"""Calculate other derived features from the lightcurve."""

		# We start out with an empty list of features:
		features = {}

		# Extract primary frequencies from lightcurve and add to features:
		features.update(freqextr(lightcurve))

		# Calculate FliPer features:
		features.update(FliPer(lightcurve))

		return features
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The meta-classifier.

.. codeauthor:: James S. Kuszlewicz <kuszlewicz@mps.mpg.de>
"""

import logging
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .. import BaseClassifier, utilities

#--------------------------------------------------------------------------------------------------
class Classifier_obj(RandomForestClassifier):
	"""
	Wrapper for sklearn RandomForestClassifier.
	"""
	def __init__(self, n_estimators=100, min_samples_split=2):
		super().__init__(
			n_estimators=n_estimators,
			min_samples_split=min_samples_split,
			class_weight='balanced',
			max_depth=3
		)
		self.trained = False

#--------------------------------------------------------------------------------------------------
class MetaClassifier(BaseClassifier):
	"""
	The meta-classifier.

	.. codeauthor:: James S. Kuszlewicz <kuszlewicz@mps.mpg.de>
	"""

	def __init__(self, clfile='meta_classifier.pickle', featdir='', *args, **kwargs):
		"""
		Initialise the classifier object.

		Parameters:
			clfile (str): Filepath to previously pickled Classifier_obj
			featfile (str):	Filepath to pre-calculated features, if available.
		"""
		# Initialise parent
		super().__init__(*args, **kwargs)

		# Start logger:
		logger = logging.getLogger(__name__)

		self.clfile = clfile
		self.classifier = None

		if clfile is not None:
			self.clfile = os.path.join(self.data_dir, clfile)
		else:
			self.clfile = None

		# Check if pre-trained classifier exists
		if self.clfile is not None and os.path.exists(self.clfile):
			# Load pre-trained classifier
			self.load(self.clfile)

		# Check for features TODO: THIS NEEDS TO BE CHANGED!
		if featdir is not None:
			logger.warning("This needs to be edited when we know how the features will be parsed!")
			self.featdir = os.path.join(self.data_dir, featdir)
			if not os.path.exists(self.featdir):
				os.makedirs(self.featdir)
		else:
			logger.error("No features detected!")
			raise ValueError("Exiting.")

		# Set up classifier
		if self.classifier is None:
			self.classifier = Classifier_obj()

		self.indiv_classifiers = ['rfgc', 'SLOSH', 'xgb']

	#----------------------------------------------------------------------------------------------
	def save(self, outfile):
		"""
		Saves the classifier object with pickle.
		"""
		utilities.savePickle(outfile, self.classifier)

	#----------------------------------------------------------------------------------------------
	def load(self, infile, somfile=None):
		"""
		Loads classifier object.
		"""
		self.classifier = utilities.loadPickle(infile)

	#----------------------------------------------------------------------------------------------
	def do_classify(self, features):
		"""
		Classify a single lightcurve.

		Parameters:
			features (dict): Dictionary of features.

		Returns:
			dict: Dictionary of stellar classifications. -10 for NA results.
		"""
		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		#update to access lightcurve id parameter, if exists
		#logger.info("Object ID: "+str(lightcurve.id))

		if not self.classifier.trained:
			logger.error('Classifier has not been trained. Exiting.')
			raise ValueError('Classifier has not been trained. Exiting.')

		logger.debug("Importing features...")
		featarray = np.array(features['other_classifiers']['prob']).reshape(1,-1)

		logger.debug("We are starting the magic...")
		# Comes out with shape (1,8), but instead want shape (8,) so squeeze
		classprobs = self.classifier.predict_proba(featarray).squeeze()
		logger.debug("Classification complete")

		result = {}
		for c, cla in enumerate(self.classifier.classes_):
			key = self.StellarClasses(cla)
			result[key] = classprobs[c]
		return result

	#----------------------------------------------------------------------------------------------
	def train(self, tset, savecl=True, recalc=False, overwrite=False):
		"""
		Train the classifier.
		Assumes lightcurve time is in days


		"""
		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		# Check for pre-calculated features
		fitlabels = self.parse_labels(tset.labels())

		logger.info("Importing features...")
		# This bit is hardcoded! Not good for generalisability!
		for idx, i in enumerate(tset.features()):
			if idx == 0:
				features = np.array(i['other_classifiers']['prob'])
				preds = np.array(i['other_classifiers']['class'])
			else:
				features = np.vstack((features, np.array(i['other_classifiers']['prob'])))
				preds = np.vstack((preds, np.array(i['other_classifiers']['class'])))

		logger.info("Features imported. Shape = %s", np.shape(features))

		self.classifier.oob_score = True
		logger.info("Fitting model.")
		self.classifier.fit(features, fitlabels)
		logger.info('Trained. OOB Score = %s', self.classifier.oob_score_)
		self.classifier.trained = True

		if savecl and self.classifier.trained and self.clfile is not None:
			if overwrite or recalc or not os.path.exists(self.clfile):
				logger.info("Saving pickled classifier instance to '%s'", self.clfile)
				self.save(self.clfile)

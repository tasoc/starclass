#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The meta-classifier.

.. codeauthor:: James S. Kuszlewicz <kuszlewicz@mps.mpg.de>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import os
import numpy as np
import itertools
from bottleneck import allnan, anynan
from sklearn.ensemble import RandomForestClassifier
from .. import BaseClassifier, utilities
from ..constants import classifier_list

#--------------------------------------------------------------------------------------------------
class Classifier_obj(RandomForestClassifier):
	"""
	Wrapper for sklearn RandomForestClassifier.
	"""
	def __init__(self, n_estimators=100, min_samples_split=2, random_state=None):
		super().__init__(
			n_estimators=n_estimators,
			min_samples_split=min_samples_split,
			class_weight='balanced',
			max_depth=3,
			random_state=random_state
		)
		self.trained = False

#--------------------------------------------------------------------------------------------------
class MetaClassifier(BaseClassifier):
	"""
	The meta-classifier.

	Attributes:
		clfile (str): Path to the file where the classifier is saved.
		classifier (:class:`Classifier_obj`): Actual classifier object.
		features_used (list): List of features used for training.

	.. codeauthor:: James S. Kuszlewicz <kuszlewicz@mps.mpg.de>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, clfile='meta_classifier.pickle', *args, **kwargs):
		"""
		Initialize the classifier object.

		Parameters:
			clfile (str): Filepath to previously pickled Classifier_obj
		"""
		# Initialize parent
		super().__init__(*args, **kwargs)

		self.clfile = None
		self.classifier = None
		self.features_used = None

		if clfile is not None:
			self.clfile = os.path.join(self.data_dir, clfile)

		# Check if pre-trained classifier exists
		if self.clfile is not None and os.path.exists(self.clfile):
			# Load pre-trained classifier
			self.load(self.clfile)

		# Set up classifier
		if self.classifier is None:
			self.classifier = Classifier_obj(random_state=self.random_state)

	#----------------------------------------------------------------------------------------------
	def save(self, outfile):
		"""
		Saves the classifier object with pickle.
		"""
		utilities.savePickle(outfile, [self.classifier, self.features_used])

	#----------------------------------------------------------------------------------------------
	def load(self, infile):
		"""
		Loads classifier object.
		"""
		self.classifier, self.features_used = utilities.loadPickle(infile)

	#----------------------------------------------------------------------------------------------
	def build_features_table(self, features, total=None):
		"""
		Build table of features.

		Parameters:
			features (iterable): Features to build table from.
			total (int, optional): Number of features in ``features``. If not provided,
				the length of ``features`` is found using :func:`len`.

		Returns:
			ndarray: Two dimensional float32 ndarray with probabilities from all classifiers.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		if total is None:
			total = len(features)

		featarray = np.full((total, len(self.features_used)), np.NaN, dtype='float32')
		for k, feat in enumerate(features):
			tab = feat['other_classifiers']
			for j, (classifier, stcl) in enumerate(self.features_used):
				indx = (tab['classifier'] == classifier) & (tab['class'] == stcl)
				if any(indx):
					featarray[k, j] = tab['prob'][indx]

		return featarray

	#----------------------------------------------------------------------------------------------
	def do_classify(self, features):
		"""
		Classify a single lightcurve.

		Parameters:
			features (dict): Dictionary of features.

		Returns:
			dict: Dictionary of stellar classifications.
		"""
		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		if not self.classifier.trained:
			logger.error('Classifier has not been trained. Exiting.')
			raise ValueError('Classifier has not been trained. Exiting.')

		# Build features array from the probabilities from the other classifiers:
		# TODO: What about NaN values?
		logger.debug("Importing features...")
		featarray = self.build_features_table([features], total=1)

		if anynan(featarray):
			raise ValueError("Features contains NaNs")

		logger.debug("We are starting the magic...")
		# Comes out with shape (1,8), but instead want shape (8,) so squeeze
		classprobs = self.classifier.predict_proba(featarray).squeeze()
		logger.debug("Classification complete")

		# Format the output:
		result = {}
		for c, cla in enumerate(self.classifier.classes_):
			key = self.StellarClasses(cla)
			result[key] = classprobs[c]
		return result

	#----------------------------------------------------------------------------------------------
	def train(self, tset, savecl=True, overwrite=False):
		"""
		Train the Meta-classifier.

		Parameters:
			tset (:class:`TrainingSet`): Training set to train classifier on.
			savecl (bool, optional): Save the classifier to file?
			overwrite (bool, optional): Overwrite existing classifer save file.

		.. codeauthor:: James S. Kuszlewicz <kuszlewicz@mps.mpg.de>
		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		# Check for pre-calculated features
		fitlabels = self.parse_labels(tset.labels())

		# First create list of all possible classifiers:
		all_classifiers = list(classifier_list)
		all_classifiers.remove('meta')

		# Create list of all features:
		# Save this to object, we are using it to keep track of which features were used
		# to train the classifier:
		self.features_used = list(itertools.product(all_classifiers, self.StellarClasses))
		#features_names = ['{0:s}_{1:s}'.format(classifier, stcl.name) for classifier, stcl in self.features_used]
		#logger.debug("Feature names: %s", features_names)

		# Create table of features:
		# Create as float32, since that is what RandomForestClassifier converts it to anyway.
		logger.info("Importing features...")
		features = self.build_features_table(tset.features(), total=len(tset.train_idx))

		# Remove columns that are all NaN:
		# This can be classifiers that never returns a given class or a classifier that
		# has not been run at all.
		keepcols = ~allnan(features, axis=0)
		features = features[:, keepcols]
		self.features_used = [x for i, x in enumerate(self.features_used) if keepcols[i]]

		# Throw an error if a classifier is not run at all:
		run_classifiers = set([fu[0] for fu in self.features_used])
		if run_classifiers != set(all_classifiers):
			raise Exception("Classifier did not contribute at all: %s" % set(all_classifiers).difference(run_classifiers))

		# Raise an exception if there are NaNs left in the features:
		if anynan(features):
			raise ValueError("Features contains NaNs")

		logger.info("Features imported. Shape = %s", features.shape)

		# Run actual training:
		self.classifier.oob_score = True
		logger.info("Fitting model.")
		self.classifier.fit(features, fitlabels)
		logger.info('Trained. OOB Score = %s', self.classifier.oob_score_)
		self.classifier.trained = True

		if savecl and self.classifier.trained and self.clfile is not None:
			if overwrite or not os.path.exists(self.clfile):
				logger.info("Saving pickled classifier instance to '%s'", self.clfile)
				self.save(self.clfile)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The meta-classifier.

.. codeauthor:: James S. Kuszlewicz <kuszlewicz@mps.mpg.de>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import itertools
import numpy as np
from bottleneck import allnan, anynan
from sklearn.ensemble import RandomForestClassifier
from .. import BaseClassifier, io
from ..constants import classifier_list
from ..exceptions import UntrainedClassifierError

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
			max_depth=7,
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

		# Link to the internal RandomForestClassifier classifier model,
		# which can be used for calculating feature importances:
		self._classifier_model = self.classifier

	#----------------------------------------------------------------------------------------------
	def save(self, outfile):
		"""
		Saves the classifier object with pickle.
		"""
		io.savePickle(outfile, [self.classifier, self.features_used])

	#----------------------------------------------------------------------------------------------
	def load(self, infile):
		"""
		Loads classifier object.
		"""
		# Load the pickle file:
		self.classifier, self.features_used = io.loadPickle(infile)

		# Extract the features names based on the loaded classifier:
		self.features_names = [f'{classifier:s}_{stcl.name:s}' for classifier, stcl in self.features_used]
		self.logger.debug("Feature names: %s", self.features_names)

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

		Raises:
			UntrainedClassifierError: If classifier has not been trained.
			ValueError: If any features are NaN.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		if not self.classifier.trained:
			raise UntrainedClassifierError('Classifier has not been trained. Exiting.')

		# Build features array from the probabilities from the other classifiers:
		# TODO: What about NaN values?
		self.logger.debug("Importing features...")
		featarray = self.build_features_table([features], total=1)

		if anynan(featarray):
			raise ValueError("Features contains NaNs")

		self.logger.debug("We are starting the magic...")
		# Comes out with shape (1,8), but instead want shape (8,) so squeeze
		classprobs = self.classifier.predict_proba(featarray).squeeze()
		self.logger.debug("Classification complete")

		# Format the output:
		result = {}
		for c, cla in enumerate(self.classifier.classes_):
			key = self.StellarClasses(cla)
			result[key] = classprobs[c]
		return result, featarray

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
		# Check for pre-calculated features
		fitlabels = self.parse_labels(tset.labels())

		# First create list of all possible classifiers:
		all_classifiers = list(classifier_list)
		all_classifiers.remove('meta')

		# Create list of all features:
		# Save this to object, we are using it to keep track of which features were used
		# to train the classifier:
		self.features_used = list(itertools.product(all_classifiers, self.StellarClasses))
		self.features_names = [f'{classifier:s}_{stcl.name:s}' for classifier, stcl in self.features_used]

		# Create table of features:
		# Create as float32, since that is what RandomForestClassifier converts it to anyway.
		self.logger.info("Importing features...")
		features = self.build_features_table(tset.features(), total=len(tset))

		# Remove columns that are all NaN:
		# This can be classifiers that never returns a given class or a classifier that
		# has not been run at all.
		keepcols = ~allnan(features, axis=0)
		features = features[:, keepcols]
		self.features_used = [x for i, x in enumerate(self.features_used) if keepcols[i]]
		self.features_names = [x for i, x in enumerate(self.features_names) if keepcols[i]]

		# Throw an error if a classifier is not run at all:
		run_classifiers = set([fu[0] for fu in self.features_used])
		if run_classifiers != set(all_classifiers):
			raise RuntimeError("Classifier did not contribute at all: %s" % set(all_classifiers).difference(run_classifiers))

		# Raise an exception if there are NaNs left in the features:
		if anynan(features):
			raise ValueError("Features contains NaNs")

		self.logger.info("Features imported. Shape = %s", features.shape)

		# Run actual training:
		self.classifier.oob_score = True
		self.logger.info("Fitting model.")
		self.classifier.fit(features, fitlabels)
		self.logger.info('Trained. OOB Score = %s', self.classifier.oob_score_)
		self.classifier.trained = True

		if savecl and self.classifier.trained and self.clfile is not None:
			if overwrite or not os.path.exists(self.clfile):
				self.logger.info("Saving pickled classifier instance to '%s'", self.clfile)
				self.save(self.clfile)

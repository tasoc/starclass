#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Sorting-Hat Classifier (Supervised randOm foRest variabiliTy classIfier using high-resolution pHotometry Attributtes in TESS data).

.. codeauthor:: Jeroen Audenaert <jeroen.audenaert@kuleuven.be>
"""

import logging
import os.path
import os
from sklearn.ensemble import RandomForestClassifier
from . import Sorting_Hat_featcalc as fc
from .. import BaseClassifier, io

#--------------------------------------------------------------------------------------------------
class Classifier_obj(RandomForestClassifier):
	"""
	Wrapper for sklearn RandomForestClassifier.
	"""
	def __init__(self, n_estimators=1000, max_features='auto', min_samples_split=2, class_weight='balanced', criterion='gini', max_depth=15, random_state=None):
		super().__init__(n_estimators=n_estimators,
			max_features=max_features,
			min_samples_split=min_samples_split,
			class_weight=class_weight, criterion=criterion, max_depth=max_depth,
			random_state=random_state)
		self.trained = False

#--------------------------------------------------------------------------------------------------
class SortingHatClassifier(BaseClassifier):
	"""
	Sorting-Hat Classifier
	"""
	def __init__(self, clfile='sortinghat_classifier_v01.pickle', n_estimators=1000,
					max_features='auto', min_samples_split=2, *args, **kwargs):
		"""
		Initialize the classifier object.

		Parameters:
			clfile (str): Filepath to previously pickled Classifier_obj.
			featfile (str):	Filepath to pre-calculated features, if available.
			n_estimators (int): number of trees in forest
			max_features (int): see sklearn.RandomForestClassifier
			min_samples_split (int): see sklearn.RandomForestClassifier
		"""
		# Initialise parent
		super().__init__(*args, **kwargs)

		self.classifier = None

		if clfile is not None:
			self.clfile = os.path.join(self.data_dir, clfile)
		else:
			self.clfile = None

		if self.features_cache is not None:
			self.featdir = os.path.join(self.features_cache, 'sortinghat_features')
			os.makedirs(self.featdir, exist_ok=True)
		else:
			self.featdir = None

		if self.clfile is not None:
			if os.path.exists(self.clfile):
				# load pre-trained classifier
				self.load(self.clfile)

		self.features_names = fc.feature_names()

		if self.classifier is None:
			# Create new untrained classifier
			self.classifier = Classifier_obj(
				n_estimators=n_estimators,
				max_features=max_features,
				min_samples_split=min_samples_split,
				random_state=self.random_state)

	#----------------------------------------------------------------------------------------------
	def save(self, outfile):
		"""
		Save the classifier object with pickle.
		"""
		io.savePickle(outfile, self.classifier)

	#----------------------------------------------------------------------------------------------
	def load(self, infile):
		"""
		Load classifier object.
		"""
		self.classifier = io.loadPickle(infile)

	#----------------------------------------------------------------------------------------------
	def do_classify(self, features, recalc=False):
		"""
		Classify a single lightcurve.
		Assumes lightcurve time is in days
		Assumes featdict contains ['logf1'],['logf2'],['logf3'], in units of muHz
		Assumes featdict contains ['varrat'],['number_significantharmonic']
		Assumes featdict contains ['skewness'],['flux_ratio']
		Assumes featdict contains ['mse_mean'],['mse_max'],['mse_std'],['mse_power']
		Assumes featdict contains ['diff_entropy_lc'],['diff_entropy_as']

		Parameters:
			lightcurve (``lightkurve.TessLightCurve`` object): Lightcurve.
			featdict (dict): Dictionary of other features.

		Returns:
			dict: Dictionary of stellar classifications. -10 for NA results.
		"""
		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		if not self.classifier.trained:
			logger.error('Classifier has not been trained. Exiting.')
			raise ValueError('Classifier has not been trained. Exiting.')

		# If self.classifier.trained=True, calculate additional features

		logger.debug("Calculating features...")
		featarray = fc.featcalc(features, savefeat=self.featdir, recalc=recalc)
		#logger.info("Features calculated.")

		# Do the magic:
		#logger.info("We are starting the magic...")
		classprobs = self.classifier.predict_proba(featarray)[0]
		logger.debug("Classification complete")

		result = {}
		for c, cla in enumerate(self.classifier.classes_):
			key = self.StellarClasses(cla)
			result[key] = classprobs[c]
		return result, featarray

	#----------------------------------------------------------------------------------------------
	def train(self, tset, savecl=True, recalc=False, overwrite=False):
		"""
		Train the classifier.
		Assumes lightcurve time is in days
		Assumes featdict contains ['logf1'],['logf2'],['logf3'], in units of muHz
		Assumes featdict contains ['varrat'],['number_significantharmonic']
		Assumes featdict contains ['skewness'],['flux_ratio']
		Assumes featdict contains ['mse_mean'],['mse_max'],['mse_std'],['mse_power']
		Assumes featdict contains ['diff_entropy_lc'],['diff_entropy_as']

		Parameters:
			labels (ndarray, [n_objects]): labels for training set lightcurves.
			features (iterable of dict): features, inc lightcurves
			savecl - save classifier? (overwrite or recalc must be true for an old classifier to be overwritten)
			overwrite reruns SOM
			recalc recalculates features

		"""
		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		if self.classifier.trained:
			return

		# Check for pre-calculated features

		fitlabels = self.parse_labels(tset.labels())

		logger.info('Calculating/Loading Features.')
		featarray = fc.featcalc(tset.features(), savefeat=self.featdir, recalc=recalc)
		logger.info('Features calculated/loaded.')

		self.classifier.oob_score = True
		self.classifier.fit(featarray, fitlabels)
		logger.info('Trained. OOB Score = %f', self.classifier.oob_score_)
		#logger.info([estimator.tree_.max_depth for estimator in self.classifier.estimators_])
		self.classifier.oob_score = False
		self.classifier.trained = True

		if savecl and self.classifier.trained:
			if self.clfile is not None:
				if not os.path.exists(self.clfile) or overwrite or recalc:
					logger.info("Saving pickled classifier instance to '%s'", self.clfile)
					self.save(self.clfile)

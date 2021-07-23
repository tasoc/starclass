#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Sorting-Hat Classifier (Supervised randOm foRest variabiliTy classIfier using high-resolution pHotometry Attributtes in TESS data).

.. codeauthor:: Jeroen Audenaert <jeroen.audenaert@kuleuven.be>
"""

import numpy as np
from bottleneck import anynan
import scipy.stats as stat
import logging
import os.path
import os
from sklearn.ensemble import RandomForestClassifier
from . import Sorting_Hat_featcalc as fc
from .. import BaseClassifier, io
from ..utilities import get_periods

# Number of frequencies used as features:
NFREQUENCIES = 3

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

		if self.clfile is not None and os.path.exists(self.clfile):
			# load pre-trained classifier
			self.load(self.clfile)

		self.features_names = ['f' + str(i+1) for i in range(NFREQUENCIES)]
		self.features_names += [
			'varrat',
			'number_significantharmonic',
			'skewness',
			'flux_ratio',
			'diff_entropy_lc',
			'diff_entropy_as',
			'mse_mean',
			'mse_max',
			'mse_std',
			'mse_power'
		]

		if self.classifier is None:
			# Create new untrained classifier
			self.classifier = Classifier_obj(
				n_estimators=n_estimators,
				max_features=max_features,
				min_samples_split=min_samples_split,
				random_state=self.random_state)

		# Link to the internal RandomForestClassifier classifier model,
		# which can be used for calculating feature importances:
		self._classifier_model = self.classifier

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
	def featcalc(self, features, total=None, recalc=False):
		"""
		Calculates features for set of lightcurves
		"""

		if isinstance(features, dict): # trick for single features
			features = [features]
		if total is None:
			total = len(features)

		featout = np.empty([total, len(self.features_names)], dtype='float32')
		for k, obj in enumerate(features):
			# Load features from the provided (cached) features if they exist:
			featout[k, :] = [obj.get(key, np.NaN) for key in self.features_names]

			# If not all features are already populated, we are going to recalculate them all:
			if recalc or anynan(featout[k, :]):
				lc = fc.prepLCs(obj['lightcurve'], linflatten=False)

				periods, _, _ = get_periods(obj, NFREQUENCIES, lc.time, in_days=False)
				featout[k, :NFREQUENCIES] = periods

				#EBper = EBperiod(lc.time, lc.flux, periods[0], linflatten=linflatten-1)
				#featout[k, 0] = EBper # overwrites top period

				featout[k, NFREQUENCIES:NFREQUENCIES+2] = fc.compute_varrat(obj)
				#featout[k, NFREQUENCIES+1:NFREQUENCIES+2] = fc.compute_lpf1pa11(obj)
				featout[k, NFREQUENCIES+2:NFREQUENCIES+3] = stat.skew(lc.flux)
				featout[k, NFREQUENCIES+3:NFREQUENCIES+4] = fc.compute_flux_ratio(lc.flux)
				featout[k, NFREQUENCIES+4:NFREQUENCIES+5] = fc.compute_differential_entropy(lc.flux)
				featout[k, NFREQUENCIES+5:NFREQUENCIES+6] = fc.compute_differential_entropy(obj['powerspectrum'].standard[1])
				featout[k, NFREQUENCIES+6:NFREQUENCIES+10] = fc.compute_multiscale_entropy(lc.flux)
				#featout[k, NFREQUENCIES+10:NFREQUENCIES+11] = fc.compute_max_lyapunov_exponent(lc.flux)

		return featout

	#----------------------------------------------------------------------------------------------
	def do_classify(self, features, recalc=False):
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

		# If self.classifier.trained=True, calculate additional features

		logger.debug("Calculating features...")
		featarray = self.featcalc(features, total=1, recalc=recalc)
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

		Parameters:
			labels (ndarray, [n_objects]): labels for training set lightcurves.
			features (iterable of dict): features, inc lightcurves.
			savecl: save classifier? (overwrite or recalc must be true for an old classifier to be overwritten)
			overwrite: reruns SOM
			recalc: recalculates features

		"""
		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		if self.classifier.trained:
			return

		# Check for pre-calculated features

		fitlabels = self.parse_labels(tset.labels())

		logger.info('Calculating/Loading Features.')
		featarray = self.featcalc(tset.features(), total=len(tset), recalc=recalc)
		logger.info('Features calculated/loaded.')

		self.classifier.oob_score = True
		self.classifier.fit(featarray, fitlabels)
		logger.info('Trained. OOB Score = %f', self.classifier.oob_score_)
		#logger.info([estimator.tree_.max_depth for estimator in self.classifier.estimators_])
		self.classifier.oob_score = False
		self.classifier.trained = True

		if savecl and self.clfile is not None:
			if not os.path.exists(self.clfile) or overwrite or recalc:
				logger.info("Saving pickled classifier instance to '%s'", self.clfile)
				self.save(self.clfile)

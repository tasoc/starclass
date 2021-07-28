#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The RF-GC classifier (general random forest).

.. codeauthor:: David Armstrong <d.j.armstrong@warwick.ac.uk>
"""

import numpy as np
from bottleneck import anynan
import logging
import os.path
import os
import copy
from sklearn.ensemble import RandomForestClassifier
from . import RF_GC_featcalc as fc
from .. import BaseClassifier, io
from ..utilities import get_periods
from ..exceptions import UntrainedClassifierError

# Number of frequencies used as features:
NFREQUENCIES = 6

#--------------------------------------------------------------------------------------------------
class Classifier_obj(RandomForestClassifier):
	"""
	Wrapper for sklearn RandomForestClassifier with attached SOM.
	"""
	def __init__(self, n_estimators=1000, max_features=4, min_samples_split=2, random_state=None):
		super().__init__(n_estimators=n_estimators,
			max_features=max_features,
			min_samples_split=min_samples_split,
			class_weight='balanced', max_depth=15,
			random_state=random_state)
		self.trained = False
		self.som = None

#--------------------------------------------------------------------------------------------------
class RFGCClassifier(BaseClassifier):
	"""
	General Random Forest

	.. codeauthor:: David Armstrong <d.j.armstrong@warwick.ac.uk>
	"""
	def __init__(self, clfile='rfgc_classifier_v01.pickle', somfile='rfgc_som.txt',
		dimx=1, dimy=400, cardinality=64, n_estimators=1000,
		max_features=4, min_samples_split=2, *args, **kwargs):
		"""
		Initialize the classifier object.

		Parameters:
			clfile (str): Filepath to previously pickled Classifier_obj.
			somfile (str): Filepath to trained SOM saved using fc.kohonenSave
			featfile (str):	Filepath to pre-calculated features, if available.
			dimx (int): dimension 1 of SOM in somfile, if given
			dimy (int): dimension 2 of SOM in somfile, if given
			cardinality (int): N bins per SOM pixel in somfile, if given
			n_estimators (int): number of trees in forest
			max_features (int): see sklearn.RandomForestClassifier
			min_samples_split (int): see sklearn.RandomForestClassifier
		"""
		# Initialise parent
		super().__init__(*args, **kwargs)

		self.classifier = None

		if somfile is not None:
			self.somfile = os.path.join(self.data_dir, somfile)
		else:
			self.somfile = None

		if clfile is not None:
			self.clfile = os.path.join(self.data_dir, clfile)
		else:
			self.clfile = None

		if self.clfile is not None:
			if os.path.exists(self.clfile):
				# load pre-trained classifier
				self.load(self.clfile, self.somfile)

		if self.classifier is None:
			self.classifier = Classifier_obj(n_estimators=n_estimators,
				max_features=max_features,
				min_samples_split=min_samples_split,
				random_state=self.random_state)

			if self.classifier.som is None and self.somfile is not None:
				# load som
				if os.path.exists(self.somfile):
					self.classifier.som = fc.loadSOM(self.somfile, random_seed=self.random_seed)

		# List of feature names used by the classifier:
		self.features_names = ['EBperiod']
		self.features_names += ['p' + str(i+1) for i in range(1, NFREQUENCIES)]
		self.features_names += [
			'ampratio21',
			'ampratio31',
			'phasediff21',
			'phasediff31',
			'SOM_map',
			'SOM_range',
			'p2p_98_phasefold',
			'p2p_mean_phasefold',
			'p2p_98_lc',
			'p2p_mean_lc',
			'psi',
			'zc',
			'Fp07',
			'Fp7',
			'Fp20',
			'Fp50'
		]
		if self.linfit:
			self.features_names.append('detrend_coeff_norm')

		# Link to the internal RandomForestClassifier classifier model,
		# which can be used for calculating feature importances:
		self._classifier_model = self.classifier

	#----------------------------------------------------------------------------------------------
	def save(self, outfile, somoutfile='som.txt'):
		"""
		Saves the classifier object with pickle.

		som object saved as this MUST be the one used to train the classifier.
		"""
		fc.kohonenSave(self.classifier.som.K, os.path.join(self.data_dir, somoutfile)) # overwrites
		tempsom = copy.deepcopy(self.classifier.som)
		self.classifier.som = None
		io.savePickle(outfile, self.classifier)
		self.classifier.som = tempsom

	#----------------------------------------------------------------------------------------------
	def load(self, infile, somfile=None):
		"""
		Loads classifier object.

		somfile MUST match the som used to train the classifier.
		"""
		self.classifier = io.loadPickle(infile)

		if somfile is not None and os.path.exists(somfile):
			self.classifier.som = fc.loadSOM(somfile)

		if self.classifier.som is None:
			self.classifier.trained = False

	#--------------------------------------------------------------------------------------------------
	def featcalc(self, features, total=None, cardinality=64, linflatten=False, recalc=False):
		"""
		Calculates features for set features.
		"""

		if isinstance(features, dict): # trick for single features
			features = [features]
		if total is None:
			total = len(features)

		# Loop through the provided features and build feature table:
		featout = np.empty([total, len(self.features_names)], dtype='float32')
		for k, obj in enumerate(features):
			# Load features from the provided (cached) features if they exist:
			featout[k, :] = [obj.get(key, np.NaN) for key in self.features_names]

			# If not all features are already populated, we are going to recalculate them all:
			if recalc or anynan(featout[k, :]):

				lc = fc.prepLCs(obj['lightcurve'], linflatten=linflatten)

				periods, n_usedfreqs, usedfreqs = get_periods(obj, NFREQUENCIES, lc.time, ignore_harmonics=True)
				featout[k, :NFREQUENCIES] = periods

				EBper = fc.EBperiod(lc.time, lc.flux, periods[0], linflatten=True)
				featout[k, 0] = EBper # overwrites top period

				featout[k, NFREQUENCIES:NFREQUENCIES+2] = fc.freq_ampratios(obj, n_usedfreqs, usedfreqs)

				featout[k, NFREQUENCIES+2:NFREQUENCIES+4] = fc.freq_phasediffs(obj, n_usedfreqs, usedfreqs)

				# Self Organising Map
				featout[k, NFREQUENCIES+4:NFREQUENCIES+6] = fc.SOMloc(self.classifier.som, lc.time, lc.flux, EBper, cardinality)

				featout[k, NFREQUENCIES+6:NFREQUENCIES+8] = fc.phase_features(lc.time, lc.flux, EBper)

				featout[k, NFREQUENCIES+8:NFREQUENCIES+10] = fc.p2p_features(lc.flux)

				# Higher Order Crossings:
				psi, zc = fc.compute_hocs(lc.time, lc.flux, 5)
				featout[k, NFREQUENCIES+10:NFREQUENCIES+12] = psi, zc[0]

				# FliPer:
				featout[k, NFREQUENCIES+12:NFREQUENCIES+16] = obj['Fp07'], obj['Fp7'], obj['Fp20'], obj['Fp50']

				# If we are running with linfit enabled, add an extra feature
				# which is the absoulte value of the fitted linear trend, divided
				# with the point-to-point scatter:
				if self.linfit:
					slope_feature = np.abs(obj['detrend_coeff'][0]) / obj['ptp']
					featout[k, NFREQUENCIES+16] = slope_feature

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
			raise UntrainedClassifierError('Classifier has not been trained. Exiting.')

		# Assumes that if self.classifier.trained=True,
		# ...then self.classifier.som is not None

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
			tset (``TrainingSet``): labels for training set lightcurves.
			features (iterable of dict): features, inc lightcurves.
			savecl (bool, optional): Save classifier?
				(``overwrite`` or ``recalc`` must be true for an old classifier to be overwritten).
			overwrite (bool, optional): Reruns SOM.
			recalc (bool, optional): Recalculates features.

		"""
		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		if self.classifier.trained:
			return

		# Check for pre-calculated features

		fitlabels = self.parse_labels(tset.labels())

		logger.info('Calculating features...')

		# Check for pre-calculated som
		if self.classifier.som is None:
			logger.info("No SOM loaded. Creating new SOM, saving to '%s'.", self.somfile)
			self.classifier.som = fc.makeSOM(tset.features(), outfile=self.somfile, overwrite=overwrite, random_seed=self.random_seed)
			logger.info('SOM created and saved.')

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
				logger.info("Saving SOM to '%s'", self.somfile)
				self.save(self.clfile, self.somfile)

	#----------------------------------------------------------------------------------------------
	def loadsom(self, somfile):
		"""
		Loads a SOM, if not done at init.
		"""
		self.classifier.som = fc.loadSOM(somfile)

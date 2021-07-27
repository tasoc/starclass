#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General XGB Classification

.. codeauthor:: Refilwe Kgoadi <refilwe.kgoadi1@my.jcu.edu.au>
"""
import logging
import os
import copy
from xgboost import XGBClassifier as xgb
from . import xgb_feature_calc as xgb_features
from .. import BaseClassifier, io

#--------------------------------------------------------------------------------------------------
class XGBClassifier(BaseClassifier):
	"""
	General XGB Classification

	.. codeauthor:: Refilwe Kgoadi <refilwe.kgoadi1@my.jcu.edu.au>
	"""

	def __init__(self, clfile='xgb_classifier_1.pickle', *args, **kwargs):
		"""
		Initialize the classifier object with optimised parameters.

		Parameters:
			clfile (str): saved classifier file.
			n_estimators (int): number of boosted trees in the ensemble.
			max_depth (int): maximum depth of each tree in the ensemble.
			learning_rate: boosting learning rate.
			reg_alpha: L1 regularization on the features.
			objective: learning objective of the algorithm.
			booster: booster used in the tree.
			eval_metric: Evaluation metric.

		.. codeauthor:: Refilwe Kgoadi <refilwe.kgoadi1@my.jcu.edu.au>
		"""

		# Initialize the parent class:
		super().__init__(*args, **kwargs)

		# Attributes of this classifier:
		self.classifier = None
		self.classifier_file = None

		if clfile is not None:
			self.classifier_file = os.path.join(self.data_dir, clfile)

		if self.classifier_file is not None and os.path.exists(self.classifier_file):
			# Load pre-trained classifier
			self.load(self.classifier_file)
		else:
			# Create new untrained classifier:
			self.classifier = xgb(
				booster='gbtree',
				colsample_bytree=0.7,
				eval_metric='mlogloss',
				gamma=7.5,
				learning_rate=0.1,
				max_depth=6,
				min_child_weight=1,
				n_estimators=500,
				objective='multi:softmax',
				random_state=self.random_seed, # XGBoost uses misleading names
				reg_alpha=1e-5,
				subsample=0.8,
				use_label_encoder=False,
				n_jobs=1
			)
			self.trained = False

		# List of feature names used by the classifier:
		self.features_names = [
			'skewness',
			'kurtosis',
			'shapiro_wilk',
			'eta',
			'PeriodLS',
			'amp1', # Freq_amp_0
			'ampratio21', # Freq_ampratio_21
			'ampratio31', # Freq_ampratio_31
			'phasediff21', # Freq_phasediff_21
			'phasediff31', # Freq_phasediff_31
			'Rcs',
			'psi_Rcs'
		]

		# Link to the internal XBB classifier model,
		# which can be used for calculating feature importances:
		self._classifier_model = self.classifier

	#----------------------------------------------------------------------------------------------
	def save(self, outfile):
		"""
		Save xgb classifier object with pickle
		"""
		#self.classifier = None
		temp_classifier = copy.deepcopy(self.classifier)
		io.savePickle(outfile, self.classifier)
		self.classifier = temp_classifier

	#----------------------------------------------------------------------------------------------
	def load(self, infile):
		"""
		Loading the xgb clasifier
		"""
		self.classifier = io.loadPickle(infile)
		self.trained = True # Assume any classifier loaded is already trained

	#----------------------------------------------------------------------------------------------
	def do_classify(self, features):
		"""
		My classification that will be run on each lightcurve

		Parameters:
			features (dict): Dictionary of other features.

		Returns:
			dict: Dictionary of stellar classifications.
		"""

		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		if not self.trained:
			raise ValueError("Untrained Classifier")

		# If classifer has been trained, calculate features
		logger.debug("Calculating features...")
		feature_results = xgb_features.feature_extract(features, self.features_names, total=1)
		#logger.info('Feature Extraction done')

		# Do the magic:
		xgb_classprobs = self.classifier.predict_proba(feature_results)[0]
		logger.debug("Classification complete")

		class_results = {}
		for k, stcl in enumerate(self.StellarClasses):
			# Cast to float for prediction
			class_results[stcl] = float(xgb_classprobs[k])

		return class_results, feature_results

	#----------------------------------------------------------------------------------------------
	def train(self, tset, savecl=True, recalc=False, overwrite=False, save_feature_importances=True):
		"""
		Training classifier using the ...
		"""

		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		if self.trained:
			return

		logger.info('Calculating/Loading Features.')
		featarray = xgb_features.feature_extract(tset.features(), self.features_names, total=len(tset), recalc=recalc)
		logger.info('Features calculated/loaded.')

		# Convert classification labels to integers:
		intlookup = {key.value: value for value, key in enumerate(self.StellarClasses)}
		fit_labels = [intlookup[lbl] for lbl in self.parse_labels(tset.labels())]

		logger.info('Training ...')
		#logger.info('SHAPES ', str(np.shape(featarray)))
		#logger.info('SHAPES ', str(np.shape(fit_labels)))
		self.classifier.fit(featarray, fit_labels)

		if save_feature_importances:
			importances = self.classifier.feature_importances_.astype(float)
			feature_importances = dict(zip(self.features_names, importances))
			io.saveJSON(os.path.join(self.data_dir, 'xgbClassifier_feature_importances.json'), feature_importances)

		self.trained = True

		if savecl and self.classifier_file is not None:
			if not os.path.exists(self.classifier_file) or overwrite:
				logger.info('Saving pickled xgb classifier to %s', self.classifier_file)
				self.save(self.classifier_file)

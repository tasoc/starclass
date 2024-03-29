#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General XGB Classification

.. codeauthor:: Refilwe Kgoadi <refilwe.kgoadi1@my.jcu.edu.au>
"""

import os
import xgboost as xgb
from . import xgb_feature_calc as xgb_features
from .. import BaseClassifier
from ..exceptions import UntrainedClassifierError

#--------------------------------------------------------------------------------------------------
class XGBClassifier(BaseClassifier):
	"""
	General XGB Classification

	.. codeauthor:: Refilwe Kgoadi <refilwe.kgoadi1@my.jcu.edu.au>
	"""

	def __init__(self, clfile='xgb_classifier.json', *args, **kwargs):
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
			self.classifier = xgb.XGBClassifier(
				booster='gbtree',
				colsample_bytree=0.7,
				eval_metric='mlogloss',
				gamma=7.5,
				learning_rate=0.1,
				max_depth=6,
				min_child_weight=1,
				n_estimators=500,
				objective='multi:softprob',
				num_class=len(self.StellarClasses),
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
		Save xgb classifier object.
		"""
		self.classifier.save_model(outfile)

	#----------------------------------------------------------------------------------------------
	def load(self, infile):
		"""
		Load the xgb clasifier.

		Parameters:
			infile (str): Path to file from which to load the trained XGB classifier model.
		"""
		self.classifier = xgb.XGBClassifier()
		self.classifier.load_model(infile)
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
		if not self.trained:
			raise UntrainedClassifierError("Untrained Classifier")

		# If classifer has been trained, calculate features
		self.logger.debug("Calculating features...")
		feature_results = xgb_features.feature_extract(features, self.features_names, total=1)
		#self.logger.info('Feature Extraction done')

		# Do the magic:
		xgb_classprobs = self.classifier.predict_proba(feature_results)[0]
		self.logger.debug("Classification complete")

		class_results = {}
		for k, stcl in enumerate(self.StellarClasses):
			# Cast to float for prediction
			class_results[stcl] = float(xgb_classprobs[k])

		return class_results, feature_results

	#----------------------------------------------------------------------------------------------
	def train(self, tset, savecl=True, recalc=False, overwrite=False):
		"""
		Training classifier using the ...
		"""
		if self.trained:
			return

		self.logger.info('Calculating/Loading Features.')
		featarray = xgb_features.feature_extract(tset.features(), self.features_names, total=len(tset), recalc=recalc)
		self.logger.info('Features calculated/loaded.')

		# Convert classification labels to integers:
		intlookup = {key.value: value for value, key in enumerate(self.StellarClasses)}
		fit_labels = [intlookup[lbl] for lbl in self.parse_labels(tset.labels())]

		self.logger.info('Training...')
		self.classifier.fit(featarray, fit_labels)
		self.trained = True

		if savecl and self.classifier_file is not None:
			if not os.path.exists(self.classifier_file) or overwrite:
				self.logger.info('Saving xgb classifier to %s', self.classifier_file)
				self.save(self.classifier_file)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
General XGB Classification

.. codeauthor:: Refilwe Kgoadi <refilwe.kgoadi1@my.jcu.edu.au>
"""
import logging
import os
import copy
import json
from xgboost import XGBClassifier as xgb
from . import xgb_feature_calc as xgb_features
from .. import BaseClassifier, utilities

#--------------------------------------------------------------------------------------------------
class Classifier_obj(xgb):
	"""
	Wrapper for sklearn XGBClassifier
	"""

	def __init__(self,base_score=0.5, booster='gbtree', colsample_bylevel=1,
		colsample_bytree=0.7, eval_metric='mlogloss', gamma=7.5,
		learning_rate=0.1, max_delta_step=0, max_depth=6,
		min_child_weight=1, missing=None, n_estimators=500, n_jobs=1,
		nthread=None, objective='multi:softmax', random_state=154,
		reg_alpha=1e-5, reg_lambda=1, scale_pos_weight=1, seed=154,
		silent=True, subsample=0.6):

		super().__init__(
			booster=booster,
			eval_metric=eval_metric,
			colsample_bytree=colsample_bytree,
			subsample=subsample,
			gamma=gamma,
			learning_rate=learning_rate,
			max_depth=max_depth,
			n_estimators=n_estimators,
			objective=objective,
			reg_alpha=reg_alpha
		)

		#self.trained = False

#--------------------------------------------------------------------------------------------------
class XGBClassifier(BaseClassifier):
	"""
	General XGB Classification

	.. codeauthor:: Refilwe Kgoadi <refilwe.kgoadi1@my.jcu.edu.au>
	"""

	def __init__(self, clfile='xgb_classifier_1.pickle',
		featdir="xgb_features", n_estimators=500, gamma=7.5,
		min_child_weight=1, subsample=0.8, max_depth=6,
		learning_rate=0.1, reg_alpha=1e-5,
		objective='multi:softmax', colsample_bytree=0.7, random_state=154,
		booster='gbtree', eval_metric='mlogloss', *args, **kwargs):
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
		super(self.__class__, self).__init__(*args, **kwargs)

		# Attributes of this classifier:
		self.classifier = None
		self.classifier_file = None
		self.featdir = None

		if clfile is not None:
			self.classifier_file = os.path.join(self.data_dir, clfile)

		if self.features_cache is not None and featdir is not None:
			self.featdir = os.path.join(self.features_cache, featdir)
			os.makedirs(self.featdir, exist_ok=True)

		if self.classifier_file is not None and os.path.exists(self.classifier_file):
			# Load pre-trained classifier
			self.load(self.classifier_file)
			self.trained = True # Assume any classifier loaded is already trained
		else:
			# Create new untrained classifier:
			self.classifier = Classifier_obj(
				booster=booster,
				colsample_bytree=colsample_bytree,
				eval_metric=eval_metric,
				gamma=gamma,
				learning_rate=learning_rate,
				max_depth=max_depth,
				min_child_weight=min_child_weight,
				n_estimators=n_estimators,
				objective=objective,
				random_state=random_state,
				reg_alpha=reg_alpha,
				subsample=subsample
			)
			self.trained = False

	#----------------------------------------------------------------------------------------------
	def save(self, outfile):
		"""
		Save xgb classifier object with pickle
		"""

		#self.classifier = None
		temp_classifier = copy.deepcopy(self.classifier)
		utilities.savePickle(outfile, self.classifier)
		self.classifier = temp_classifier

	#----------------------------------------------------------------------------------------------
	def load(self, infile):
		"""
		Loading the xgb clasifier
		"""

		self.classifier = utilities.loadPickle(infile)

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
			logger.error('Please train classifer')
			raise ValueError("Untrained Classifier")

		# If classifer has been trained, calculate features
		logger.debug("Calculating features...")
		feature_results = xgb_features.feature_extract(features) # TODO: Come back to this
		#logger.info('Feature Extraction done')

		# Do the magic:
		#logger.info("We are staring the magic...")
		xgb_classprobs = self.classifier.predict_proba(feature_results)[0]
		logger.debug("Classification complete")
		class_results = {}

		for c, cla in enumerate(self.classifier.classes_):
			key = self.StellarClasses(cla)
			# Cast to float for prediction
			class_results[key] = float(xgb_classprobs[c])

		return class_results

	#----------------------------------------------------------------------------------------------
	def train(self, tset, savecl=True, recalc=False, overwrite=False, feat_import=True):
		"""
		Training classifier using the ...
		"""

		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		if self.trained:
			return

		logger.info('Calculating/Loading Features.')
		featarray = xgb_features.feature_extract(tset.features(), savefeat=self.featdir, recalc=recalc)
		logger.info('Features calculated/loaded.')

		#print(list(featarray))
		#featarray = featarray.drop(['LinearTrend', 'PairSlopeTrend'], axis=1)
		#print(list(featarray))
		#featarray.to_csv(self.data_dir+'/features.csv', index=False)

		#if self.feature is not None:
		#	if os.path.exists(self.features_file):
		#		logger.info('Loading features from precalculated file.')
		#		feature_results = pd.read_csv(self.features_file)
		#		precalc = True

		fit_labels = self.parse_labels(tset.labels())

		#if not precalc:
		#	logger.info('Extracting Features ...')
		#	# Calculate features
		#	feature_results = xgb_features.feature_extract(features) ## absolute_import ##
		#	# Save calcualted features
		#	if savefeat:
		#		if self.features_file is not None:
		#			if not os.path.exists(self.features_file) or overwrite:
		#				logger.info('Saving extracted features to feets_features.txt')
		#				feature_results.to_csv(self.features_file, index=False)
		try:
			logger.info('Training ...')
			#logger.info('SHAPES ', str(np.shape(featarray)))
			#logger.info('SHAPES ', str(np.shape(fit_labels)))
			self.classifier.fit(featarray, fit_labels)
			if feat_import:
				importances = self.classifier.feature_importances_.astype(float)
				feature_importances = zip(list(featarray), importances)
				with open(self.data_dir+'/xgbClassifier_feat_import.json', 'w') as outfile:
					json.dump(list(feature_importances), outfile)

			self.trained = True
		except:
			logger.exception('Training error ...')

		if savecl and self.trained:
			if self.classifier_file is not None:
				if not os.path.exists(self.classifier_file) or overwrite:
					logger.info('Saving pickled xgb classifier to '+self.classifier_file)
					self.save(self.classifier_file)
					#self.save_model(self.classifier_file)

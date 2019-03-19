#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The meta-classifier.

.. codeauthor::  James S. Kuszlewicz <kuszlewicz@mps.mpg.de>
"""

from __future__ import division, print_function, with_statement, absolute_import
import logging
import os.path
import numpy as np
import os
import pickle
import itertools
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from .. import BaseClassifier, StellarClasses
from .. import utilities

class Classifier_obj(RandomForestClassifier):
    """
    Wrapper for sklearn RandomForestClassifier
    """
	def __init__(self, n_estimators=1000, min_samples_split=2):
		super(self.__class__, self).__init__(n_estimators=n_estimators,
										min_samples_split=min_samples_split,
										class_weight='balanced')
		self.trained = False

class xgb_Classifier_obj(XGBClassifier):
    """
	Wrapper for sklearn XGBClassifier

    """

    def __init__(self,base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, eval_metric='mlogloss', gamma=0,
       learning_rate=0.1, max_delta_step=0, max_depth=13,
       min_child_weight=1, missing=None, n_estimators=550, n_jobs=1,
       nthread=None, objective='multi:softmax', random_state=0,
       reg_alpha=1e-05, reg_lambda=1, scale_pos_weight=1, seed=125,
       silent=True, subsample=1):

        super(self.__class__, self).__init__(booster=booster,eval_metric=eval_metric,
             learning_rate=learning_rate, max_depth=max_depth,n_estimators=n_estimators,
             objective=objective,reg_alpha=reg_alpha)

        self.trained = False

class MetaClassifier(BaseClassifier):
    """
    The meta-classifier.

    .. codeauthor::  James S. Kuszlewicz <kuszlewicz@mps.mpg.de>
    """
    def __init__(self, clfile='meta_classifier.pickle', featdir='',
                       *args, **kwargs):
        """
        Initialise the classifier object.

        Parameters:
            clfile (str): Filepath to previously pickled Classifier_obj
			featfile (str):	Filepath to pre-calculated features, if available.
        """
        # Initialise parent
		super(self.__class__, self).__init__(*args, **kwargs)

		self.classifier = None

		# Check if pre-trained classifier exists
		if self.clfile is not None:
			if os.path.exists(self.clfile):
				#load pre-trained classifier
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



		self.class_keys = {}
		self.class_keys['RRLyr/Ceph'] = StellarClasses.RRLYR_CEPHEID
		self.class_keys['transit/eclipse'] = StellarClasses.ECLIPSE
		self.class_keys['solar'] = StellarClasses.SOLARLIKE
		self.class_keys['dSct/bCep'] = StellarClasses.DSCT_BCEP
		self.class_keys['gDor/spB'] = StellarClasses.GDOR_SPB
		self.class_keys['transient'] = StellarClasses.TRANSIENT
		self.class_keys['contactEB/spots'] = StellarClasses.CONTACT_ROT
		self.class_keys['aperiodic'] = StellarClasses.APERIODIC
		self.class_keys['constant'] = StellarClasses.CONSTANT
		self.class_keys['rapid'] = StellarClasses.RAPID

	def save(self, outfile):
		"""
		Saves the classifier object with pickle.
		"""
		utilities.savePickle(outfile,self.classifier)

	def load(self, infile, somfile=None):
		"""
		Loads classifier object.
		"""
		self.classifier = utilities.loadPickle(infile)


	def do_classify(self, features, recalc=False):
		"""
		Classify a single lightcurve.
		Assumes lightcurve time is in days
		Assumes featdict contains ['freq1'],['freq2']...['freq6'] in units of muHz
		Assumes featdict contains ['amp1'],['amp2'],['amp3']
									(amplitudes not amplitude ratios)
		Assumes featdict contains ['phase1'],['phase2'],['phase3']
									(phases not phase differences)

		Parameters:
			lightcurve (``lightkurve.TessLightCurve`` object): Lightcurve.
			featdict (dict): Dictionary of other features.

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

		# Assumes that if self.classifier.trained=True,
		# ...then self.classifier.som is not None

		logger.info("Importing features...")
		logger.error("Not yet implemented!")
		sys.exit()
		logger.info("Features imported.")

		# Do the magic:
		logger.info("We are starting the magic...")
		classprobs = self.classifier.predict_proba(featarray)[0]
		logger.info("Classification complete")

		result = {}
		for c, cla in enumerate(self.classifier.classes_):
			key = self.class_keys[cla]
			result[key] = classprobs[c]
		return result

	def train(self, features, labels, savecl=True, recalc=False, overwrite=False):
		"""
		Train the classifier.
		Assumes lightcurve time is in days


		"""
		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		# Check for pre-calculated features

		fitlabels = self.parse_labels(labels)

		logger.info("Importing features...")
		logger.error("Not yet implemented!")
		sys.exit()
		logger.info("Features imported.")

		try:
			self.classifier.oob_score = True
			self.classifier.fit(featarray, fitlabels)
			logger.info('Trained. OOB Score = ' + str(self.classifier.oob_score_))
			self.classifier.oob_score = False
			self.classifier.trained = True
		except:
			logger.exception('Training Error') # add more details...

		if savecl and self.classifier.trained:
			if self.clfile is not None:
				if not os.path.exists(self.clfile) or overwrite or recalc:
					logger.info('Saving pickled classifier instance to rfgc_classifier_v01.pickle')
					self.save(self.clfile)


	def parse_labels(self,labels,removeduplicates=False):
		"""
		"""
		fitlabels = []
		for lbl in labels:
			if removeduplicates:
				#is it multi-labelled? In which case, what takes priority?
				#or duplicate it once for each label
				if len(lbl)>1:#Priority order loosely based on signal clarity
					if StellarClasses.ECLIPSE in lbl:
						fitlabels.append('transit/eclipse')
					elif StellarClasses.RRLYR_CEPHEID in lbl:
						fitlabels.append('RRLyr/Ceph')
					elif StellarClasses.CONTACT_ROT in lbl:
						fitlabels.append('contactEB/spots')
					elif StellarClasses.DSCT_BCEP in lbl:
						fitlabels.append('dSct/bCep')
					elif StellarClasses.GDOR_SPB in lbl:
						fitlabels.append('gDor/spB')
					elif StellarClasses.SOLARLIKE in lbl:
						fitlabels.append('solar')
					else:
						fitlabels.append(lbl[0].value)
				else:
					#then convert to str
					fitlabels.append(lbl[0].value)
			else:
				fitlabels.append(lbl[0].value)
		return np.array(fitlabels)

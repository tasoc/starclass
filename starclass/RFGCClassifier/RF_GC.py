#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The RF-GC classifier (general random forest).

.. codeauthor::  David Armstrong <d.j.armstrong@warwick.ac.uk>
"""

import logging
import os.path
import numpy as np
import os
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from . import RF_GC_featcalc as fc
from .. import BaseClassifier, StellarClasses
from .. import utilities

class Classifier_obj(RandomForestClassifier):
	"""
	Wrapper for sklearn RandomForestClassifier with attached SOM.
	"""
	def __init__(self, n_estimators=1000, max_features=4, min_samples_split=2):
		super(self.__class__, self).__init__(n_estimators=n_estimators,
										max_features=max_features,
										min_samples_split=min_samples_split,
										class_weight='balanced', max_depth=15)
		self.trained = False
		self.som = None

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
		super(self.__class__, self).__init__(*args, **kwargs)

		self.classifier = None

		if somfile is not None:
			self.somfile = os.path.join(self.data_dir, somfile)
		else:
			self.somfile = None

		if clfile is not None:
			self.clfile = os.path.join(self.data_dir, clfile)
		else:
			self.clfile = None

		if self.features_cache is not None:
			self.featdir = os.path.join(self.features_cache, 'rfgc_features')
			os.makedirs(self.featdir, exist_ok=True)
		else:
			self.featdir = None

		if self.clfile is not None:
			if os.path.exists(self.clfile):
				#load pre-trained classifier
				self.load(self.clfile, self.somfile)

		if self.classifier is None:
			self.classifier = Classifier_obj(n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split)
			if self.classifier.som is None and self.somfile is not None:
				#load som
				if os.path.exists(self.somfile):
					self.classifier.som = fc.loadSOM(self.somfile)


	def save(self, outfile, somoutfile='som.txt'):
		"""
		Saves the classifier object with pickle.

		som object saved as this MUST be the one used to train the classifier.
		"""
		fc.kohonenSave(self.classifier.som.K,os.path.join(self.data_dir,somoutfile)) #overwrites
		tempsom = copy.deepcopy(self.classifier.som)
		self.classifier.som = None
		utilities.savePickle(outfile, self.classifier)
		self.classifier.som = tempsom


	def load(self, infile, somfile=None):
		"""
		Loads classifier object.

		somfile MUST match the som used to train the classifier.
		"""
		self.classifier = utilities.loadPickle(infile)

		if somfile is not None:
			if os.path.exists(somfile):
				self.classifier.som = fc.loadSOM(somfile)

		if self.classifier.som is None:
		    self.classifier.trained = False


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

		if not self.classifier.trained:
			logger.error('Classifier has not been trained. Exiting.')
			raise ValueError('Classifier has not been trained. Exiting.')

		# Assumes that if self.classifier.trained=True,
		# ...then self.classifier.som is not None

		logger.info("Calculating features...")
		featarray = fc.featcalc(features, self.classifier.som, savefeat=self.featdir, recalc=recalc)
		#logger.info("Features calculated.")

		# Do the magic:
		#logger.info("We are starting the magic...")
		classprobs = self.classifier.predict_proba(featarray)[0]
		logger.info("Classification complete")

		result = {}
		for c, cla in enumerate(self.classifier.classes_):
			key = StellarClasses(cla)
			result[key] = classprobs[c]
		return result


	def train(self, tset, savecl=True, recalc=False, overwrite=False):
		"""
		Train the classifier.
		Assumes lightcurve time is in days
		Assumes featdict contains ['freq1'],['freq2']...['freq6'] in units of muHz
		Assumes featdict contains ['amp1'],['amp2'],['amp3']
									(amplitudes not amplitude ratios)
		Assumes featdict contains ['phase1'],['phase2'],['phase3']
									(phases not phase differences)

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

		logger.info('Calculating features...')

		# Check for pre-calculated som
		if self.classifier.som is None:
			logger.info("No SOM loaded. Creating new SOM, saving to '%s'.", self.somfile)
			#make copy of features iterator
			self.classifier.som = fc.makeSOM(tset.features(), outfile=self.somfile, overwrite=overwrite)
			logger.info('SOM created and saved.')
			logger.info('Calculating/Loading Features.')
			featarray = fc.featcalc(tset.features(), self.classifier.som, savefeat=self.featdir, recalc=recalc)
		else:
			logger.info('Calculating/Loading Features.')
			featarray = fc.featcalc(tset.features(), self.classifier.som, savefeat=self.featdir, recalc=recalc)
		logger.info('Features calculated/loaded.')

		try:
			self.classifier.oob_score = True
			self.classifier.fit(featarray, fitlabels)
			logger.info('Trained. OOB Score = %f', self.classifier.oob_score_)
			#logger.info([estimator.tree_.max_depth for estimator in self.classifier.estimators_])
			self.classifier.oob_score = False
			self.classifier.trained = True
		except:
			logger.exception('Training Error') # add more details...

		if savecl and self.classifier.trained:
			if self.clfile is not None:
				if not os.path.exists(self.clfile) or overwrite or recalc:
					logger.info("Saving pickled classifier instance to '%s'", self.clfile)
					logger.info("Saving SOM to '%s'", self.somfile)
					self.save(self.clfile, self.somfile)


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


	def loadsom(self, somfile, dimx=1, dimy=400, cardinality=64):
		"""
		Loads a SOM, if not done at init.
		"""
		self.classifier.som = fc.loadSOM(somfile, dimx, dimy, cardinality)

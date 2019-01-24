#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The RF-GC classifier (general random forest).

.. codeauthor::  David Armstrong <d.j.armstrong@warwick.ac.uk>
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
										class_weight='balanced',max_depth=15)
		self.trained = False
		self.som = None

class RFGCClassifier(BaseClassifier):
	"""
	General Random Forest

	.. codeauthor:: David Armstrong <d.j.armstrong@warwick.ac.uk>
	"""
	def __init__(self, clfile='rfgc_classifier_v01.pickle', somfile='som.txt',
					featdir='rfgc_features',
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

		if not os.path.exists(self.data_dir):
			os.makedirs(self.data_dir)

		if somfile is not None:
			self.somfile = os.path.join(self.data_dir, somfile)
		else:
			self.somfile = None

		if clfile is not None:
			self.clfile = os.path.join(self.data_dir, clfile)
		else:
			self.clfile = None

		if featdir is not None:
			self.featdir = os.path.join(self.data_dir, featdir)
			if not os.path.exists(self.featdir):
			    os.makedirs(self.featdir)
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


	def save(self, outfile, somoutfile='som.txt'):
		"""
		Saves the classifier object with pickle.

		som object saved as this MUST be the one used to train the classifier.
		"""
		fc.kohonenSave(self.classifier.som.K,os.path.join(self.data_dir,somoutfile)) #overwrites
		tempsom = copy.deepcopy(self.classifier.som)
		self.classifier.som = None
		utilities.savePickle(outfile,self.classifier)
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

		#update to access lightcurve id parameter, if exists
		#logger.info("Object ID: "+str(lightcurve.id))

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
			key = self.class_keys[cla]
			result[key] = classprobs[c]
		return result

	def train(self, features, labels, savecl=True, recalc=False, overwrite=False):
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

		# Check for pre-calculated features

		fitlabels = self.parse_labels(labels)

		logger.info('Calculating features...')

		# Check for pre-calculated som
		if self.classifier.som is None:
			logger.info('No SOM loaded. Creating new SOM, saving to ''som.txt''.')
			#make copy of features iterator
			features1,features2 = itertools.tee(features,2)
			self.classifier.som = fc.makeSOM(features1, outfile=os.path.join(self.data_dir, 'som.txt'), overwrite=overwrite)
			logger.info('SOM created and saved.')
			logger.info('Calculating/Loading Features.')
			featarray = fc.featcalc(features2, self.classifier.som, savefeat=self.featdir, recalc=recalc)
		else:
			logger.info('Calculating/Loading Features.')
			featarray = fc.featcalc(features, self.classifier.som, savefeat=self.featdir, recalc=recalc)
		logger.info('Features calculated/loaded.')

		try:
			self.classifier.oob_score = True
			self.classifier.fit(featarray, fitlabels)
			logger.info('Trained. OOB Score = ' + str(self.classifier.oob_score_))
			#logger.info([estimator.tree_.max_depth for estimator in self.classifier.estimators_])
			self.classifier.oob_score = False
			self.classifier.trained = True
		except:
			logger.exception('Training Error') # add more details...

		if savecl and self.classifier.trained:
			if self.clfile is not None:
				if not os.path.exists(self.clfile) or overwrite or recalc:
					logger.info('Saving pickled classifier instance to rfgc_classifier_v01.pickle')
					logger.info('Saving SOM to som.txt (will overwrite)')
					self.save(self.clfile,self.somfile)


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


	def plotConfMatrix(self,confmatrix,ticklabels):
		'''
		Plot a confusion matrix. Axes size and labels are hardwired.

		Parameters:
			cfmatrix (ndarray, [nobj x n_classes]): Confusion matrix. Format - np.savetxt
							on the output of self.makeConfMatrix()
			ticklabels (array, [n_classes]): labels for plot axes.
		'''
		import pylab as p
		p.ion()

		norms = np.sum(confmatrix,axis=1)
		for i in range(len(confmatrix[:,0])):
			confmatrix[i,:] /= norms[i]
		p.figure()
		p.clf()
		p.imshow(confmatrix,interpolation='nearest',origin='lower',cmap='YlOrRd')

		for x in range(len(confmatrix[:,0])):
			 for y in range(len(confmatrix[:,0])):
				 if confmatrix[y,x] > 0:
					 if confmatrix[y,x]>0.7:
						 p.text(x,y,str(np.round(confmatrix[y,x],decimals=2)),
						 		va='center',ha='center',color='w')
					 else:
						 p.text(x,y,str(np.round(confmatrix[y,x],decimals=2)),
						 		va='center',ha='center')

		for x in np.arange(confmatrix.shape[0]):
			p.plot([x+0.5,x+0.5],[-0.5,confmatrix.shape[0]-0.5],'k--')
		for y in np.arange(confmatrix.shape[0]):
			p.plot([-0.5,confmatrix.shape[0]-0.5],[y+0.5,y+0.5],'k--')
		p.xlim(-0.5,confmatrix.shape[0]-0.5)
		p.ylim(-0.5,confmatrix.shape[0]-0.5)
		p.xlabel('Predicted Class')
		p.ylabel('True Class')
		#class labels
		p.xticks(np.arange(confmatrix.shape[0]),ticklabels,rotation='vertical')
		p.yticks(np.arange(confmatrix.shape[0]),ticklabels)

	def crossValidate(self, features, labels):
		'''
		Creates cross-validated class probabilities. Splits dataset into groups of 10.
		May take some time.

		Parameters:
			features (ndarray, [n_objects x n_features]): Array of all features.
			labels (ndarray, [n_objects]): labels for each row of features

		Returns:
			cvprobs: (ndarray, [nobjects, nclasses]: cross-validated class probabilities
			classorder (ndarray, [n_classes]): classes corresponding to each column
												of cvprobs
		'''
		shuffleidx = np.random.choice(len(labels),len(labels),replace=False)
		cvfeatures = features[shuffleidx,:]
		cvlabels = labels[shuffleidx]
		kf = KFold(n_splits=int(cvfeatures.shape[0]/10))
		probs = []
		self.classifier.oob_score = False
		for train_index,test_index in kf.split(cvfeatures,cvlabels):
			self.classifier.fit(cvfeatures[train_index,:],cvlabels[train_index])
			sortclasses = np.argsort(self.classifier.classes_)
			tempprobs = self.classifier.predict_proba(cvfeatures[test_index,:])
			probs.append(tempprobs[sortclasses])
		cvprobs = np.vstack(probs)
		unshuffleidx = np.argsort(shuffleidx)
		cvprobs = cvprobs[unshuffleidx]
		return cvprobs,self.classifier.classes_[sortclasses]

	def makeConfMatrix(self,classprobs,correct_labels):
		'''
		Generates a confusion matrix from a set of class probabilities.

		Parameters:
			classprobs (ndarray, [n_objects,n_classes]): Class probabilities
			correct_labels (ndarray, [n_objects]): True labels.

		Returns:
			cfmatrix (ndarray, [nclasses, nclasses]): Confusion matrix.
		'''
		class_vals=np.argmax(classprobs,axis=1)+1
		cfmatrix = confusion_matrix(correct_labels,class_vals)
		return cfmatrix

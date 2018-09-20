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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from . import RF_GC_featcalc as fc
from .. import BaseClassifier, StellarClasses


class Classifier_obj(RandomForestClassifier):
	"""
	Wrapper for sklearn RandomForestClassifier with attached SOM.
	"""
	def __init__(self, n_estimators=1000, max_features=3, min_samples_split=2):
		super(self.__class__, self).__init__(n_estimators=n_estimators,
										max_features=max_features,
										min_samples_split=min_samples_split,
										class_weight='balanced')
		self.trained = False
		self.som = None

class RFGCClassifier(BaseClassifier):
	"""
	General Random Forest

	.. codeauthor:: David Armstrong <d.j.armstrong@warwick.ac.uk>
	"""
	def __init__(self, saved_rf=None, somfile='som.txt', dimx=1, dimy=400, cardinality=64,
		n_estimators=1000, max_features=3, min_samples_split=2, *args, **kwargs):
		"""
		Initialize the classifier object.

		Parameters:
			saved_rf (str): Filepath to previously pickled Classifier_obj.
			somfile (str): Filepath to trained SOM saved using fc.kohonenSave
			dimx (int): dimension 1 of SOM in somfile, if given
			dimy (int): dimension 2 of SOM in somfile, if given
			cardinality (int): N bins per SOM pixel in somfile, if given
			n_estimators (int): number of trees in forest
			max_features (int): see sklearn.RandomForestClassifier
			min_samples_split (int): see sklearn.RandomForestClassifier
		"""
		# Initialise parent
		super(self.__class__, self).__init__(*args, **kwargs)

		if saved_rf is not None and os.path.exists(saved_rf):
			#load pre-trained classifier
			saved_rf = os.path.join(self.data_dir, saved_rf)
			self.load(saved_rf)
		else:
			self.classifier = Classifier_obj(n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split)

		if self.classifier.som is None and somfile is not None:
			#load som
			somfile = os.path.join(self.data_dir, somfile)
			if os.path.exists(somfile):
				self.classifier.som = fc.loadSOM(somfile, dimx, dimy, cardinality)
			
		#if saved_rf is None:
	#		self.save(os.path.join(self.data_dir, 'rfgc_classifier_v01.pickle'))

	def save(self, outfile):
		"""
		Saves the classifier object with pickle.
		"""
		with open(outfile, 'wb') as fid:
			pickle.dump(self.classifier, fid)

	def load(self, infile):
		"""
		Loads classifier object.
		"""
		self.classifier = pickle.load(infile)

	def do_classify(self, features):
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
		featarray = fc.featcalc_single(features, self.classifier.som)
		logger.info("Features calculated.")

		# Do the magic:
		logger.info("We are starting the magic...")
		classprobs = self.classifier.predict_proba(featarray)
		logger.info("Classification complete")

		result = {}
		for c, cla in enumerate(self.classifier.classes_):
			result[cla] = classprobs[c]
		return result

	def train(self, features, labels, featuredat=None, savecl=False, savefeat=False):
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
			featuredat (str): filepath of pre-calculated features, if available.
		"""
		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		# Check for pre-calculated features
		if featuredat is None:
			logger.info('No feature file given. Calculating.')

			# Check for pre-calculated som
			if self.classifier.som is None:
				logger.info('No SOM loaded. Creating new SOM, saving to ''./som.txt''.')
				#dataprep and train SOM. Save to default loc.
				self.classifier.som = fc.trainSOM(features, outfile='som.txt')

			featarray = fc.featcalc_set(features, self.classifier.som)
		else:
			featarray = np.genfromtxt(featuredat)
        
		#for col in np.arange(featarray.shape[1]):
		#    print('Column: '+str(col))
		#    print(np.isfinite(featarray[:,col]).sum(axis=0))
		#print('Total complete rows:')
		#print(np.sum(np.isfinite(featarray).sum(axis=1)==featarray.shape[1]))
		
		fitlabels = self.parse_labels(labels)
		
		if savefeat:
			np.savetxt(os.path.join(self.data_dir, 'rfgc_classifier_feat.txt'),featarray)
            
		try:
			self.classifier.oob_score = True
			self.classifier.fit(featarray, fitlabels)
			logger.info('Trained. OOB Score = ' + str(self.classifier.oob_score_))
			self.classifier.oob_score = False
			self.classifier.trained = True
		except:
			logger.exception('Training Error') # add more details...

		if savecl:
			self.save(os.path.join(self.data_dir, 'rfgc_classifier_v01.pickle'))
        


	def parse_labels(self,labels):
		"""
		"""
		fitlabels = []
		for lbl in labels:
			#is it multi-labelled? In which case, what takes priority?
			#or duplicate it once for each label      
			if len(lbl)>1:#Priority order loosely based on signal clarity
				if StellarClasses.ECLIPSE in lbl: 
					fitlabels.append('ECLIPSE')
				elif StellarClasses.RRLYR_CEPHEID in lbl:
					fitlabels.append('RRLYR_CEPHEID')
				elif StellarClasses.CONTACT_ROT in lbl:
					fitlabels.append('CONTACT_ROT')
				elif StellarClasses.DSCT_BCEP in lbl:
					fitlabels.append('DSCT_BCEP')
				elif StellarClasses.GDOR_SPB in lbl:
					fitlabels.append('GDOR_SPB')
				elif StellarClasses.SOLARLIKE in lbl:
					fitlabels.append('SOLARLIKE')
				else:
					fitlabels.append(lbl[0].value)
			else:
				#then convert to integer
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
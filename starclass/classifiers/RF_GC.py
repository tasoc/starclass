#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The RF-GC classifier (general random forest).

.. codeauthor::  David Armstrong <d.j.armstrong@warwick.ac.uk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import logging
import numpy as np
import os
import pickle
import RF_GC_featcalc as fc
from sklearn.ensemble import RandomForestClassifier
from .. import BaseClassifier, StellarClasses


class Classifier_obj(RandomForestClassifier):

    def __init(self, n_estimators=1000, max_features=3, min_samples_split=2)
        super().__init__(self, n_estimators=n_estimators, 
        								max_features=max_features, 
        								min_samples_split=min_samples_split,
        								class_weight='balanced')
        self.trained = False
        self.som = None								
    
class RFGCClassifier(BaseClassifier):
	"""
	General Random Forest

	.. codeauthor:: David Armstrong <d.j.armstrong@warwick.ac.uk>
	
	No option to internally save/load as yet, but if you pre-calculate features, 
	training is fast.
	Or you can pickle dump/load the class from an external script.
	"""

	def __init__(self, saved_rf=None, somfile=None, dimx=1, dimy=400, cardinality=64,
				  n_estimators=1000, max_features=3, min_samples_split=2, 
				  *args, **kwargs):
		"""
		Initialize the classifier object.
		"""
		# Initialise parent
		super(self.__class__, self).__init__(*args, **kwargs)
		
		if saved_rf is not None and os.path.exists(saved_rf):
		    #load pre-trained classifier
		    self.classifier = pickle.load(saved_rf)
		else:
		    self.classifier = Classifier_obj(n_estimators, max_features, 
		    								  min_samples_split)
		    
        if self.classifier.som is None and somfile is not None:
            #load som
            self.classifier.som = fc.loadSOM(somfile, dimx, dimy, cardinality)
            

    def save(self,outfile):
        pickle.dump(self.classifier,outfile)
        
    def load(self,infile):
        self.classifier = pickle.load(infile)
               
	def do_classify(self, lightcurve, featdict):
		"""
		My classification that will be run on each lightcurve

	    Assumes lightcurves are in days
	    
	    Assumes featdict contains ['freq1'],['freq2'] etc in units of muHz
	    
	    Assumes featdict contains ['amp1'],['amp2'] etc (amplitudes not amplitude ratios)
	    
	    Assumes featdict contains ['phase1'],['phase2'] etc (phases not phase differences)

		Parameters:
			lightcurve (``lightkurve.TessLightCurve`` object): Lightcurve.
			featdict (dict): Dictionary of other features.

		Returns:
			dict: Dictionary of stellar classifications.
		"""
		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)
		
		#update to access lightcurve id parameter, if exists
		#logger.info("Object ID: "+str(lightcurve.id)) 
		
        if not self.classifier.trained:
            logger.error('Classifier has not been trained. Exiting.')
            result = {
			StellarClasses.SOLARLIKE: -10,
			StellarClasses.ECLIPSE: -10,
			StellarClasses.RRCEPH: -10,
			StellarClasses.GDOR_SPB: -10,
			StellarClasses.DSCT_BCEP: -10,
			StellarClasses.TRANSIENT: -10,
			StellarClasses.CONTACT_ROT: -10,
			StellarClasses.APERIODIC: -10,
			StellarClasses.CONSTANT: -10,
			StellarClasses.RAPID: -10
		    }
		    return result
        
        # Assumes that if self.classifier.trained=True, 
        # ...then self.classifier.som is not None
        logger.info("Calculating features...")
        featarray = fc.featcalc_single(lightcurve,featdict,self.classifier.som)
        logger.info("Features calculated.")
        
		# Do the magic:
		logger.info("We are starting the magic...")
		classprobs = self.classifier.predict_proba(featarray)
        logger.info("Classification complete")
        
        result = {}
        for c,cla in emnumerate(self.classifier.classes_)
		    result[cla] = classprobs[c]

		return result

	def train(self, lightcurves, labels, featdict, featuredat=None):
	    """
	    Assumes lightcurves are in days
	    
	    Assumes featdict contains ['freq1'],['freq2'] etc in units of muHz
	    
	    Assumes featdict contains ['amp1'],['amp2'] etc (amplitudes not amplitude ratios)
	    
	    Assumes featdict contains ['phase1'],['phase2'] etc (phases not phase differences)
	    
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
		        self.classifier.som = fc.trainSOM(lightcurves, 
		        									featdict, 
		        									outfile='som.txt')
		            
            features = fc.featcalc_set(lightcurves,featdict,self.classifier.som)
        else:
            features = np.genfromtxt(featuredat)
        
        try:
            self.classifier.oob_score = True
            self.classifier.fit(features,labels)
            logger.info('Trained. OOB Score = '+str(self.classifier.oob_score_))
            self.classifier.oob_score = False
            self.classifier.trained = True
        except:
            logger.error('Training Error') #add more details...

    def loadsom(self,somfile, dimx=1, dimy=400, cardinality=64):
        self.classifier.som = fc.loadSOM(somfile, dimx, dimy, cardinality)   

        
    def plotConfMatrix(self,conffmatrix,ticklabels):
        '''
		Plot a confusion matrix. Axes size and labels are hardwired.
		
		Parameters
		-----------------
		cfmatrix: ndarray
			Confusion matrix. Format - np.savetxt
			on the output of self.makeConfMatrix()
		ticklabels
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
        for y in np.arange(confmatrix.shape[0]:
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
		
		Parameters
		-----------------
		
		
		Returns
		-----------------
		cvprobs: ndarray, size (nobjects, nclasses)
			cross-validated class probabilities one row per object
		classorder:
		
        '''
        from sklearn.model_selection import KFold
        shuffleidx = np.random.choice(len(labels),len(labels),replace=False)
        cvfeatures = features[shuffleidx,:]
        cvlabels = labels[shuffleidx]
        kf = KFold(n_splits=int(cvfeatures.shape[0]/10))
        probs = []
        self.classifier.oob_score = False
        for train_index,test_index in kf.split(cvfeatures,cvlabels):
            self.classifier.fit(cvfeatures[train_index,:],cvlabels[train_index])
            sortclasses = np.argsort(self.classifier.classes))
            tempprobs = self.classifier.predict_proba(cvfeatures[test_index,:])
            probs.append(tempprobs[sortclasses])
        cvprobs = np.vstack(probs)
        unshuffleidx = np.argsort(shuffleidx)
        cvprobs = cvprobs[unshuffleidx]
        return cvprobs,self.classifier.classes_[sortclasses]
    
    def makeConfMatrix(self,classprobs,correct_labels):
        '''
		Generates a confusion matrix from a set of class probabilities.
		
		Parameters
		-----------------
		classprobs: ndarray
			Array of class probabilities, size (nobjects, nclasses)
		
		Returns
		-----------------
		cfmatrix: ndarray, size (nclasses, nclasses)
			Confusion matrix for the classifier. 
        '''
        from sklearn.metrics import confusion_matrix
        class_vals=np.argmax(classprobs,axis=1)+1
        cfmatrix = confusion_matrix(correct_labels,class_vals)
        return cfmatrix
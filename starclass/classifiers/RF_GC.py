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
			features (dict): Dictionary of other features.

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
			StellarClasses.TRANSECLIPSE: -10,
			StellarClasses.RRLYRCEPH = -10,
			StellarClasses.GDOR = -10,
			StellarClasses.DSCT = -10,
			StellarClasses.TRANSIENT = -10,
			StellarClasses.CONTACT_ROT = -10,
			StellarClasses.APERIODIC = -10,
			StellarClasses.CONSTANT = -10,
			StellarClasses.RAPID: -10
		    }
		    return result
        
        # Assumes that if self.classifier.trained=True, 
        # then self.classifier.som is not None
        logger.info("Calculating features...")
        featarray = fc.featcalc_single(lightcurve,featdict,self.classifier.som)
        logger.info("Features calculated.")
        
		# Do the magic:
		logger.info("We are starting the magic...")
		classprobs = self.classifier.predict_proba(featarray)
        logger.info("Classification complete")
        
		# Dummy result where the target is 98% a solar-like
		# and 2% classical pulsator:
		result = {
			StellarClasses.SOLARLIKE: 0.98,
			StellarClasses.TRANSECLIPSE:   ,
			StellarClasses.RRLYRCEPH =   ,
			StellarClasses.GDOR =   ,
			StellarClasses.DSCT =   ,
			StellarClasses.TRANSIENT =   ,
			StellarClasses.CONTACT_ROT =   ,
			StellarClasses.APERIODIC =   ,
			StellarClasses.CONSTANT =   ,
			StellarClasses.RAPID: -10
		}

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
            
        self.classifier.oob_score = True
        self.classifier.fit(features,labels)
        logger.info('Trained. OOB Score = '+str(self.classifier.oob_score_))
        self.classifier.trained = True

    def loadsom(self,somfile, dimx=1, dimy=400, cardinality=64):
        self.classifier.som = fc.loadSOM(somfile, dimx, dimy, cardinality)   

    def defineGroups_tdasim3(self):
        '''
		Hardwired class definitions. Could use updating to be versatile.
				
		Returns
		-----------------
		groups: ndarray, size (nobjects)
        '''
        ids = self.features[:,0]
        groups = np.zeros(len(ids))
        for i in range(len(ids)):
            idx = np.where(self.metaids=='Star'+str(int(ids[i])))[0][0]
            if self.types[idx].strip(' ') == 'Eclipse':
                groups[i] = 1
            elif 'Cepheid' in self.types[idx].strip(' '):
                groups[i] = 2
            elif 'Solar-like' in self.types[idx].strip(' '):
                groups[i] = 3
            elif 'RR Lyrae; RRab' in self.types[idx].strip(' '):
                groups[i] = 4    
            elif 'RR Lyrae; ab type' in self.types[idx].strip(' '):
                groups[i] = 4 
            elif 'RR Lyrae; RRc' in self.types[idx].strip(' '):
                groups[i] = 5
            elif 'RR Lyrae; c type' in self.types[idx].strip(' '):
                groups[i] = 5
            elif 'RR Lyrae; RRd' in self.types[idx].strip(' '):
                groups[i] = 6
            elif 'bCep+SPB hybrid' in self.types[idx].strip(' '):
                groups[i] = 7            
            elif 'bCep' in self.types[idx].strip(' '):
                groups[i] = 8    
            elif 'SPB' in self.types[idx].strip(' '):
                groups[i] = 9
            elif 'bumpy' in self.types[idx].strip(' '):
                groups[i] = 10                                                         
            elif self.types[idx].strip(' ') == 'LPV;MIRA' or self.types[idx].strip(' ') == 'LPV;SR':
                groups[i] = 11
            elif self.types[idx].strip(' ') == 'roAp':
                groups[i] = 12
            elif self.types[idx].strip(' ') == 'Constant':
                groups[i] = 13
            elif 'dSct+gDor hybrid' in self.types[idx].strip(' '): #early to avoid gDor misgrouping
                groups[i] = 0
            elif 'gDor' in self.types[idx].strip(' '):
                groups[i] = 14 
            elif 'dsct' in self.types[idx].strip(' '):
                groups[i] = 15  
        return groups
        
    def plotConfMatrix(self,cfmatrixfile='cfmatrix_noisyv2.txt'):
        '''
		Plot a confusion matrix. Axes size and labels are hardwired. Could use
		updating to be versatile, along with defineGroups(). Defaults to use 
		self.cfmatrix, will use input file if not populated.
		
		Parameters
		-----------------
		cfmatrixfile: str, optional
			Filepath to a txt file containing a confusion matrix. Format - np.savetxt
			on the output of self.makeConfMatrix()
        '''
        import pylab as p
        p.ion()
        if self.cfmatrix is not None:
            confmatrix = self.cfmatrix.astype('float')
        else:
            confmatrix = np.genfromtxt(cfmatrixfile)
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
                         p.text(x,y,str(np.round(confmatrix[y,x],decimals=2)),va='center',ha='center',color='w')
                     else:
                         p.text(x,y,str(np.round(confmatrix[y,x],decimals=2)),va='center',ha='center')

        for x in [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5]:
            p.plot([x,x],[-0.5,14.5],'k--')
        for y in [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5]:
            p.plot([-0.5,14.5],[y,y],'k--')
        p.xlim(-0.5,14.5)
        p.ylim(-0.5,14.5)
        p.xlabel('Predicted Class')
        p.ylabel('True Class')
        #class labels
        p.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],['Eclipse', 'Cepheid', 'Solar', 'RRab', 'RRc', 'RRd', 'b+S hybrid', 'bCep', 'SPB', 'bumpy', ' LPV', 'roAp', 'Const', 'gDor', 'dSct'],rotation='vertical')
        p.yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],['Eclipse', 'Cepheid', 'Solar', 'RRab', 'RRc', 'RRd', 'b+S hybrid', 'bCep', 'SPB', 'bumpy', ' LPV', 'roAp', 'Const', 'gDor', 'dSct'])
    
    def crossValidate(self):
        '''
		Creates cross-validated class probabilities. Splits dataset into groups of 10.
		May take some time.
		
		Returns
		-----------------
		cvprobs: ndarray, size (nobjects, nclasses)
			cross-validated class probabilities one row per object
        '''
        from sklearn.model_selection import KFold
        shuffleidx = np.random.choice(len(self.groups),len(self.groups),replace=False)
        cvfeatures = self.features[shuffleidx,1:]
        cvgroups = self.groups[shuffleidx]
        kf = KFold(n_splits=int(cvfeatures.shape[0]/10))
        probs = []
        self.oob_score = False
        for train_index,test_index in kf.split(cvfeatures,cvgroups):
            print(test_index)
            self.fit(cvfeatures[train_index,:],cvgroups[train_index])
            probs.append(self.predict_proba(cvfeatures[test_index,:]))  
        self.cvprobs = np.vstack(probs)
        unshuffleidx = np.argsort(shuffleidx)
        self.cvprobs = self.cvprobs[unshuffleidx]
        return self.cvprobs
    
    def makeConfMatrix(self,cvprobfile='cvprobs_noisy.txt'):
        '''
		Generates a confusion matrix from a set of class probabilities. Defaults to 
		use self.cvprobs. If not populated, reads from input file.
		
		Parameters
		-----------------
		cvprobfile: str, optional
			Filepath to an array of class probabilities, size (nobjects, nclasses)
		
		Returns
		-----------------
		cfmatrix: ndarray, size (nclasses, nclasses)
			Confusion matrix for the classifier. 
        '''
        if self.cvprobs is None:
            self.cvprobs = np.genfromtxt(cvprobfile)
        from sklearn.metrics import confusion_matrix
        self.class_vals=np.argmax(self.cvprobs,axis=1)+1
        self.cfmatrix = confusion_matrix(self.groups,self.class_vals)
        return self.cfmatrix
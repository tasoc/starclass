#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An example classifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import logging
#import os.path
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier as xgb
from sklearn.metrics import confusion_matrix,accuracy_score
from . import xgb_feature_calc as xgb_features
from .. import BaseClassifier, StellarClasses
from .. import utilities


class Classifier_obj(xgb):
	"""
	Wrapper for sklearn RandomForestClassifier with attached SOM.
	"""
	def __init__(self,  n_estimators=800,max_depth=11,learning_rate = 0.1,reg_alpha=1e-5,):
	
		super(self.__class__, self).__init__(n_estimators=n_estimators,max_depth=max_depth,
                                  learning_rate =learning_rate,reg_alpha=reg_alpha,
                                  objective ='multi:softmax',booster='gbtree',
                                  eval_metric='mlogloss')
		self.trained = False


class XGBClassifier(BaseClassifier):
    """
	XGB Classification 

	.. codeauthor:: Refilwe Kgoadi <refilwe.kgoadi1@my.jcu.edu.au>
	"""
    def __init__(self,classifier_file='xgb_classifier_1.pickle',
                 features_file="feets_features.txt",
                 n_estimators=800,max_depth=11,learning_rate = 0.1,
              reg_alpha=1e-5,objective ='multi:softmax',booster='gbtree', 
              eval_metric='mlogloss', *args, **kwargs):
        """
		Initialize the classifier object.

		Parameters:
			classifier_file (str): svaed classifier file.
			#xgbclassifier_features (str):	Features 
			n_estimators (int): number of boosted trees in the ensemble
			max_depth (int): maximum depth of each tree in the ensemble
			learning_rate=boosting learning rate
            reg_alpha=L1 regulaarization on the features
            objective =learning objective of the algorithm
            booster=booster used in the tree,
            eval_metric=Evaluation metric
		"""
        
        super(self.__class__, self).__init__(*args, **kwargs)
        
        self.classifier = None
        #self.trained = False
        
        if classifier_file is not None:
            self.classifier_file = os.path.join(self.data_dir,classifier_file)
        else:
            self.classifier_file = None
        
        if self.classifier_file is not None:
            if os.path.exists(self.classifier_file):
                # Load pre-trained classifier
                self.load(self.classifier_file)
            
        if features_file is not None:
            self.features_file = os.path.join(self.data_dir, features_file)
        else:
            self.features_file = None
            
        if self.classifier is None:
            self.classifier = xgb(n_estimators=n_estimators,max_depth=max_depth,
                                  learning_rate =learning_rate,reg_alpha=reg_alpha,
                                  objective ='multi:softmax',booster='gbtree',
                                  eval_metric='mlogloss')
            
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
      
#    def save(self, outfile, ) 
   
    def load(self, infile, classifier_file=None):
       """
       Loading the xgb clasifier
       
       """
       self.classifier = utilities.loadPickle(infile)
        
    def do_classify(self, lightcurve, features):
        """
		My classification that will be run on each lightcurve

		Parameters:
			lightcurve (``lightkurve.TessLightCurve`` object): Lightcurve.
			features (dict): Dictionary of other features.

		Returns:
			dict: Dictionary of stellar classifications.
		"""

    	# Start a logger that should be used to output e.g. debug information:
        logger = logging.getLogger(__name__)
        
        # Do the magic:
        #logger.info("We are staring the magic...")
        #self.something.doit(lightcurve, features)
        
        if not self.classifier.trained:
            logger.error('Please train classifer')
            raise ValueError("Untrained Classifier")
            
        ## If classifer has been trained, calculate features
        logger.info('Feature Extraction')
        feature_results = xgb_features.feature_extract_single(features) ## Come back to this 
        logger.info('Feature Extraction done')
        
        # Do the magic:
        logger.info("We are staring the magic...")
        xgb_classprobs = self.classifier.predict_proba(feature_results)[0]
        logger.info('Done')
        
        class_results = {}
        
        for c, cla in enumerate(self.classifier.classes_):
            key = self.class_keys[cla]
            class_results[key] = xgb_classprobs[c]
            
        return class_results
    
    def train(self, features, labels, savecl=True, savefeat=True,overwrite=False):
        """
        Training classifier using the ...
        
        """
        
        # Start a logger that should be used to output e.g. debug information:
        logger = logging.getLogger(__name__)
        
        # Check for pre-calculated features
        precalc = False
        if self.features_file is not None:
            if os.path.exists(self.features_file):
                logger.info('Loading features from precalculated file.')
                feature_results = pd.read_csv(self.features_file)
                precalc = True
                
        fit_labels = self.parse_labels(labels)
        
        if not precalc:
            logger.info('Extracting Features ...')
            # Calculate features
            feature_results = xgb_features.feature_extract(features) ## absolute_import ##
            # Save calcualted features
            if savefeat:
                if self.features_file is not None:
                    if not os.path.exists(self.features_file) or overwrite:
                        logger.info('Saving extracted features to feets_features.txt')
                        np.savetxt(self.features_file,feature_results)
                        
        try:
            logger.info('Training ...')
            self.classifier.fit(feature_results,fit_labels)
            self.classifier.trained = True
        except:
            logger.exception('Training error ...')
            
        if savecl and self.classifier.trained:
            if self.classifier_file is not None:
                if not os.path.exists(self.classifier_file) or overwrite:
                    logger.info('Saving pickled xgb classifier to xgb_classifier_1.pickle')
                    self.save(self.classifier_file)
                    
    def parse_labels(self,labels):
        #""" """
        fit_labels = []
        for label in labels:
            if len(label)>1:#Priority order loosely based on signal clarity
                if StellarClasses.ECLIPSE in label:
                    fit_labels.append('transit/eclipse')
                elif StellarClasses.RRLYR_CEPHEID in label:
                    fit_labels.append('RRLyr/Ceph')
                elif StellarClasses.CONTACT_ROT in label:
                    fit_labels.append('contactEB/spots')
                elif StellarClasses.DSCT_BCEP in label:
                    fit_labels.append('dSct/bCep')
                elif StellarClasses.GDOR_SPB in label:
                    fit_labels.append('gDor/spB')
                elif StellarClasses.SOLARLIKE in label:
                    fit_labels.append('solar')
                else:
                    #then convert to str
                    fit_labels.append(label[0].value)              
            else:
                #then convert to str
                fit_labels.append(label[0].value)              
        return np.array(fit_labels)

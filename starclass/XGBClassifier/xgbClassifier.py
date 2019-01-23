# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import division, print_function, with_statement, absolute_import
import logging
#import os.path
import os
import copy
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier as xgb
from sklearn.metrics import confusion_matrix,accuracy_score
from . import xgb_feature_calc as xgb_features
from .. import BaseClassifier, StellarClasses
from .. import utilities

#nan = np.nan

class Classifier_obj(xgb):
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


class XGBClassifier(BaseClassifier):

    """

    General XGB Classification

	.. codeauthor:: Refilwe Kgoadi <refilwe.kgoadi1@my.jcu.edu.au>

    """
    def __init__(self,clfile='xgb_classifier_1.pickle',
                 features_file="feets_features.csv",n_estimators=750,
                 max_depth=13,learning_rate = 0.1,reg_alpha=1e-5,
                 objective ='multi:softmax',
                 booster='gbtree',eval_metric='mlogloss', *args, **kwargs):

        """

		Initialize the classifier object with optimised parameters.

		Parameters:
			clfile (str): svaed classifier file.
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

        if clfile is not None:
            self.classifier_file = os.path.join(self.data_dir,clfile)
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
            self.classifier = Classifier_obj(#base_score=base_score,
             booster=booster,#colsample_bylevel=colsample_bylevel,
             #colsample_bytree=colsample_bytree,
             eval_metric=eval_metric,
             #gamma=gamma,
             learning_rate=learning_rate,#max_delta_step=max_delta_step,
             max_depth=max_depth, #min_child_weight=min_child_weight,
             #missing=missing,
             n_estimators=n_estimators,#n_jobs=n_jobs,
             #nthread=nthread,
             objective=objective,#random_state=random_state,
             reg_alpha=reg_alpha #reg_lambda=reg_lambda,
             #scale_pos_weight=scale_pos_weight, seed=seed,silent=silent,subsample=subsample
             )

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

        Save xgb classifier object with pickle

        """

        self.classifier = None
        temp_classifier = copy.deepcopy(self.classifier)
        utilities.savePickle(outfile,self.classifier)
        self.classifier = temp_classifier

    def load(self, infile, classifier_file=None):

        """
        Loading the xgb clasifier

        """

        self.classifier = utilities.loadPickle(infile)

    def do_classify(self, features):

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
        feature_results = xgb_features.feature_extract(features) ## Come back to this
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

    def train(self, features, labels, savecl=True,
              savefeat=True,overwrite=False, feat_import=True):

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
                        feature_results.to_csv(self.features_file, index=False)
        try:
            logger.info('Training ...')
            model = self.classifier.fit(feature_results, fit_labels)
            if feat_import == True:
                importances = model.feature_importances_.astype(float)
                feature_importances = zip(list(feature_results),
                                          importances)
                with open('xgbClassifier_feat_import.json', 'w') as outfile:
                    json.dump(list(feature_importances), outfile)

            self.classifier.trained = True
        except:
            logger.exception('Training error ...')

        if savecl and self.classifier.trained:
            if self.classifier_file is not None:
                if not os.path.exists(self.classifier_file) or overwrite:
                    logger.info('Saving pickled xgb classifier to '+self.classifier_file)
                    self.save(self.classifier_file)
                    #self.save_model(self.classifier_file)

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

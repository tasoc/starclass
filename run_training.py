#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility function for running classifiers.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import matplotlib.pyplot as plt
import os.path
import argparse
import logging
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from starclass import training_sets, RFGCClassifier, XGBClassifier, SLOSHClassifier

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Utility function for training stellar classifiers.')
	parser.add_argument('-c', '--classifier', help='Classifier to use.', default='rfgc', choices=('rfgc', 'slosh', 'foptics', 'xgb', 'meta'))
	parser.add_argument('-l', '--level', help='Classification level', default='L1', choices=('L1', 'L2'))
	parser.add_argument('--datalevel', help="", default='corr', choices=('raw', 'corr')) # TODO: Come up with better name than "datalevel"?
	parser.add_argument('-t', '--trainingset', help='Train classifier using this training-set.', default='keplerq9', choices=('tdasim', 'keplerq9', 'keplerq9-linfit'))
	parser.add_argument('-tf', '--testfraction', help='Test-set fraction (only relevant if --train activated)', type=float, default=0.0)
	parser.add_argument('-o', '--overwrite', help='Overwrite existing results.', action='store_true')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	args = parser.parse_args()

	# Check args
	if args.testfraction < 0.0 or args.testfraction > 1:
		parser.error('Testfraction must be between 0 and 1')

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)
	logger_parent = logging.getLogger('starclass')
	logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)

	# Choose which classifier to use
	# For now, there is only one...
	current_classifier = args.classifier
	classifier = {
		'rfgc': RFGCClassifier,
		'slosh': SLOSHClassifier,
		#'foptics': FOPTICSClassifier,
		'xgb': XGBClassifier,
		'meta': None
	}[current_classifier]

	# Pick the training set:
	if args.train == 'tdasim':
		tset = training_sets.tda_simulations(datalevel=args.datalevel, tf=args.testfraction, classifier=current_classifier)
	elif args.train == 'keplerq9':
		tset = training_sets.keplerq9(datalevel=args.datalevel, tf=args.testfraction, classifier=current_classifier)
	elif args.train == 'keplerq9-linfit':
		tset = training_sets.keplerq9linfit(datalevel=args.datalevel, tf=args.testfraction, classifier=current_classifier)

	# Name output classifier file
	clfile = current_classifier + '_' + str(np.round(args.testfraction,decimals=2))

	# Initialize the classifier:
	with classifier(level=args.level, tset=args.train, features_cache=tset.features_cache, clfile=clfile) as stcl:
		# Run the training of the classifier:
		logger.debug("Starting training...")
		stcl.train(tset)
		logger.debug("Training done...")

		if args.testfraction > 0.0:
			logger.debug("Starting testing...")

			# Convert to values
			labels_test_val = []
			for lbl in tset.labels_test(level=args.level):
				labels_test_val.append(lbl[0].value)
			labels_test = np.array(labels_test_val)

			# Classify test set (has to be one by one unless we change classifiers)
			y_pred = []
			for testobj in tset.features_test():
				res = stcl.classify(testobj)
				#logger.info(res)
				prediction = max(res, key=lambda key: res[key]).value
				logger.info(prediction)
				y_pred.append(prediction)
			y_pred = np.array(y_pred)

			# Compare to known labels:
			acc = accuracy_score(labels_test, y_pred)
			logger.info('Accuracy: ', str(acc))
			cf = confusion_matrix(labels_test, y_pred) #labels probably not in right format
			logger.info('CF Matrix:')
			logger.info(cf)



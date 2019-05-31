#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility function for running classifiers.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import argparse
import logging
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from starclass import training_sets, RFGCClassifier, XGBClassifier, SLOSHClassifier, MetaClassifier

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Utility function for training stellar classifiers.')
	parser.add_argument('-c', '--classifier', help='Classifier to use.', default='rfgc', choices=('rfgc', 'slosh', 'foptics', 'xgb', 'meta'))
	parser.add_argument('-l', '--level', help='Classification level', default='L1', choices=('L1', 'L2'))
	parser.add_argument('--datalevel', help="", default='corr', choices=('raw', 'corr')) # TODO: Come up with better name than "datalevel"?
	parser.add_argument('-t', '--trainingset', help='Train classifier using this training-set.', default='keplerq9', choices=('tdasim', 'keplerq9', 'keplerq9-linfit'))
	parser.add_argument('-tf', '--testfraction', help='Test-set fraction', type=float, default=0.0)
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
		'meta': MetaClassifier
	}[current_classifier]

	# Pick the training set:
	if args.trainingset == 'tdasim':
		tset = training_sets.tda_simulations(datalevel=args.datalevel, tf=args.testfraction, classifier=current_classifier)
	elif args.trainingset == 'keplerq9':
		tset = training_sets.keplerq9(tf=args.testfraction, classifier=current_classifier)
	elif args.trainingset == 'keplerq9-linfit':
		tset = training_sets.keplerq9linfit(datalevel=args.datalevel, tf=args.testfraction, classifier=current_classifier)

	# Initialize the classifier:
	with classifier(level=args.level, features_cache=tset.features_cache, tset_key=tset.key) as stcl:
		# Run the training of the classifier:
		logger.info("Starting training...")
		stcl.train(tset)
		logger.info("Training done...")

		if tset.testfraction > 0:
			logger.info("Starting testing...")

			# Convert to values
			labels_test = np.array([lbl[0].value for lbl in tset.labels_test(level=args.level)])

			# Classify test set (has to be one by one unless we change classifiers)
			# TODO: Run in paralllel
			# TODO: Use TaskManager for this?
			y_pred = []
			for features in tset.features_test():
				# Classify this start from the test-set:
				res = stcl.classify(features)

				# TODO: Save results for this classifier/trainingset in database

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

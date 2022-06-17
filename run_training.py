#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line script for training classifiers.

The default is to train the Meta Classifier, which includes training all other classifiers as well.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import starclass

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Utility function for training stellar classifiers.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing results.', action='store_true')
	parser.add_argument('--log', type=str, default=None, metavar='{LOGFILE}', help="Log to file.")
	parser.add_argument('--log-level', type=str, default=None, choices=['debug','info','warning','error'],
		help="Logging level to use in file-logging. If not set, use the same level as the console.")
	parser.add_argument('--clear-cache', help='Clear existing features cache before running.', action='store_true')
	# Option to select which classifier to train:
	parser.add_argument('-c', '--classifier',
		default='meta',
		choices=starclass.classifier_list,
		metavar='{CLASSIFIER}',
		help='Classifier to train. Choises are ' + ", ".join(starclass.classifier_list) + '.')
	# Option to select training set:
	parser.add_argument('-t', '--trainingset',
		default='keplerq9v3',
		choices=starclass.trainingset_list,
		metavar='{TSET}',
		help='Train classifier using this training-set. Choises are ' + ", ".join(starclass.trainingset_list) + '.')

	parser.add_argument('-l', '--level', help='Classification level', default='L1', choices=('L1', 'L2'))
	parser.add_argument('--linfit', help='Enable linfit in training set.', action='store_true')
	#parser.add_argument('--datalevel', help="", default='corr', choices=('raw', 'corr')) # TODO: Come up with better name than "datalevel"?
	parser.add_argument('-tf', '--testfraction', type=float, default=0.0, help='Holdout/test-set fraction')
	parser.add_argument('--output', type=str, default=None, help='Directory where trained models and diagnostics will be saved. Default is to save in the programs data directory.')
	args = parser.parse_args()

	# Check args
	if args.testfraction < 0 or args.testfraction >= 1:
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
	console.setLevel(logging_level)
	logger = logging.getLogger('starclass')
	logger.addHandler(console)

	# Add log-file if the user asked for it:
	if args.log is not None:
		os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
		filehandler = logging.FileHandler(os.path.abspath(args.log), mode='w', encoding='utf8')
		filehandler.setFormatter(formatter)
		filehandler.setLevel(logging_level if args.log_level is None else args.log_level.upper())
		logging_level = min(logging_level, filehandler.level)
		logger.addHandler(filehandler)

	# The logging level of the logger objects needs to be the smallest
	# logging level enabled in either of the handlers:
	logger.setLevel(logging_level)

	# Pick the training set:
	tsetclass = starclass.get_trainingset(args.trainingset)
	tset = tsetclass(level=args.level, tf=args.testfraction, linfit=args.linfit)

	# If we were asked to do so, clear the cache before proceding:
	if args.clear_cache:
		tset.clear_cache()

	# The Meta-classifier requires us to first train all of the other classifiers
	# using cross-validation
	if args.classifier == 'meta':
		# Loop through all the other classifiers and initialize them:
		# TODO: Run in parallel?
		with starclass.TaskManager(tset.todo_file, overwrite=args.overwrite, classes=tset.StellarClasses) as tm:
			# Loop through all classifiers, excluding the MetaClassifier:
			for cla_key in tm.all_classifiers:
				# Check if everything is already populated for this classifier, and if so skip it:
				numtasks = tm.get_number_tasks(classifier=cla_key)
				if numtasks == 0:
					logger.info("Cross-validation results already populated for %s", cla_key)
					continue

				# Split the tset object into cross-validation folds.
				# These are objects with exactly the same properties as the original one,
				# except that they will run through different subsets of the training and test sets:
				cla = starclass.get_classifier(cla_key)
				for tset_fold in tset.folds(n_splits=5):
					with cla(tset=tset_fold, features_cache=tset.features_cache, data_dir=args.output) as stcl:
						logger.info('Training %s on Fold %d/%d...', stcl.classifier_key, tset_fold.fold, tset_fold.crossval_folds)
						stcl.train(tset_fold)
						logger.info("Training done.")
						logger.info("Classifying test-set using %s...", stcl.classifier_key)
						stcl.test(tset_fold, save=tm.save_results)

				# Now train all classifiers on the full training-set (minus the holdout-set),
				# and test on the holdout set:
				with cla(tset=tset, features_cache=tset.features_cache, data_dir=args.output) as stcl:
					logger.info('Training %s on full training-set...', stcl.classifier_key)
					stcl.train(tset)
					logger.info("Training done.")
					logger.info("Classifying holdout-set using %s...", stcl.classifier_key)
					stcl.test(tset, save=tm.save_results, feature_importance=True)

		# For the MetaClassifier, we should switch this on for the final training:
		tset.fake_metaclassifier = True

	# Initialize the classifier:
	classifier = starclass.get_classifier(args.classifier)
	with starclass.TaskManager(tset.todo_file, overwrite=False, classes=tset.StellarClasses) as tm:
		with classifier(tset=tset, features_cache=tset.features_cache, data_dir=args.output) as stcl:
			# Run the training of the classifier:
			logger.info("Training %s on full training-set...", args.classifier)
			stcl.train(tset)
			logger.info("Training done.")
			logger.info("Classifying holdout-set using %s...", args.classifier)
			stcl.test(tset, save=tm.save_results, feature_importance=True)

	logger.info("Done.")

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()

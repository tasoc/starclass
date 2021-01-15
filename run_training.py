#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line script for training classifiers.

The default is to train the Meta Classifier, which includes training all other classifiers as well.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import logging
import starclass

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Utility function for training stellar classifiers.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing results.', action='store_true')
	parser.add_argument('--clear-cache', help='Clear existing features cache before running.', action='store_true')
	#
	parser.add_argument('-c', '--classifier',
		default='rfgc',
		choices=starclass.classifier_list,
		metavar='{CLASSIFIER}',
		help='Classifier to use. Choises are ' + ", ".join(starclass.classifier_list) + '.')

	parser.add_argument('-t', '--trainingset',
		default='keplerq9v3',
		choices=starclass.trainingset_list,
		metavar='{TSET}',
		help='Train classifier using this training-set. Choises are ' + ", ".join(starclass.trainingset_list) + '.')

	parser.add_argument('-l', '--level', help='Classification level', default='L1', choices=('L1', 'L2'))
	parser.add_argument('--linfit', help='Enable linfit in training set.', action='store_true')
	#parser.add_argument('--datalevel', help="", default='corr', choices=('raw', 'corr')) # TODO: Come up with better name than "datalevel"?
	parser.add_argument('-tf', '--testfraction', help='Holdout/test-set fraction', type=float, default=0.0)
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
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)
	logger_parent = logging.getLogger('starclass')
	logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)

	# Choose which classifier to use
	current_classifier = args.classifier

	# Pick the training set:
	tsetclass = starclass.get_trainingset(args.trainingset)
	tset = tsetclass(level=args.level, tf=args.testfraction, linfit=args.linfit)

	# If we were asked to do so, clear the cache before proceding:
	if args.clear_cache:
		tset.clear_cache()

	# The Meta-classifier requires us to first train all of the other classifiers
	# using cross-validation
	if current_classifier == 'meta':
		# Loop through all the other classifiers and initialize them:
		# TODO: Run in parallel?
		# TODO: Check if results are already present
		with starclass.TaskManager(tset.todo_file, overwrite=args.overwrite, classes=tset.StellarClasses) as tm:
			for cla_key in tm.all_classifiers:
				# Split the tset object into cross-validation folds.
				# These are objects with exactly the same properties as the original one,
				# except that they will run through different subsets of the training and test sets:
				cla = starclass.get_classifier(cla_key)
				for tset_fold in tset.folds(n_splits=5, tf=0.2):
					data_dir = tset.key + '/meta_fold{0:02d}'.format(tset_fold.fold)
					with cla(tset=tset, features_cache=tset.features_cache, data_dir=data_dir) as stcl:
						logger.info('Training %s on Fold %d/%d...', stcl.classifier_key, tset_fold.fold, tset_fold.crossval_folds)
						stcl.train(tset_fold)
						logger.info("Classifying test-set...")
						stcl.test(tset_fold, save=tm.save_results)

				# Now train all classifiers on the full training-set (minus the holdout-set),
				# and test on the holdout set:
				with cla(tset=tset, features_cache=tset.features_cache) as stcl:
					logger.info('Training %s on full training-set...', stcl.classifier_key)
					stcl.train(tset)
					logger.info("Classifying test-set using %s...", stcl.classifier_key)
					stcl.test(tset, save=tm.save_results)

	# Initialize the classifier:
	classifier = starclass.get_classifier(current_classifier)
	with starclass.TaskManager(tset.todo_file, overwrite=False, classes=tset.StellarClasses) as tm:
		with classifier(tset=tset, features_cache=tset.features_cache) as stcl:
			# Run the training of the classifier:
			logger.info("Training %s on full training-set...", current_classifier)
			stcl.train(tset)
			logger.info("Training done...")
			logger.info("Classifying test-set using %s...", current_classifier)
			stcl.test(tset, save=tm.save_results)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()

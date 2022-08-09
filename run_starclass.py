#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface for running classifications.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import argparse
import logging
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import starclass

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Command-line interface for running stellar classifiers.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing results.', action='store_true')
	parser.add_argument('--clear-cache', help='Clear existing features cache tables before running. Can only be used together with --overwrite.', action='store_true')
	# Option to select which classifier to run:
	parser.add_argument('-c', '--classifier',
		default=None,
		choices=starclass.classifier_list,
		metavar='{CLASSIFIER}',
		help='Classifier to run. Default is to run all classifiers. Choises are ' + ", ".join(starclass.classifier_list) + '.')
	# Option to select training set:
	parser.add_argument('-t', '--trainingset',
		default='keplerq9v3',
		choices=starclass.trainingset_list,
		metavar='{TSET}',
		help='Train classifier using this training-set. Choises are ' + ", ".join(starclass.trainingset_list) + '.')

	parser.add_argument('-l', '--level', help='Classification level.', default='L1', choices=('L1', 'L2'))
	parser.add_argument('--linfit', help='Enable linfit in training set.', action='store_true')
	#parser.add_argument('--datalevel', help="", default='corr', choices=('raw', 'corr')) # TODO: Come up with better name than "datalevel"?
	#parser.add_argument('--starid', type=int, help='TIC identifier of target.', nargs='?', default=None)
	# Lightcurve truncate override switch:
	group = parser.add_mutually_exclusive_group(required=False)
	group.add_argument('--truncate', dest='truncate', action='store_true', help='Force light curve truncation.')
	group.add_argument('--no-truncate', dest='truncate', action='store_false', help='Force no light curve truncation.')
	parser.set_defaults(truncate=None)
	# Data directory:
	parser.add_argument('--datadir', type=str, default=None, help='Directory where trained models and diagnostics will be loaded. Default is to load from the programs data directory.')
	# Input todo-file/directory:
	parser.add_argument('input_folder', type=str, nargs='?', default=None, help='Input directory to run classification on.')
	args = parser.parse_args()

	# Cache tables (MOAT) should not be cleared unless results tables are also cleared.
	# Otherwise we could end up with non-complete MOAT tables.
	if args.clear_cache and not args.overwrite:
		parser.error("--clear-cache can not be used without --overwrite")

	# Set logging level:
	logging_level = logging.INFO
	fmtstr = '%(asctime)s - %(levelname)s - %(message)s'
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG
		fmtstr = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s'

	# Setup logging:
	formatter = logging.Formatter(fmtstr)
	logger = logging.getLogger('starclass')
	if not logger.hasHandlers():
		console = starclass.utilities.TqdmLoggingHandler()
		console.setFormatter(formatter)
		logger.addHandler(console)
	logger.setLevel(logging_level)

	# Settings for tqdm:
	tqdm_settings = {'disable': None if logger.isEnabledFor(logging.INFO) else True}

	# Get input and output folder from environment variables:
	input_folder = args.input_folder
	if input_folder is None:
		input_folder = os.environ.get('STARCLASS_INPUT')
	if input_folder is None:
		parser.error("No input folder specified")
	if not os.path.exists(input_folder):
		parser.error("INPUT_FOLDER does not exist")
	if os.path.isdir(input_folder):
		todo_file = os.path.join(input_folder, 'todo.sqlite')
	else:
		todo_file = os.path.abspath(input_folder)
		input_folder = os.path.dirname(input_folder)

	# Choose which classifier to use:
	# If nothing was specified, run all classifiers, and automatically switch between them:
	if args.classifier is None:
		current_classifier = starclass.classifier_list[0]
		change_classifier = True
	else:
		current_classifier = args.classifier
		change_classifier = False

	# Make sure we have turned plotting to non-interactive:
	starclass.plots.plots_noninteractive()

	# Initialize training set:
	tsetclass = starclass.get_trainingset(args.trainingset)
	tset = tsetclass(level=args.level, linfit=args.linfit)

	# Running:
	# When simply running the classifier on new stars:
	stcl = None
	with starclass.TaskManager(todo_file, overwrite=args.overwrite, classes=tset.StellarClasses) as tm:
		# If we were asked to do so, start by clearing the existing MOAT tables:
		if args.overwrite and args.clear_cache:
			tm.moat_clear()

		# Get number of tasks:
		numtasks = tm.get_number_tasks(classifier=args.classifier)
		logger.info("%d tasks to be run", numtasks)

		with tqdm(total=numtasks, **tqdm_settings) as pbar:
			while True:
				tasks = tm.get_task(classifier=current_classifier, change_classifier=change_classifier)
				logger.debug(tasks)
				if tasks is None:
					break
				tm.start_task(tasks)

				# ----------------- This code would run on each worker ------------------------

				# Make sure we can loop through tasks,
				# even in the case we have only gotten one:
				results = []
				if isinstance(tasks, dict):
					tasks = [tasks]

				if tasks[0]['classifier'] != current_classifier or stcl is None:
					current_classifier = tasks[0]['classifier']
					if stcl:
						stcl.close()
					stcl = starclass.get_classifier(current_classifier)
					stcl = stcl(tset=tset, features_cache=None, truncate_lightcurves=args.truncate, data_dir=args.datadir)

				for task in tasks:
					res = stcl.classify(task)
					results.append(res)

				# ----------------- This code would run on each worker ------------------------

				# Return to TaskManager to be saved:
				tm.save_results(results)

				# Update progressbar:
				pbar.update(1)

		# Assign final classes:
		if args.classifier is None or args.classifier == 'meta':
			try:
				tm.assign_final_class(tset, data_dir=args.datadir)
			except starclass.exceptions.DiagnosticsNotAvailableError:
				logger.error("Could not assign final classes due to missing diagnostics information.")

	logger.info("Done.")

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()

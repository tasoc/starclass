#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility function for running classifiers.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import matplotlib.pyplot as plt
import os.path
import argparse
import logging
from starclass import TaskManager, RFGCClassifier, XGBClassifier, SLOSHClassifier, FOPTICSClassifier, MetaClassifier

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Utility function for running stellar classifiers.')
	parser.add_argument('-c', '--classifier', help='Classifier to use.', default='rfgc', choices=('rfgc', 'slosh', 'foptics', 'xgb', 'meta'))
	parser.add_argument('-l', '--level', help='Classification level', default='L1', choices=('L1', 'L2'))
	parser.add_argument('--datalevel', help="", default='corr', choices=('raw', 'corr')) # TODO: Come up with better name than "datalevel"?
	parser.add_argument('-o', '--overwrite', help='Overwrite existing results.', action='store_true')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	#parser.add_argument('--starid', type=int, help='TIC identifier of target.', nargs='?', default=None)
	parser.add_argument('input_folder', type=str, help='Input directory to run classification on.', nargs='?', default=None)
	args = parser.parse_args()

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

	# Get input and output folder from environment variables:
	input_folder = args.input_folder
	if input_folder is None:
		input_folder = os.environ.get('STARCLASS_INPUT')

	if input_folder is None:
		parser.error("No input folder specified")

	# Choose which classifier to use
	# For now, there is only one...
	current_classifier = args.classifier
	classifier = {
		'rfgc': RFGCClassifier,
		'slosh': SLOSHClassifier,
		'foptics': FOPTICSClassifier,
		'xgb': XGBClassifier,
		'meta': MetaClassifier
	}[current_classifier]

	# Path to TODO file and feature cache:
	todo_file = os.path.join(input_folder, 'todo.sqlite')
	features_cache = os.path.join(input_folder, 'features_cache_%s' % args.datalevel)

	if not os.path.exists(features_cache):
		os.makedirs(features_cache)

	# Running:
	# When simply running the classifier on new stars:
	with TaskManager(todo_file, overwrite=args.overwrite) as tm:

		with classifier(level=args.level, features_cache=features_cache, tset_key='keplerq9') as stcl:

			while True:
				task = tm.get_task(classifier=current_classifier)
				if task is None: break
				tm.start_task(task)

				#if task['classifier'] != current_classifier:
				#	stcl.close()
				#	stcl = classifier(level=args.level, features_cache=features_cache)

				# ----------------- This code would run on each worker ------------------------

				fname = os.path.join(input_folder, task['lightcurve']) # These are the lightcurves INCLUDING SYSTEMATIC NOISE
				features = stcl.load_star(task, fname)

				print(features)
				lc = features['lightcurve']
				lc.show_properties()

				plt.close('all')
				lc.plot()

				res = stcl.classify(features)
				#res = {
				#	StellarClasses.SOLARLIKE: np.random.rand(),
				#	StellarClasses.RRLYR: np.random.rand()
				#}

				# ----------------- This code would run on each worker ------------------------

				# Pad results with metadata and return to TaskManager to be saved:
				res.update({
					'priority': task['priority'],
					'classifier': task['classifier'],
					'status': 1
				})
				tm.save_result(res)

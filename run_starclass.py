#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility function for running classifiers.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import matplotlib.pyplot as plt
import os.path
import argparse
import logging
from timeit import default_timer
import starclass

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Utility function for running stellar classifiers.')
	parser.add_argument('-c', '--classifier', help='Classifier to use.', default='rfgc', choices=starclass.classifier_list)
	parser.add_argument('-t', '--trainingset', help='Train classifier using this training-set.', default='keplerq9v3', choices=starclass.trainingset_list)
	parser.add_argument('-l', '--level', help='Classification level', default='L1', choices=('L1', 'L2'))
	#parser.add_argument('--datalevel', help="", default='corr', choices=('raw', 'corr')) # TODO: Come up with better name than "datalevel"?
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

	# Initialize training set:
	tsetclass = starclass.get_trainingset(args.trainingset)
	tset = tsetclass(level=args.level)

	# Running:
	# When simply running the classifier on new stars:
	stcl = None
	with starclass.TaskManager(input_folder, overwrite=args.overwrite, classes=tset.StellarClasses) as tm:
		while True:
			task = tm.get_task(classifier=current_classifier)
			if task is None:
				break
			tm.start_task(task)

			if task['classifier'] != current_classifier or stcl is None:
				current_classifier = task['classifier']
				if stcl:
					stcl.close()
				stcl = starclass.get_classifier(current_classifier)
				stcl = stcl(tset=tset, features_cache=None)

			# ----------------- This code would run on each worker ------------------------

			fname = os.path.join(input_folder, task['lightcurve'])
			features = stcl.load_star(task, fname)

			print(features)
			lc = features['lightcurve']
			lc.show_properties()

			plt.close('all')
			lc.plot()

			res = task.copy()

			tic_predict = default_timer()
			res['starclass_results'] = stcl.classify(features)
			toc_predict = default_timer()

			# ----------------- This code would run on each worker ------------------------

			# Pad results with metadata and return to TaskManager to be saved:
			res.update({
				'status': starclass.STATUS.OK,
				'elaptime': toc_predict - tic_predict
			})
			tm.save_results(res)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()

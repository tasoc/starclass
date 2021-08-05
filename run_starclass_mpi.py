#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheduler using MPI for running the TASOC classification
pipeline on a large scale multi-core computer.

The setup uses the task-pull paradigm for high-throughput computing
using ``mpi4py``. Task pull is an efficient way to perform a large number of
independent tasks when there are more tasks than processors, especially
when the run times vary for each task.

The basic example was inspired by
https://github.com/jbornschein/mpi4py-examples/blob/master/09-task-pull.py

Example
-------
To run the program using four processes (one master and three workers) you can
execute the following command:

>>> mpiexec -n 4 python run_starclass_mpi.py

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from mpi4py import MPI
import argparse
import logging
import traceback
import os
import enum
import itertools
import starclass
from timeit import default_timer

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run TESS Corrections in parallel using MPI.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing results.', action='store_true')
	parser.add_argument('--chunks', type=int, default=10, help="Number of tasks sent to each worker at a time.")
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

	parser.add_argument('-l', '--level', help='Classification level', default='L1', choices=('L1', 'L2'))
	parser.add_argument('--linfit', help='Enable linfit in training set.', action='store_true')
	#parser.add_argument('--datalevel', help="", default='corr', choices=('raw', 'corr')) # TODO: Come up with better name than "datalevel"?
	# Lightcurve truncate override switch:
	group = parser.add_mutually_exclusive_group(required=False)
	group.add_argument('--truncate', dest='truncate', action='store_true', help='Force light curve truncation.')
	group.add_argument('--no-truncate', dest='truncate', action='store_false', help='Force no light curve truncation.')
	parser.set_defaults(truncate=None)
	# Data directory:
	parser.add_argument('--datadir', type=str, default=None, help='Directory where trained models and diagnostics will be loaded. Default is to load from the programs data directory.')
	# Input folder:
	parser.add_argument('input_folder', type=str, help='Input directory. This directory should contain a TODO-file and corresponding lightcurves.', nargs='?', default=None)
	args = parser.parse_args()

	# Cache tables (MOAT) should not be cleared unless results tables are also cleared.
	# Otherwise we could end up with non-complete MOAT tables.
	if args.clear_cache and not args.overwrite:
		parser.error("--clear-cache can not be used without --overwrite")
	# Make sure chunks are sensible:
	if args.chunks < 1:
		parser.error("--chunks should be an integer larger than 0.")

	# Get input and output folder from environment variables:
	input_folder = args.input_folder
	if input_folder is None:
		input_folder = os.environ.get('STARCLASS_INPUT')
	if not input_folder:
		parser.error("Please specify an INPUT_FOLDER.")
	if not os.path.exists(input_folder):
		parser.error("INPUT_FOLDER does not exist")
	if os.path.isdir(input_folder):
		todo_file = os.path.join(input_folder, 'todo.sqlite')
	else:
		todo_file = os.path.abspath(input_folder)
		input_folder = os.path.dirname(input_folder)

	# Initialize the training set:
	tsetclass = starclass.get_trainingset(args.trainingset)
	tset = tsetclass(level=args.level, linfit=args.linfit)

	# Define MPI message tags
	tags = enum.IntEnum('tags', ('READY', 'DONE', 'EXIT', 'START'))

	# Initializations and preliminaries
	comm = MPI.COMM_WORLD   # get MPI communicator object
	size = comm.size        # total number of processes
	rank = comm.rank        # rank of this process
	status = MPI.Status()   # get MPI status object

	if rank == 0:
		try:
			with starclass.TaskManager(todo_file, cleanup=True, overwrite=args.overwrite, classes=tset.StellarClasses) as tm:
				# If we were asked to do so, start by clearing the existing MOAT tables:
				if args.overwrite and args.clear_cache:
					tm.moat_clear()

				# Get list of tasks:
				#numtasks = tm.get_number_tasks()
				#tm.logger.info("%d tasks to be run", numtasks)

				# Number of available workers:
				num_workers = size - 1

				# Create a set of initial classifiers to initialize the workers as:
				# If nothing was specified run all classifiers, and automatically switch between them:
				if args.classifier is None:
					change_classifier = True
					initial_classifiers = []
					for k, c in enumerate(itertools.cycle(tm.all_classifiers)):
						if k >= num_workers:
							break
						initial_classifiers.append(c)
				else:
					initial_classifiers = [args.classifier]*num_workers
					change_classifier = False

				tm.logger.info("Initial classifiers: %s", initial_classifiers)

				# Start the master loop that will assign tasks
				# to the workers:
				closed_workers = 0
				tm.logger.info("Master starting with %d workers", num_workers)
				while closed_workers < num_workers:
					# Ask workers for information:
					data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
					source = status.Get_source()
					tag = status.Get_tag()

					if tag == tags.DONE:
						# The worker is done with a task
						tm.logger.debug("Got data from worker %d: %s", source, data)
						tm.save_results(data)

					if tag in (tags.DONE, tags.READY):
						# Worker is ready, so send it a task
						# If provided, try to find a task that is with the same classifier
						cl = initial_classifiers[source-1] if data is None else data[0].get('classifier')
						tasks = tm.get_task(classifier=cl, change_classifier=change_classifier, chunk=args.chunks)
						if tasks:
							tm.start_task(tasks)
							tm.logger.debug("Sending %d tasks to worker %d", len(tasks), source)
							comm.send(tasks, dest=source, tag=tags.START)
						else:
							comm.send(None, dest=source, tag=tags.EXIT)

					elif tag == tags.EXIT:
						# The worker has exited
						tm.logger.info("Worker %d exited.", source)
						closed_workers += 1

					else: # pragma: no cover
						# This should never happen, but just to
						# make sure we don't run into an infinite loop:
						raise RuntimeError(f"Master received an unknown tag: '{tag}'")

				# Assign final classes:
				if args.classifier is None or args.classifier == 'meta':
					try:
						tm.assign_final_class(tset, data_dir=args.datadir)
					except starclass.exceptions.DiagnosticsNotAvailableError:
						tm.logger.error("Could not assign final classes due to missing diagnostics information.")

				tm.logger.info("Master finishing")

		except: # noqa: E722, pragma: no cover
			# If something fails in the master
			print(traceback.format_exc().strip())
			comm.Abort(1)

	else:
		# Worker processes execute code below
		# Configure logging within starclass:
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		logger = logging.getLogger('starclass')
		logger.addHandler(console)
		logger.setLevel(logging.WARNING)

		# Get the class for the selected method:
		current_classifier = None
		stcl = None

		try:
			# Send signal that we are ready for task:
			comm.send(None, dest=0, tag=tags.READY)

			while True:
				# Receive a task from the master:
				tic_wait = default_timer()
				tasks = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
				tag = status.Get_tag()
				toc_wait = default_timer()

				if tag == tags.START:
					# Make sure we can loop through tasks,
					# even in the case we have only gotten one:
					results = []
					if isinstance(tasks, dict):
						tasks = [tasks]

					# Run the classification prediction:
					if tasks[0]['classifier'] != current_classifier or stcl is None:
						current_classifier = tasks[0]['classifier']
						if stcl:
							stcl.close()
						stcl = starclass.get_classifier(current_classifier)
						stcl = stcl(tset=tset, features_cache=None, truncate_lightcurves=args.truncate, data_dir=args.datadir)

					# Loop through the tasks given to us:
					for task in tasks:
						result = stcl.classify(task)

						# Pad results with metadata and return to TaskManager to be saved:
						result['worker_wait_time'] = toc_wait - tic_wait
						results.append(result)

					# Send the result back to the master:
					comm.send(results, dest=0, tag=tags.DONE)

					# Attempt some cleanup:
					# TODO: Is this even needed?
					del task, result

				elif tag == tags.EXIT:
					# We were told to EXIT, so lets do that
					break

				else: # pragma: no cover
					# This should never happen, but just to
					# make sure we don't run into an infinite loop:
					raise RuntimeError(f"Worker received an unknown tag: '{tag}'")

		except: # noqa: E722, pragma: no cover
			logger.exception("Something failed in worker")

		finally:
			comm.send(None, dest=0, tag=tags.EXIT)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()

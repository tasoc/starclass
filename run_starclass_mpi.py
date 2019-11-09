#!/usr/bin/env python
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

#------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run TESS Corrections in parallel using MPI.')
	parser.add_argument('-c', '--classifier', help='Classifier to use.', default=None, choices=('rfgc', 'slosh', 'foptics', 'xgb', 'meta'))
	parser.add_argument('-l', '--level', help='Classification level', default='L1', choices=('L1', 'L2'))
	parser.add_argument('--datalevel', help="", default='corr', choices=('raw', 'corr')) # TODO: Come up with better name than "datalevel"?
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing results.', action='store_true')
	parser.add_argument('input_folder', type=str, help='Input directory. This directory should contain a TODO-file and corresponding lightcurves.', nargs='?', default=None)
	args = parser.parse_args()

	# Get input and output folder from environment variables:
	input_folder = args.input_folder
	if input_folder is None:
		input_folder = os.environ.get('STARCLASS_INPUT')
	if not input_folder:
		parser.error("Please specify an INPUT_FOLDER.")

	# Setup the location of the features cache:
	#features_cache = os.path.join(input_folder, 'features_cache_%s' % args.datalevel)
	#if not os.path.exists(features_cache):
	#	os.makedirs(features_cache)

	# Define MPI message tags
	tags = enum.IntEnum('tags', ('READY', 'DONE', 'EXIT', 'START'))

	# Initializations and preliminaries
	comm = MPI.COMM_WORLD   # get MPI communicator object
	size = comm.size        # total number of processes
	rank = comm.rank        # rank of this process
	status = MPI.Status()   # get MPI status object

	if rank == 0:
		try:
			with starclass.TaskManager(input_folder, cleanup=True, overwrite=args.overwrite) as tm:
				# Get list of tasks:
				#numtasks = tm.get_number_tasks()
				#tm.logger.info("%d tasks to be run", numtasks)

				# Number of available workers:
				num_workers = size - 1

				# Create a set of initial classifiers to initialize the workers as:
				initial_classifiers = []
				for k, c in enumerate(itertools.cycle(tm.all_classifiers)):
					if k >= num_workers: break
					initial_classifiers.append(c)

				print(initial_classifiers)

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
						tm.logger.info("Got data from worker %d: %s", source, data)
						tm.save_results(data)

					if tag in (tags.DONE, tags.READY):
						# Worker is ready, so send it a task
						# If provided, try to find a task that is with the same classifier
						cl = initial_classifiers[source-1] if data is None else data.get('classifier')
						task = tm.get_task(classifier=cl, change_classifier=True)
						if task:
							tm.start_task(task)
							comm.send(task, dest=source, tag=tags.START)
							tm.logger.info("Sending task %d to worker %d", task['priority'], source)
						else:
							comm.send(None, dest=source, tag=tags.EXIT)

					elif tag == tags.EXIT:
						# The worker has exited
						tm.logger.info("Worker %d exited.", source)
						closed_workers += 1

					else:
						# This should never happen, but just to
						# make sure we don't run into an infinite loop:
						raise Exception("Master received an unknown tag: '{0}'".format(tag))

				tm.logger.info("Master finishing")

		except:
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
		ClassificationClass = {
			'rfgc': starclass.RFGCClassifier,
			'slosh': starclass.SLOSHClassifier,
			#'foptics': starclass.FOPTICSClassifier,
			'xgb': starclass.XGBClassifier,
			'meta': starclass.MetaClassifier
		}
		current_classifier = None
		stcl = None

		try:
			# Send signal that we are ready for task:
			comm.send(None, dest=0, tag=tags.READY)

			while True:
				# Receive a task from the master:
				tic_wait = default_timer()
				task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
				tag = status.Get_tag()
				toc_wait = default_timer()

				if tag == tags.START:
					result = task.copy()

					# Run the classification prediction:
					try:
						if task['classifier'] != current_classifier or stcl is None:
							current_classifier = task['classifier']
							if stcl: stcl.close()
							stcl = ClassificationClass[current_classifier](level=args.level, features_cache=None, tset_key='keplerq9')

						fname = os.path.join(input_folder, task['lightcurve'])
						features = stcl.load_star(task, fname)
						
						tic_predict = default_timer()
						result['starclass_results'] = stcl.classify(features)
						toc_predict = default_timer()
						
						result['elaptime'] = toc_predict - tic_predict
						result['status'] = starclass.STATUS.OK
					except:
						# Something went wrong
						error_msg = traceback.format_exc().strip()
						result.update({
							'status': starclass.STATUS.ERROR,
							'details': {'errors': [error_msg]},
						})

					# Pad results with metadata and return to TaskManager to be saved:
					result.update({
						'worker_wait_time': toc_wait - tic_wait
					})

					# Send the result back to the master:
					comm.send(result, dest=0, tag=tags.DONE)

					# Attempt some cleanup:
					# TODO: Is this even needed?
					del task, result

				elif tag == tags.EXIT:
					# We were told to EXIT, so lets do that
					break

				else:
					# This should never happen, but just to
					# make sure we don't run into an infinite loop:
					raise Exception("Worker received an unknown tag: '{0}'".format(tag))

		except:
			logger.exception("Something failed in worker")

		finally:
			comm.send(None, dest=0, tag=tags.EXIT)

if __name__ == '__main__':
	main()
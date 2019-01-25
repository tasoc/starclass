#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A TaskManager which keeps track of which targets to process.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import numpy as np
import os
import sqlite3
import logging
from astropy.table import Table

class TaskManager(object):
	"""
	A TaskManager which keeps track of which targets to process.
	"""

	def __init__(self, todo_file, cleanup=False, overwrite=False):
		"""
		Initialize the TaskManager which keeps track of which targets to process.

		Parameters:
			todo_file (string): Path to the TODO-file.
			cleanup (boolean): Perform cleanup/optimization of TODO-file before
			                   during initialization. Default=False.
			overwrite (boolean): Overwrite any previously calculated results. Default=False.

		Raises:
			IOError: If TODO-file could not be found.
		"""

		if os.path.isdir(todo_file):
			todo_file = os.path.join(todo_file, 'todo.sqlite')

		if not os.path.exists(todo_file):
			raise IOError('Could not find TODO-file')


		# Keep a list of all the possible classifiers here:
		self.all_classifiers = set(['rfgc', 'slosh', 'foptics', 'xgb'])

		# Setup logging:
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		self.logger = logging.getLogger(__name__)
		self.logger.addHandler(console)
		self.logger.setLevel(logging.INFO)

		# Load the SQLite file:
		self.conn = sqlite3.connect(todo_file)
		self.conn.row_factory = sqlite3.Row
		self.cursor = self.conn.cursor()

		# Reset the status of everything for a new run:
		# TODO: This should obviously be removed once we start running for real
		if overwrite:
			self.cursor.execute("DROP TABLE IF EXISTS starclass;")
			self.conn.commit()

		# Create table for diagnostics:
		self.cursor.execute("""CREATE TABLE IF NOT EXISTS starclass (
			priority INT NOT NULL,
			classifier TEXT NOT NULL,
			status INT NOT NULL,
			class TEXT,
			prob REAL,
			FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
		);""")
		self.cursor.execute("CREATE INDEX IF NOT EXISTS priority_classifier_idx ON starclass (priority, classifier);")
		self.conn.commit()

		# Run a cleanup/optimization of the database before we get started:
		if cleanup:
			self.logger.info("Cleaning TODOLIST before run...")
			try:
				self.conn.isolation_level = None
				self.cursor.execute("VACUUM;")
			except:
				raise
			finally:
				self.conn.isolation_level = ''

	def close(self):
		"""Close TaskManager and all associated objects."""
		self.cursor.close()
		self.conn.close()

	def __exit__(self, *args):
		self.close()

	def __enter__(self):
		return self

	def get_number_tasks(self):
		"""
		Get number of tasks due to be processed.

		Returns:
			int: Number of tasks due to be processed.
		"""
		raise NotImplementedError()

	def _query_task(self, cl):
		self.cursor.execute("""
			SELECT
				todolist.priority,
				todolist.starid,
				tmag,
				lightcurve,
				mean_flux,
				variance,
				variability,
				camera,
				ccd
			FROM
				todolist
				INNER JOIN diagnostics ON todolist.priority=diagnostics.priority
			WHERE
				todolist.status=1
				AND todolist.priority NOT IN (
					SELECT starclass.priority FROM starclass WHERE starclass.priority=todolist.priority AND starclass.classifier=:classifier
				)
			ORDER BY todolist.priority LIMIT 1;""", {'classifier': cl})
		task = self.cursor.fetchone()
		if task:
			task = dict(task)
			task['classifier'] = cl

			# If the classifier that is running is the meta-classifier,
			# add the results from all other classifiers to the task dict:
			if cl == 'meta':
				self.cursor.execute("SELECT classifier,class,prob FROM starclass WHERE priority=? AND classifier != 'meta';", (task['priority'], ))

				#for row in self.cursor.fetchall():


				task['other_classifiers'] = Table(
					rows=self.cursor.fetchall(),
					names=('classifier', 'class', 'prob'),
					dtype=('S256', 'S256', 'float32')
				)

			return task
		return None

	def get_task(self, classifier=None):
		"""
		Get next task to be processed.

		Parameters:
			classifier (string, optional): Classifier to get next task for.
				If no tasks are available for this classifier, a task for
				another classifier will be returned.

		Returns:
			dict or None: Dictionary of settings for task.
		"""

		task = None
		if classifier is not None:
			task = self._query_task(classifier)

		# If no task is returned for the given classifier, find another
		# classifier where tasks are available:
		if task is None:
			# Make a search on all the classifiers, and record the next
			# task for all of them:
			all_tasks = []
			for cl in self.all_classifiers.difference([classifier]):
				task = self._query_task(cl)
				if task is not None:
					all_tasks.append(task)

			# Pick the classifier that has reached the lowest priority:
			if all_tasks:
				indx = np.argmin([t['priority'] for t in all_tasks])
				return all_tasks[indx]

		return task

	def save_result(self, result):
		"""
		Save results and diagnostics. This will update the TODO list.

		Parameters:
			results (dict): Dictionary of results and diagnostics.
		"""

		priority = result.pop('priority')
		classifier = result.pop('classifier')
		status = result.pop('status')

		# Store the results in database:
		self.cursor.execute("DELETE FROM starclass WHERE priority=? AND classifier=?;", (priority, classifier))
		for key, value in result.items():
			self.cursor.execute("INSERT INTO starclass (priority,classifier,status,class,prob) VALUES (:priority,:classifier,:status,:class,:prob);", {
				'priority': priority,
				'classifier': classifier,
				'status': status,
				'class': key.name,
				'prob': value
			})
		self.conn.commit()

	def start_task(self, task):
		"""
		Mark a task as STARTED in the TODO-list.
		"""
		self.cursor.execute("INSERT INTO starclass (priority,classifier,status) VALUES (:priority,:classifier,6);", task)
		self.conn.commit()

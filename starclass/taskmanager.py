#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A TaskManager which keeps track of which targets to process.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import os
import sqlite3
import logging
from astropy.table import Table
from . import StellarClasses

class TaskManager(object):
	"""
	A TaskManager which keeps track of which targets to process.
	"""

	def __init__(self, todo_file, cleanup=False, readonly=False, overwrite=False):
		"""
		Initialize the TaskManager which keeps track of which targets to process.

		Parameters:
			todo_file (string): Path to the TODO-file.
			cleanup (boolean): Perform cleanup/optimization of TODO-file before
			                   during initialization. Default=False.
			overwrite (boolean): Overwrite any previously calculated results. Default=False.

		Raises:
			FileNotFoundError: If TODO-file could not be found.
		"""

		if os.path.isdir(todo_file):
			todo_file = os.path.join(todo_file, 'todo.sqlite')

		if not os.path.exists(todo_file):
			raise FileNotFoundError('Could not find TODO-file')

		self.readonly = readonly

		# Keep a list of all the possible classifiers here:
		self.all_classifiers = set(['rfgc', 'slosh', 'xgb'])

		# Setup logging:
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		self.logger = logging.getLogger(__name__)
		self.logger.addHandler(console)
		self.logger.setLevel(logging.INFO)

		# Load the SQLite file:
		#if self.readonly:
		#	self.conn = sqlite3.connect('file:' + todo_file + '?mode=ro', uri=True)
		#else:
		self.conn = sqlite3.connect(todo_file)
		self.conn.row_factory = sqlite3.Row
		self.cursor = self.conn.cursor()
		self.cursor.execute("PRAGMA foreign_keys=ON;")

		# Reset the status of everything for a new run:
		if overwrite:
			self.cursor.execute("DROP TABLE IF EXISTS starclass;")
			self.conn.commit()

		# Create table for diagnostics:
		self.cursor.execute("""CREATE TABLE IF NOT EXISTS starclass (
			priority INTEGER NOT NULL,
			classifier TEXT NOT NULL,
			status INTEGER NOT NULL,
			class TEXT,
			prob REAL,
			FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
		);""")
		self.cursor.execute("CREATE INDEX IF NOT EXISTS priority_classifier_idx ON starclass (priority, classifier);")
		self.conn.commit()

		# Analyze the tables for better query planning:
		self.cursor.execute("ANALYZE;")

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

	def _query_task(self, classifier=None, priority=None):

		search_query = []
		if classifier is not None:
			search_query.append("todolist.priority NOT IN (SELECT starclass.priority FROM starclass WHERE starclass.priority=todolist.priority AND starclass.classifier='%s')" % classifier)

		if priority is not None:
			search_query.append('todolist.priority=%d' % priority)

		if search_query:
			search_query = "AND " + " AND ".join(search_query)

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
				{0}
			ORDER BY todolist.priority LIMIT 1;""".format(search_query))
		task = self.cursor.fetchone()
		if task:
			task = dict(task)
			task['classifier'] = classifier

			# Add things from the catalog file:
			#catalog_file = os.path.join(????, 'catalog_sector{sector:03d}_camera{camera:d}_ccd{ccd:d}.sqlite')
			# cursor.execute("SELECT ra,decl as dec,teff FROM catalog WHERE starid=?;", (task['starid'], ))
			#task.update()

			# If the classifier that is running is the meta-classifier,
			# add the results from all other classifiers to the task dict:
			#if classifier == 'meta':
			self.cursor.execute("SELECT classifier,class,prob FROM starclass WHERE priority=? AND status=1 AND classifier != 'meta';", (task['priority'], ))

			# Add as a Table to the task list:
			rows = []
			for r in self.cursor.fetchall():
				rows.append([r['classifier'], StellarClasses[r['class']], r['prob']])
			if not rows: rows = None
			task['other_classifiers'] = Table(
				rows=rows,
				names=('classifier', 'class', 'prob'),
			)

			return task
		return None

	def get_task(self, priority=None, classifier=None, change_classifier=True):
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
		task = self._query_task(classifier, priority=priority)

		# If no task is returned for the given classifier, find another
		# classifier where tasks are available:
		if task is None and change_classifier:
			# Make a search on all the classifiers, and record the next
			# task for all of them:
			all_tasks = []
			for cl in self.all_classifiers.difference([classifier]):
				task = self._query_task(cl, priority=priority)
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
		worker_wait_time = result.pop('worker_wait_time')
		details = result.pop('details')

		# Store the results in database:
		try:
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
		except:
			self.conn.rollback()
			raise

	def start_task(self, task):
		"""
		Mark a task as STARTED in the TODO-list.
		"""
		try:
			self.cursor.execute("INSERT INTO starclass (priority,classifier,status) VALUES (:priority,:classifier,6);", task)
			self.conn.commit()
		except:
			self.conn.rollback()
			raise

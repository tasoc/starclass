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
from . import STATUS, StellarClasses

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
		self.all_classifiers = ('rfgc', 'slosh', 'xgb')

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
		self.cursor.execute("PRAGMA locking_mode=EXCLUSIVE;")
		self.cursor.execute("PRAGMA journal_mode=TRUNCATE;")

		# Reset the status of everything for a new run:
		if overwrite:
			self.cursor.execute("DROP TABLE IF EXISTS starclass_diagnostics;")
			self.cursor.execute("DROP TABLE IF EXISTS starclass_results;")
			self.conn.commit()

		# Create table for diagnostics:
		self.cursor.execute("""CREATE TABLE IF NOT EXISTS starclass_diagnostics (
			priority INTEGER NOT NULL,
			classifier TEXT NOT NULL,
			status INTEGER NOT NULL,
			elaptime REAL,
			worker_wait_time REAL,
			errors TEXT,
			PRIMARY KEY (priority, classifier),
			FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
		);""")
		self.cursor.execute("CREATE INDEX IF NOT EXISTS starclass_diag_status_idx ON starclass_diagnostics (status);")
		self.cursor.execute("""CREATE TABLE IF NOT EXISTS starclass_results (
			priority INTEGER NOT NULL,
			classifier TEXT NOT NULL,
			class TEXT NOT NULL,
			prob REAL NOT NULL,
			FOREIGN KEY (priority, classifier) REFERENCES starclass_diagnostics(priority, classifier) ON DELETE CASCADE ON UPDATE CASCADE
		);""")
		self.cursor.execute("CREATE INDEX IF NOT EXISTS starclass_resu_priority_classifier_idx ON starclass_results (priority, classifier);")

		# Make sure we have proper indicies that should have been created by the previous pipeline steps:
		self.cursor.execute("CREATE INDEX IF NOT EXISTS corr_status_idx ON todolist (corr_status);")

		# Find out if data-validation information exists:
		self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='datavalidation_corr';")
		self.datavalidation_exists = (self.cursor.fetchone() is not None)
		if not self.datavalidation_exists:
			self.logger.warning("DATA-VALIDATION information is not available in this TODO-file. Assuming all targets are good.")

		# Analyze the tables for better query planning:
		self.cursor.execute("ANALYZE;")
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

	def _query_task(self, classifier=None, priority=None):

		search_joins = []
		search_query = []

		# TODO: Is this right?
		if classifier is None and priority is None:
			raise ValueError("This will just give the same again and again")

		# Build list of constrainits:
		if priority is not None:
			search_query.append('todolist.priority=%d' % priority)

		# If data-validation information is available, only include targets
		# which passed the data validation:
		if self.datavalidation_exists:
			search_joins.append("INNER JOIN datavalidation_corr ON datavalidation_corr.priority=todolist.priority")
			search_query.append("datavalidation_corr.approved=1")

		# If a classifier is specified, constrain to only that classifier:
		if classifier is not None:
			search_joins.append("LEFT JOIN starclass_diagnostics ON starclass_diagnostics.priority=todolist.priority AND starclass_diagnostics.classifier='{classifier:s}'".format(
				classifier=classifier
			))
			search_query.append("starclass_diagnostics.status IS NULL")

		# Build query string:
		if search_joins:
			search_joins = "\n".join(search_joins)
		else:
			search_joins = ''

		if search_query:
			search_query = "AND " + " AND ".join(search_query)
		else:
			search_query = ''

		self.cursor.execute("""
			SELECT
				todolist.priority,
				todolist.starid,
				todolist.tmag,
				diagnostics_corr.lightcurve AS lightcurve
			FROM
				todolist
				INNER JOIN diagnostics_corr ON todolist.priority=diagnostics_corr.priority
				{joins:s}
			WHERE
				todolist.corr_status=1
				{constraints:s}
			ORDER BY todolist.priority LIMIT 1;""".format(
			joins=search_joins,
			constraints=search_query,
			classifier=classifier
		))
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
			# FIXME: Enforce this for META only. The problem is the TrainingSet class, which doesn't know about which classifier is running it
			if classifier == 'meta' or classifier is None:
				self.cursor.execute("SELECT starclass_results.classifier,class,prob FROM starclass_results INNER JOIN starclass_diagnostics ON starclass_results.priority=starclass_diagnostics.priority AND starclass_results.classifier=starclass_diagnostics.classifier WHERE starclass_results.priority=? AND status=? AND starclass_results.classifier != 'meta';", (task['priority'], STATUS.OK.value))

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
			for cl in set(self.all_classifiers).difference([classifier]):
				task = self._query_task(cl, priority=priority)
				if task is not None:
					all_tasks.append(task)

			# Pick the classifier that has reached the lowest priority:
			if all_tasks:
				indx = np.argmin([t['priority'] for t in all_tasks])
				return all_tasks[indx]

		return task

	def save_results(self, result):
		"""
		Save results and diagnostics. This will update the TODO list.

		Parameters:
			results (dict): Dictionary of results and diagnostics.
		"""

		priority = result.get('priority')
		classifier = result.get('classifier')
		status = result.get('status')
		details = result.get('details', {})
		starclass_results = result.get('starclass_results', {})

		# Save additional diagnostics:
		error_msg = details.get('errors', None)
		if error_msg:
			error_msg = '\n'.join(error_msg)
			#self.summary['last_error'] = error_msg

		# Store the results in database:
		try:
			# Save additional diagnostics:
			self.cursor.execute("INSERT OR REPLACE INTO starclass_diagnostics (priority,classifier,status,errors,elaptime,worker_wait_time) VALUES (:priority,:classifier,:status,:errors,:elaptime,:worker_wait_time);", {
				'priority': priority,
				'classifier': classifier,
				'status': status.value,
				'elaptime': result.get('elaptime'),
				'worker_wait_time': result.get('worker_wait_time'),
				'errors': error_msg
			})

			self.cursor.execute("DELETE FROM starclass_results WHERE priority=? AND classifier=?;", (priority, classifier))
			for key, value in starclass_results.items():
				self.cursor.execute("INSERT INTO starclass_results (priority,classifier,class,prob) VALUES (:priority,:classifier,:class,:prob);", {
					'priority': priority,
					'classifier': classifier,
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
			self.cursor.execute("INSERT INTO starclass_diagnostics (priority,classifier,status) VALUES (:priority,:classifier,:status);", {
				'priority': task['priority'],
				'classifier': task['classifier'],
				'status': STATUS.STARTED.value
			})
			self.conn.commit()
		except:
			self.conn.rollback()
			raise

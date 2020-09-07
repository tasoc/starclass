#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of starclass.TaskManager.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
from astropy.table import Table
import conftest # noqa: F401
from starclass import TaskManager, STATUS, StellarClasses

#--------------------------------------------------------------------------------------------------
def test_taskmanager_get_tasks(PRIVATE_TODO_FILE):
	"""Test of TaskManager"""

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True) as tm:
		# Get the number of tasks:
		with pytest.raises(NotImplementedError):
			tm.get_number_tasks()
		#print(numtasks)
		#assert(numtasks == 168642)

		# In STARCLASS, we have to ask with either a priority or a classifier as input:
		with pytest.raises(ValueError):
			tm.get_task()

		# Get the first task in the TODO file:
		task1 = tm.get_task(classifier='slosh')
		print(task1)

		# Check that it contains what we know it should:
		# The first priority in the TODO file is the following:
		assert task1['priority'] == 17
		assert task1['starid'] == 29281992
		assert task1['lightcurve'] == 'ffi/00029/tess00029281992-s01-c1800-dr01-v04-tasoc-cbv_lc.fits.gz'
		assert task1['classifier'] == 'slosh'

		# Start task with priority=1:
		tm.start_task(task1)

		# Get the next task, which should be the one with priority=2:
		task2 = tm.get_task(classifier='slosh')
		print(task2)

		assert task2['priority'] == 26
		assert task2['starid'] == 29859905
		assert task2['lightcurve'] == 'ffi/00029/tess00029859905-s01-c1800-dr01-v04-tasoc-cbv_lc.fits.gz'
		assert task2['classifier'] == 'slosh'

		# Check that the status did actually change in the todolist:
		tm.cursor.execute("SELECT status FROM starclass_diagnostics WHERE priority=?;", (task1['priority'],))
		task1_status = tm.cursor.fetchone()['status']
		print(task1_status)

		assert task1_status == STATUS.STARTED.value

#--------------------------------------------------------------------------------------------------
def test_taskmanager_get_tasks_priority(PRIVATE_TODO_FILE):
	"""Test of TaskManager.get_tasks with priority"""

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True) as tm:
		task = tm.get_task(priority=17)
		assert task['priority'] == 17

		# Call with non-existing starid:
		task = tm.get_task(priority=-1234567890)
		assert task is None

#--------------------------------------------------------------------------------------------------
def test_taskmanager_invalid():
	"""Test of TaskManager with invalid TODO-file input."""

	# Load the first image in the input directory:
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

	with pytest.raises(FileNotFoundError):
		TaskManager(os.path.join(INPUT_DIR, 'does-not-exists'))

#--------------------------------------------------------------------------------------------------
def test_taskmanager_switch_classifier(PRIVATE_TODO_FILE):
	"""Test of TaskManager - Automatic switching between classifiers."""

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True) as tm:

		tm.cursor.execute("INSERT INTO starclass_diagnostics (priority,classifier,status) SELECT priority,'slosh',1 FROM todolist;")
		tm.cursor.execute("DELETE FROM starclass_diagnostics WHERE priority=17;")
		tm.conn.commit()

		# Get the first task in the TODO file:
		task1 = tm.get_task(classifier='slosh')
		print(task1)

		# It should be the only missing task with SLOSH:
		assert task1['priority'] == 17
		assert task1['classifier'] == 'slosh'

		# Start task with priority=1:
		tm.start_task(task1)

		# Get the next task, which should be the one with priority=2:
		task2 = tm.get_task(classifier='slosh')
		print(task2)

		# We should now get the highest priority target, but not with SLOSH:
		assert task2['priority'] == 17
		assert task2['classifier'] and task2['classifier'] != 'slosh'

#--------------------------------------------------------------------------------------------------
def test_taskmanager_meta_classifier(PRIVATE_TODO_FILE):
	"""Test of TaskManager when running with MetaClassifier"""

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True) as tm:

		# Create fake results from SLOSH:
		tm.save_results({'priority': 17, 'classifier': 'slosh', 'status': STATUS.OK, 'starclass_results': {
			StellarClasses.SOLARLIKE: 0.2,
			StellarClasses.DSCT_BCEP: 0.1,
			StellarClasses.ECLIPSE: 0.7
		}})

		# Get the first task in the TODO file for the MetaClassifier:
		task1 = tm.get_task(classifier='meta')
		print(task1)

		# It should be the only missing task with SLOSH:
		assert task1['priority'] == 17
		assert task1['classifier'] == 'meta'
		assert isinstance(task1['other_classifiers'], Table)

		tab = task1['other_classifiers']
		print(tab)

		assert tab[tab['class'] == StellarClasses.SOLARLIKE]['prob'] == 0.2
		assert tab[tab['class'] == StellarClasses.DSCT_BCEP]['prob'] == 0.1
		assert tab[tab['class'] == StellarClasses.ECLIPSE]['prob'] == 0.7

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

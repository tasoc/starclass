#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of starclass.TaskManager.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
from astropy.table import Table
import conftest # noqa: F401
from starclass import TaskManager, STATUS
from starclass.StellarClasses import StellarClassesLevel1

#--------------------------------------------------------------------------------------------------
def test_taskmanager_get_tasks(PRIVATE_TODO_FILE):
	"""Test of TaskManager"""

	input_dir = os.path.dirname(PRIVATE_TODO_FILE)

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
		assert task1['lightcurve'] == os.path.join(input_dir, 'tess00029281992-s01-c1800-dr01-v04-tasoc-cbv_lc.fits.gz')
		assert task1['classifier'] == 'slosh'

		# Start task with priority=1:
		tm.start_task(task1)

		# Get the next task, which should be the one with priority=2:
		task2 = tm.get_task(classifier='slosh')
		print(task2)

		assert task2['priority'] == 26
		assert task2['starid'] == 29859905
		assert task2['lightcurve'] == os.path.join(input_dir, 'ffi/00029/tess00029859905-s01-c1800-dr01-v04-tasoc-cbv_lc.fits.gz')
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

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, classes=StellarClassesLevel1) as tm:

		# Create fake results from SLOSH:
		tm.save_results({'priority': 17, 'classifier': 'slosh', 'status': STATUS.OK, 'starclass_results': {
			StellarClassesLevel1.SOLARLIKE: 0.2,
			StellarClassesLevel1.DSCT_BCEP: 0.1,
			StellarClassesLevel1.ECLIPSE: 0.7
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

		for row in tab:
			assert isinstance(row['class'], StellarClassesLevel1)

		assert tab[tab['class'] == StellarClassesLevel1.SOLARLIKE]['prob'] == 0.2
		assert tab[tab['class'] == StellarClassesLevel1.DSCT_BCEP]['prob'] == 0.1
		assert tab[tab['class'] == StellarClassesLevel1.ECLIPSE]['prob'] == 0.7

#--------------------------------------------------------------------------------------------------
def test_taskmanager_save_and_settings(PRIVATE_TODO_FILE):
	"""Test of TaskManager saving results and settings."""

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, classes=StellarClassesLevel1) as tm:
		# Check the settings table:
		assert tm.tset is None
		tm.cursor.execute("SELECT * FROM starclass_settings;")
		settings = tm.cursor.fetchone()
		assert settings is None

		# Start a random task:
		task = tm.get_task(classifier='meta')
		print(task)
		tm.start_task(task)

		# Make a fake result we can save;
		starclass_results = {
			StellarClassesLevel1.SOLARLIKE: 0.8,
			StellarClassesLevel1.APERIODIC: 0.2
		}
		result = task.copy()
		result['tset'] = 'keplerq9v3'
		result['classifier'] = 'meta'
		result['status'] = STATUS.OK
		result['elaptime'] = 3.14
		result['worker_wait_time'] = 1.0
		result['details'] = {'errors': ['There was actually no error']}
		result['starclass_results'] = starclass_results

		# Save the result:
		tm.save_results(result)

		# Check the setting again - it should now have changed:
		assert tm.tset == 'keplerq9v3'
		tm.cursor.execute("SELECT * FROM starclass_settings;")
		settings = tm.cursor.fetchone()
		assert settings['tset'] == 'keplerq9v3'

		# Check that the additional diagnostic was saved correctly:
		tm.cursor.execute("SELECT * FROM starclass_diagnostics WHERE priority=?;", [result['priority']])
		row = tm.cursor.fetchone()
		print(dict(row))
		assert row['status'] == STATUS.OK.value
		assert row['classifier'] == 'meta'
		assert row['elaptime'] == 3.14
		assert row['worker_wait_time'] == 1.0
		assert row['errors'] == 'There was actually no error'

		# Check that the results were saved correctly:
		tm.cursor.execute("SELECT class,prob FROM starclass_results WHERE priority=? AND classifier=?;", [result['priority'], 'meta'])
		for row in tm.cursor.fetchall():
			assert row['prob'] == starclass_results[StellarClassesLevel1[row['class']]]

		# This should fail when we try to save it:
		result['tset'] = 'another'
		with pytest.raises(ValueError) as e:
			tm.save_results(result)
		assert str(e.value) == "Attempting to mix results from multiple training sets. Previous='keplerq9v3', New='another'."

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('classifier', ['rfgc', 'xgb', 'sortinghat', 'slosh'])
def test_taskmanager_moat(PRIVATE_TODO_FILE, classifier):

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, classes=StellarClassesLevel1) as tm:
		# Start a random task:
		task = tm.get_task(classifier=classifier)
		print(task)

		#
		features_common = {'freq1': 42, 'amp1': 43, 'phase1': 4}
		features = {'unique_feature': 2187, 'special_feature': 1234}

		# Make a fake result we can save;
		result = task.copy()
		result['tset'] = 'keplerq9v3'
		result['classifier'] = classifier
		result['status'] = STATUS.OK
		result['elaptime'] = 3.14
		result['starclass_results'] = {
			StellarClassesLevel1.SOLARLIKE: 0.8,
			StellarClassesLevel1.APERIODIC: 0.2
		}
		# This is the important part in this test:
		result['features'] = features
		result['features_common'] = features_common

		# Save the result:
		tm.save_results(result)

		# Check common features were stored in the table:
		tm.cursor.execute("SELECT * FROM starclass_features_common WHERE priority=?;", [task['priority']])
		row1 = dict(tm.cursor.fetchone())
		del row1['priority']
		assert row1 == features_common

		tm.cursor.execute("SELECT * FROM starclass_features_%s WHERE priority=?;" % classifier, [task['priority']])
		row2 = dict(tm.cursor.fetchone())
		del row2['priority']
		assert row2 == features

		# Try to extract the features again, they should be identical to the ones we put in:
		extracted_features = tm.moat_query('common', task['priority'])
		assert extracted_features == features_common
		extracted_features = tm.moat_query(classifier, task['priority'])
		assert extracted_features == features

		# If we ask for the exact same target, we should get another classifier,
		# but the common features should now be provided to us:
		task2 = tm.get_task(priority=task['priority'], classifier=classifier)
		print(task2)
		assert task2['classifier'] != classifier
		assert task2['features_common'] == features_common

		# Clear the moat, which should delete all MOAT tables:
		tm.moat_clear()

		# Check that there are no more MOAT tables in the todo-file:
		tm.cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name LIKE 'starclass_features_%';")
		assert tm.cursor.fetchone()[0] == 0

		# Using the query should now return nothing:
		assert tm.moat_query('common', task['priority']) is None
		assert tm.moat_query(classifier, task['priority']) is None

#--------------------------------------------------------------------------------------------------
def test_taskmanager_moat_create_wrong(PRIVATE_TODO_FILE):
	"""Test moat-create with wrong input"""
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, classes=StellarClassesLevel1) as tm:
		# Wrong classifier:
		with pytest.raises(ValueError) as e:
			tm.moat_create('nonsense', ['freq1', 'freq2'])
		assert str(e.value) == 'Invalid classifier: nonsense'

		with pytest.raises(ValueError) as e:
			tm.moat_create('common', [])

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

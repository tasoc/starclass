#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of starclass.TaskManager.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import sqlite3
import tempfile
import shutil
import logging
from contextlib import closing
import numpy as np
from astropy.table import Table
import conftest # noqa: F401
import starclass
from starclass import TaskManager, STATUS
from starclass.StellarClasses import StellarClassesLevel1

AVALIABLE_CLASSIFIERS = list(starclass.classifier_list)
AVALIABLE_CLASSIFIERS.remove('meta')

#--------------------------------------------------------------------------------------------------
def _make_fake_result(tasks):
	single_result = isinstance(tasks, dict)
	if single_result:
		tasks = [tasks]
	results = []
	for task in tasks:
		results.append({
			'priority': task['priority'],
			'classifier': task['classifier'],
			'status': STATUS.OK,
			'elaptime': 3.14,
			'worker_wait_time': 1.0,
			'tset': 'keplerq9v3',
			'details': {'errors': ['There was actually no error']},
			'starclass_results': {
				StellarClassesLevel1.SOLARLIKE: 0.2,
				StellarClassesLevel1.DSCT_BCEP: 0.1,
				StellarClassesLevel1.ECLIPSE: 0.7
			}
		})
	if single_result:
		return results[0]
	else:
		return results

#--------------------------------------------------------------------------------------------------
def _num_saved(todo_file):
	with closing(sqlite3.connect('file:' + todo_file + '?mode=ro', uri=True)) as conn:
		cursor = conn.cursor()
		try:
			cursor.execute("SELECT COUNT(*) FROM starclass_diagnostics;")
			return cursor.fetchone()[0]
		except sqlite3.OperationalError as e:
			if str(e) == 'no such table: starclass_diagnostics':
				return 0
			raise

#--------------------------------------------------------------------------------------------------
def test_taskmanager_get_number_tasks(PRIVATE_TODO_FILE):
	"""Test of get_number_tasks"""
	# Overwriting so no results are present before we start
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True) as tm:
		# Pull in list of all classifiers:
		all_classifiers = starclass.classifier_list
		Nclassifiers = len(all_classifiers)

		# Get the number of all target that can be processed by starclass:
		tm.cursor.execute("SELECT COUNT(*) FROM todolist t INNER JOIN datavalidation_corr d ON t.priority=d.priority WHERE approved=1 AND corr_status IN (1,3);")
		realnum = tm.cursor.fetchone()[0]
		print(f"real num per classifier: {realnum:d}")

		# Get the number of tasks:
		numtasks = tm.get_number_tasks()
		print(f"all classifiers: {numtasks:d}")
		assert numtasks == realnum*Nclassifiers

		# Get the number of tasks:
		for clfier in all_classifiers:
			numtasks = tm.get_number_tasks(classifier=clfier)
			print(f"{clfier:s}: {numtasks:d}")
			assert numtasks == realnum

		# Insert fake results for all but one SLOSH target:
		tm.cursor.execute("INSERT INTO starclass_diagnostics (priority,classifier,status) SELECT priority,'slosh',1 FROM todolist;")
		tm.cursor.execute("DELETE FROM starclass_diagnostics WHERE priority=17;")
		tm.conn.commit()

		# Get the number of tasks:
		assert tm.get_number_tasks(classifier='slosh') == 1

		# Get the number of tasks:
		numtasks = tm.get_number_tasks()
		print(numtasks)
		assert numtasks == realnum*(Nclassifiers-1) + 1

		# Insert fake results for all but one SLOSH target:
		tm.cursor.execute("DELETE FROM starclass_diagnostics;")
		for clfier in all_classifiers:
			tm.cursor.execute("INSERT INTO starclass_diagnostics (priority,classifier,status) SELECT priority,?,1 FROM todolist;", [clfier])
		tm.conn.commit()

		# Get the number of tasks:
		assert tm.get_number_tasks() == 0

		# Delete everything for a single target:
		tm.cursor.execute("DELETE FROM starclass_diagnostics WHERE priority=17;")
		tm.conn.commit()

		# There should now be missing tasks for all classifiers, but only one target:
		assert tm.get_number_tasks() == Nclassifiers

#--------------------------------------------------------------------------------------------------
def test_taskmanager_get_tasks(PRIVATE_TODO_FILE):
	"""Test of TaskManager"""

	input_dir = os.path.dirname(PRIVATE_TODO_FILE)

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True) as tm:
		# In STARCLASS, we have to ask with either a priority or a classifier as input:
		with pytest.raises(ValueError):
			tm.get_task()

		# Get the first task in the TODO file:
		task1 = tm.get_task(classifier='slosh', chunk=1)[0]
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
		task2 = tm.get_task(classifier='slosh', chunk=1)[0]
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
def test_taskmanager_chunks(PRIVATE_TODO_FILE):
	"""Test TaskManager, getting chunks of tasks at a time"""

	# Reset the TODO-file completely, and mark the first task as STARTED:
	with TaskManager(PRIVATE_TODO_FILE) as tm:
		task1 = tm.get_task(classifier='rfgc')
		assert isinstance(task1, list)
		assert len(task1) == 1
		assert isinstance(task1[0], dict)

		task10 = tm.get_task(classifier='rfgc', chunk=10)
		assert isinstance(task10, list)
		assert len(task10) == 10
		for task in task10:
			assert isinstance(task, dict)

		tm.start_task(task10)
		tm.cursor.execute("SELECT COUNT(*) FROM starclass_diagnostics WHERE classifier='rfgc' AND status=?;", [STATUS.STARTED.value])
		assert tm.cursor.fetchone()[0] == 10

#--------------------------------------------------------------------------------------------------
def test_taskmanager_get_tasks_priority(PRIVATE_TODO_FILE):
	"""Test of TaskManager.get_tasks with priority"""

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True) as tm:
		task = tm.get_task(priority=17)
		assert task[0]['priority'] == 17

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
def test_taskmanager_no_classes(PRIVATE_TODO_FILE):

	with TaskManager(PRIVATE_TODO_FILE) as tm:
		# Fill fake results needed for the MetaClassifier:
		for classifier in tm.all_classifiers:
			tm.cursor.execute("INSERT INTO starclass_diagnostics (priority,classifier,status) SELECT priority,?,1 FROM todolist;", [classifier])
		tm.conn.commit()

		# The TaskManager should now throw an error when asking for classifier=meta:
		with pytest.raises(RuntimeError) as err:
			tm.get_task(classifier='meta', change_classifier=False)

	assert str(err.value) == "classes not provided to TaskManager."

#--------------------------------------------------------------------------------------------------
def test_taskmanager_no_diagnostics(PRIVATE_TODO_FILE):
	"""Test of TaskManager with invalid TODO-file, missing diagnostics_corr table."""

	# Delete the table from the TODO-file:
	with closing(sqlite3.connect(PRIVATE_TODO_FILE)) as conn:
		conn.execute('DROP TABLE IF EXISTS diagnostics_corr;')
		conn.commit()

	# The TaskManager should now throw an error:
	with pytest.raises(ValueError) as err:
		TaskManager(PRIVATE_TODO_FILE)
	assert str(err.value) == "The TODO-file does not contain diagnostics_corr. Are you sure corrections have been run?"

#--------------------------------------------------------------------------------------------------
def test_taskmanager_no_datavalidation(PRIVATE_TODO_FILE, caplog):
	"""Test to make sure we log a warning if run on a todo-file without data-validation."""

	# Remove data-valudation table from db:
	with closing(sqlite3.connect(PRIVATE_TODO_FILE)) as conn:
		conn.execute("DROP TABLE datavalidation_corr;")
		conn.commit()

	with caplog.at_level(logging.WARNING):
		TaskManager(PRIVATE_TODO_FILE, classes=StellarClassesLevel1)

	recs = caplog.records
	print(recs)
	assert len(recs) == 1
	assert recs[0].levelname == 'WARNING'
	assert recs[0].getMessage() == "DATA-VALIDATION information is not available in this TODO-file. Assuming all targets are good."

#--------------------------------------------------------------------------------------------------
def test_taskmanager_moat_wrong_existing_table(PRIVATE_TODO_FILE):
	"""Test of TaskManager with invalid TODO-file, missing diagnostics_corr table."""

	# Add a table to the TODO-file, following the naming scheme, but from
	# a wrong (dummy) classifier:
	with closing(sqlite3.connect(PRIVATE_TODO_FILE)) as conn:
		conn.execute("""CREATE TABLE starclass_features_dummy (
			priority integer not null,
			dummy_var real
		);""")
		conn.commit()

	# The TaskManager should now throw an error:
	with pytest.raises(ValueError) as err:
		TaskManager(PRIVATE_TODO_FILE)
	assert str(err.value) == "Invalid classifier: dummy"

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('chunk', [1, 10])
def test_taskmanager_switch_classifier(PRIVATE_TODO_FILE, chunk):
	"""Test of TaskManager - Automatic switching between classifiers."""

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True) as tm:

		tm.cursor.execute("INSERT INTO starclass_diagnostics (priority,classifier,status) SELECT priority,'slosh',1 FROM todolist;")
		tm.cursor.execute("DELETE FROM starclass_diagnostics WHERE priority=17;")
		tm.conn.commit()

		# Get the first task in the TODO file:
		task1 = tm.get_task(classifier='slosh', chunk=chunk)
		print(task1)

		# It should be the only missing task with SLOSH:
		assert len(task1) == 1
		task1 = task1[0]
		assert task1['priority'] == 17
		assert task1['classifier'] == 'slosh'

		# Start task with priority=1:
		tm.start_task(task1)

		# Get the next task, which should be the one with priority=2:
		task2 = tm.get_task(classifier='slosh', chunk=chunk)
		print(task2)
		assert len(task2) == chunk
		task2 = task2[0]

		# We should now get the highest priority target, but not with SLOSH:
		assert task2['priority'] == 17
		assert task2['classifier'] is not None and task2['classifier'] != 'slosh'

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('chunk', [1, 10])
def test_taskmanager_switch_classifier_meta(PRIVATE_TODO_FILE, chunk):
	"""Test of TaskManager - Automatic switching between classifiers."""

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, classes=StellarClassesLevel1) as tm:

		for classifier in tm.all_classifiers:
			tm.cursor.execute("INSERT INTO starclass_diagnostics (priority,classifier,status) SELECT priority,?,1 FROM todolist;", [classifier])
		tm.cursor.execute("DELETE FROM starclass_diagnostics WHERE priority=17 AND classifier='slosh';")
		tm.conn.commit()

		# Get the first task in the TODO file:
		task1 = tm.get_task(classifier='slosh', chunk=chunk)
		print(task1)

		# It should be the only missing task with SLOSH:
		assert len(task1) == 1
		task1 = task1[0]
		assert task1['priority'] == 17
		assert task1['classifier'] == 'slosh'

		# Start task with priority=1:
		tm.start_task(task1)

		# Pretend we are now another worker, that have started processing with MetaClassifier.
		# When we now ask for a new set of tasks, the above task can not be among them, since
		# it is not yet complete.
		task2 = tm.get_task(classifier='meta', chunk=chunk)
		print(task2)
		assert len(task2) == chunk
		priorities = [t['priority'] for t in task2]
		classifiers = [t['classifier'] for t in task2]

		assert 17 not in priorities
		assert np.all(np.array(classifiers) == 'meta')

		# Now save a dummy result for the missing classifier:
		tm.cursor.execute("UPDATE starclass_diagnostics SET status=1 WHERE priority=17 AND classifier='slosh';")
		tm.conn.commit()

		# When the next worker now ask for something to do with the MetaClassifier,
		# the task should now be avialble:
		task3 = tm.get_task(classifier='meta', chunk=chunk)
		print(task3)
		assert len(task3) == chunk
		priorities = [t['priority'] for t in task3]
		classifiers = [t['classifier'] for t in task3]

		assert 17 in priorities
		assert np.all(np.array(classifiers) == 'meta')

#--------------------------------------------------------------------------------------------------
def test_taskmanager_meta_classifier(PRIVATE_TODO_FILE):
	"""Test of TaskManager when running with MetaClassifier"""

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, classes=StellarClassesLevel1) as tm:

		# Create fake results from SLOSH
		for classifier in tm.all_classifiers:
			tm.save_results(_make_fake_result({'priority': 17, 'classifier': classifier}))

		# Get the first task in the TODO file for the MetaClassifier:
		task1 = tm.get_task(classifier='meta', chunk=1)[0]
		print(task1)

		# It should be the only missing task with SLOSH:
		assert task1['priority'] == 17
		assert task1['classifier'] == 'meta'
		assert isinstance(task1['other_classifiers'], Table)

		tab = task1['other_classifiers']
		print(tab)

		for row in tab:
			assert isinstance(row['class'], StellarClassesLevel1)

		assert np.all(tab[tab['class'] == StellarClassesLevel1.SOLARLIKE]['prob'] == 0.2)
		assert np.all(tab[tab['class'] == StellarClassesLevel1.DSCT_BCEP]['prob'] == 0.1)
		assert np.all(tab[tab['class'] == StellarClassesLevel1.ECLIPSE]['prob'] == 0.7)

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
		task = tm.get_task(classifier=AVALIABLE_CLASSIFIERS[0], chunk=1)[0]
		print(task)
		tm.start_task(task)

		# Make a fake result we can save;
		starclass_results = {
			StellarClassesLevel1.SOLARLIKE: 0.8,
			StellarClassesLevel1.APERIODIC: 0.2
		}
		result = _make_fake_result(task)
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
		assert row['classifier'] == task['classifier']
		assert row['elaptime'] == result['elaptime']
		assert row['worker_wait_time'] == result['worker_wait_time']
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
@pytest.mark.parametrize('classifier', AVALIABLE_CLASSIFIERS)
def test_taskmanager_moat(PRIVATE_TODO_FILE, classifier):

	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, classes=StellarClassesLevel1) as tm:
		# Start a random task:
		task = tm.get_task(classifier=classifier)[0]
		print(task)

		# Create dummy features which we will save and restore:
		features_common = {'freq1': 42.0, 'amp1': 43.0, 'phase1': 4.0}
		features = {'unique_feature': 2187.0, 'special_feature': 1234.0}

		# Make a fake result we can save;
		result = _make_fake_result(task)

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
		task2 = tm.get_task(priority=task['priority'], classifier=classifier)[0]
		print('TASK2: %s' % task2)
		assert task2['classifier'] != classifier
		assert task2['features_common'] == features_common

	# Reload the TaskManager, with overwrite, which should remove all previous results,
	# but the MOAT should still exist:
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, classes=StellarClassesLevel1) as tm:

		# If we ask for the exact same target, we should get THE SAME classifier,
		# but the common features should now be provided to us:
		task3 = tm.get_task(priority=task['priority'], classifier=classifier)[0]
		print('TASK3: %s' % task3)
		assert task3['classifier'] == classifier
		assert task3['features_common'] == features_common
		assert task3['features'] == features

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
def test_taskmanager_assign_final_class(SHARED_INPUT_DIR):

	tsetclass = starclass.get_trainingset('keplerq9v3')
	tset = tsetclass()

	with tempfile.TemporaryDirectory() as tmpdir:

		todo_file = os.path.join(tmpdir, 'todo.sqlite')
		datadir = os.path.join(tmpdir, tset.level, tset.key)
		diag_file = os.path.join(datadir, 'diagnostics_keplerq9v3_L1_meta.json')
		shutil.copyfile(os.path.join(SHARED_INPUT_DIR, 'meta', 'todo_with_meta.sqlite'), todo_file)
		os.makedirs(datadir)

		with TaskManager(todo_file, classes=tset.StellarClasses) as tm:
			# Make sure the new column doesn't already exist:
			tm.cursor.execute("PRAGMA table_info(todolist);")
			assert 'final_class' not in [col['name'] for col in tm.cursor], "FINAL_CLASS column already exists"

			# Running without diagnostics file, should result in custom error:
			with pytest.raises(starclass.exceptions.DiagnosticsNotAvailableError):
				tm.assign_final_class(tset, data_dir=tmpdir)

			# Create dummy JSON file:
			with open(diag_file, 'w') as fid:
				fid.write('{"foo":1,"bar":2}')

			# Running on a wrong diagnostics file, should result in custom error:
			with pytest.raises(starclass.exceptions.DiagnosticsNotAvailableError):
				tm.assign_final_class(tset, data_dir=tmpdir)

			# Copy proper diagnostics file to data directory:
			shutil.copyfile(os.path.join(SHARED_INPUT_DIR, 'diagnostics', 'diagnostics_keplerq9v3_L1_meta.json'), diag_file)

			# Run the assignment of final classes:
			tm.assign_final_class(tset, data_dir=tmpdir)

			# Check that some of the classes are populated:
			tm.cursor.execute("SELECT COUNT(*) FROM todolist WHERE final_class IS NOT NULL;")
			assert tm.cursor.fetchone()[0] > 0

			# Check that the contents of the final classes makes sense:
			first_try = []
			tm.cursor.execute("""SELECT t.priority,d.approved,final_class,s.status
				FROM todolist t
				INNER JOIN datavalidation_corr d ON t.priority=d.priority
				LEFT JOIN starclass_diagnostics s ON t.priority=s.priority AND classifier='meta';""")
			for row in tm.cursor:
				first_try.append(row['final_class'])
				if not row['approved'] or row['status'] not in (STATUS.OK.value, STATUS.WARNING.value):
					assert row['final_class'] is None, "FINAL_CLASS defined for not approved target"
				elif row['final_class'] != 'UNKNOWN':
					# This will throw KeyError if an invalid value
					tset.StellarClasses[row['final_class']]

			# Run the asignment of final classes once again:
			tm.assign_final_class(tset, data_dir=tmpdir)

			# We should get the exact same result the second time:
			tm.cursor.execute("SELECT final_class FROM todolist ORDER BY priority;")
			second_try = [row[0] for row in tm.cursor]
			np.testing.assert_array_equal(first_try, second_try)

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('interval', [
	pytest.param(-1, marks=pytest.mark.xfail(raises=ValueError)),
	pytest.param(-1.0, marks=pytest.mark.xfail(raises=ValueError)),
	pytest.param(0, marks=pytest.mark.xfail(raises=ValueError)),
	pytest.param(0.0, marks=pytest.mark.xfail(raises=ValueError)),
	pytest.param(np.nan, marks=pytest.mark.xfail(raises=ValueError)),
	pytest.param('nonsense', marks=pytest.mark.xfail(raises=ValueError)),
	1,
	1.0,
	10000,
	None
])
def test_taskmanager_backupinterval(PRIVATE_TODO_FILE, interval):
	"""Test TaskManager with invalid backup interval"""
	TaskManager(PRIVATE_TODO_FILE, overwrite=False, cleanup=False, classes=StellarClassesLevel1,
		backup_interval=interval)

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('in_memory', [True, False])
@pytest.mark.parametrize('chunk', [1, 10])
def test_taskmanager_backup_manual(PRIVATE_TODO_FILE, in_memory, chunk):

	# Manual backup:
	with TaskManager(PRIVATE_TODO_FILE, load_into_memory=in_memory, backup_interval=None) as tm:
		# In order to be able to read the database while it is still open,
		# change the locking_mode here:
		if not in_memory:
			tm.cursor.execute("PRAGMA locking_mode=NORMAL;")
			tm.conn.commit()

		for _ in range(3):
			task = tm.get_task(classifier=AVALIABLE_CLASSIFIERS[0], chunk=chunk)
			res = _make_fake_result(task)
			tm.save_results(res)

		assert _num_saved(PRIVATE_TODO_FILE) == (0 if in_memory else 3*chunk)

		tm.backup()

		assert _num_saved(PRIVATE_TODO_FILE) == 3*chunk

	# Because we have closed the TaskManager, the on-disk file should be fully up-to-date:
	assert _num_saved(PRIVATE_TODO_FILE) == 3*chunk

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('in_memory', [True, False])
@pytest.mark.parametrize('chunk', [1, 10])
def test_taskmanager_backup_automatic(PRIVATE_TODO_FILE, in_memory, chunk):

	# Automatic backup on interval:
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, load_into_memory=in_memory, backup_interval=2*chunk) as tm:
		# In order to be able to read the database while it is still open,
		# change the locking_mode here:
		if not in_memory:
			tm.cursor.execute("PRAGMA locking_mode=NORMAL;")
			tm.conn.commit()

		# Do a single chunk of tasks, save them and check if anything has been saved to
		# on-disk file yet:
		task = tm.get_task(classifier=AVALIABLE_CLASSIFIERS[0], chunk=chunk)
		res = _make_fake_result(task)
		tm.save_results(res)

		assert _num_saved(PRIVATE_TODO_FILE) == (0 if in_memory else chunk)

	# Because we have closed the TaskManager, the on-disk file should be fully up-to-date:
	assert _num_saved(PRIVATE_TODO_FILE) == chunk

	with TaskManager(PRIVATE_TODO_FILE, overwrite=False, load_into_memory=in_memory, backup_interval=2*chunk) as tm:
		# In order to be able to read the database while it is still open,
		# change the locking_mode here:
		if not in_memory:
			tm.cursor.execute("PRAGMA locking_mode=NORMAL;")
			tm.conn.commit()

		for _ in range(3):
			task = tm.get_task(classifier=AVALIABLE_CLASSIFIERS[0], chunk=chunk)
			res = _make_fake_result(task)
			tm.save_results(res)

		assert _num_saved(PRIVATE_TODO_FILE) == (3*chunk if in_memory else 4*chunk)

	# Because we have closed the TaskManager, the on-disk file should be fully up-to-date:
	assert _num_saved(PRIVATE_TODO_FILE) == 4*chunk

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

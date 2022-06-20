
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common tests of all classifiers, excluding the Meta Classifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import tempfile
import os
import sqlite3
import shutil
from contextlib import closing
from conftest import capture_run_cli
import starclass

try:
	import mpi4py # noqa: F401
	MPI_AVAILABLE = True
except ImportError:
	MPI_AVAILABLE = False

AVAILABLE_CLASSIFIERS = list(starclass.classifier_list)
AVAILABLE_CLASSIFIERS.remove('meta')

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('classifier', AVAILABLE_CLASSIFIERS) # FIXME:  + ['meta']
def test_classifiers_train_test(monkeypatch, SHARED_INPUT_DIR, classifier):

	stcl = starclass.get_classifier(classifier)

	# Pick out a task to use for testing:
	with starclass.TaskManager(SHARED_INPUT_DIR) as tm:
		task1 = tm.get_task(classifier=classifier, change_classifier=False)
		print(task1)

	with tempfile.TemporaryDirectory(prefix='starclass-testing-') as tmpdir:
		if classifier == 'meta':
			# For the MetaClassifier, we need to manipulate the training-set
			# a little bit before we can train. We have to mimic that
			# all the other classifiers have already been trained and cross-validated
			# in order to fill up the training-set todo-file with probabilities
			# which the MetaClassifier uses for training.
			tsetclass = starclass.get_trainingset('keplerq9v3')
			input_folder = tsetclass.find_input_folder()

			# Create a copy of the root files of the trainings set (ignore that actual data)
			# in the temp. directory:
			tsetdir = os.path.join(tmpdir, os.path.basename(input_folder))
			print("New dummy input folder: %s" % tsetdir)
			os.makedirs(tsetdir)
			for f in os.listdir(input_folder):
				fpath = os.path.join(input_folder, f)
				if os.path.isfile(fpath) and not f.endswith(('.sqlite', '.sqlite-journal')):
					shutil.copy(fpath, tsetdir)

			# Change the environment variable to the temp. dir:
			monkeypatch.setenv("STARCLASS_TSETS", tmpdir)

			# Copy the pre-prepared todo-file to the training-set directory:
			prepared_todo = os.path.join(SHARED_INPUT_DIR, 'meta', 'todo.sqlite')
			new_todo = os.path.join(tsetclass.find_input_folder(), tsetclass._todo_name + '.sqlite')
			shutil.copyfile(prepared_todo, new_todo)

			# Initialize the training-set in the temp folder,
			# and set it to load data as if it was the MetaClassifier:
			tset = tsetclass(tf=0.2, random_seed=42)
			tset.fake_metaclassifier = True
		else:
			tsetclass = starclass.get_trainingset('testing')
			tset = tsetclass(tf=0.2, random_seed=42)

		# Initialize the classifier and run training and testing:
		with stcl(tset=tset, features_cache=None, data_dir=tmpdir) as cl:
			print(cl.data_dir)

			# First make sure we throw an error when trying to classify using
			# an untrained classifier. We are actually calling the "deep" version
			# "do_classify" instead of the wrapper "classify", since the wrapper
			# will catch errors and put them into the results dict instead.
			with pytest.raises(starclass.exceptions.UntrainedClassifierError):
				cl.do_classify({'dummy': 'features', 'which': 'are', 'not': 'used'})

			# Run training:
			cl.train(tset)

			# Check that the features_names list is populated:
			# The classifier will have to provide a list (not the default None)
			print(cl.features_names)
			assert isinstance(cl.features_names, list)
			if classifier == 'slosh': # SLOSH is allowed to not have any feature names
				assert len(cl.features_names) == 0
			else:
				assert len(cl.features_names) > 0

			# Run testing phase:
			cl.test(tset)

			# Check that diagnostics file was generated:
			diag_file = os.path.join(cl.data_dir, 'diagnostics_' + tset.key + '_' + tset.level + '_' + classifier + '.json')
			assert os.path.isfile(diag_file), "Diagnostics file not generated"

			# Check loading of the diagnostics file:
			diag = starclass.io.loadJSON(diag_file)
			assert isinstance(diag, dict)
			assert 'confusion_matrix' in diag
			assert 'roc_best_threshold' in diag

			results1 = cl.classify(task1)
			print(results1)

		# Close the classifier and start it again, it should now load the pre-trained classifier
		# and be able to run tests without training first:
		with stcl(tset=tset, features_cache=None, data_dir=tmpdir) as cl:
			# Check that the features_names list is populated after reloading classifier:
			# The classifier will have to provide a list (not the default None)
			print(cl.features_names)
			assert isinstance(cl.features_names, list)
			if classifier != 'slosh': # SLOSH is allowed to not have any feature names
				assert len(cl.features_names) > 0

			# Run testing phase:
			cl.test(tset, feature_importance=True)

			results2 = cl.classify(task1)
			print(results2)

	# Check that classification results are the same when loading from pre-saved model:
	for key in ('starclass_results', 'classifier', 'priority'):
		assert results1[key] == results2[key], "Non-identical results before and after saving/loading model"

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('classifier', AVAILABLE_CLASSIFIERS)
def test_run_training(PRIVATE_INPUT_DIR, classifier):

	tsetclass = starclass.get_trainingset('testing')
	tset = tsetclass(tf=0.2, random_seed=42)

	with tempfile.TemporaryDirectory(prefix='starclass-testing-') as tmpdir:
		logfile = os.path.join(tmpdir, 'training.log')
		todo_file = os.path.join(PRIVATE_INPUT_DIR, 'todo_run.sqlite')

		# Train the classifier:
		out, err, exitcode = capture_run_cli('run_training.py', [
			'--classifier=' + classifier,
			'--trainingset=testing',
			'--level=L1',
			'--testfraction=0.2',
			'--log=' + logfile,
			'--log-level=info',
			'--output=' + tmpdir
		])
		assert exitcode == 0

		# Check that a log-file was indeed generated:
		assert os.path.isfile(logfile), "Log-file not generated"

		# Check that diagnostics file was generated:
		diag_file = os.path.join(tmpdir, tset.level, tset.key, 'diagnostics_' + tset.key + '_' + tset.level + '_' + classifier + '.json')
		assert os.path.isfile(diag_file), "Diagnostics file not generated"

		# Check loading of the diagnostics file:
		diag = starclass.io.loadJSON(diag_file)
		assert isinstance(diag, dict)
		assert 'confusion_matrix' in diag
		assert 'roc_best_threshold' in diag

		# We now have a trained classifier, so we should be able to run the classification:
		for mpi in ([False, True] if MPI_AVAILABLE else [False]):
			cli = 'run_starclass_mpi.py' if mpi else 'run_starclass.py'
			out, err, exitcode = capture_run_cli(cli, [
				'--debug',
				'--overwrite',
				'--classifier=' + classifier,
				'--trainingset=testing',
				'--level=L1',
				'--datadir=' + tmpdir,
				todo_file
			], mpiexec=mpi)
			assert exitcode == 0

			# Do a deep inspection of the todo-file:
			with closing(sqlite3.connect('file:' + todo_file + '?mode=ro', uri=True)) as conn:
				conn.row_factory = sqlite3.Row
				cursor = conn.cursor()

				cursor.execute("SELECT * FROM starclass_settings;")
				row = cursor.fetchall()
				assert len(row) == 1, "Only one settings row should exist"
				settings = row[0]
				print(dict(settings))
				assert settings['tset'] == 'testtset'

				cursor.execute("SELECT * FROM starclass_diagnostics WHERE priority=17;")
				row = cursor.fetchone()
				print(dict(row))
				assert row['priority'] == 17
				assert row['classifier'] == classifier
				assert row['status'] == starclass.STATUS.OK.value
				assert row['errors'] is None

				cursor.execute("SELECT * FROM starclass_results;")
				results = cursor.fetchall()
				assert len(results) == len(tset.StellarClasses)
				for row in cursor.fetchall():
					print(dict(row))
					assert row['priority'] == 17
					assert row['classifier'] == classifier
					tset.StellarClasses[row['class']] # Will result in KeyError of not correct
					assert 0 <= row['prob'] <= 1, "Invalid probability"

				cursor.execute("SELECT * FROM starclass_features_common;")
				results = cursor.fetchall()
				assert len(results) == 1
				row = dict(results[0])
				print(row)
				assert row['priority'] == 17
				assert len(row) > 1

				if classifier != 'slosh':
					cursor.execute(f"SELECT * FROM starclass_features_{classifier:s};")
					results = cursor.fetchall()
					assert len(results) == 1
					row = dict(results[0])
					print(row)
					assert row['priority'] == 17
					assert len(row) > 1
				else:
					# For SLOSH the table should not exist:
					cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='starclass_features_{classifier:s}';")
					assert len(cursor.fetchall()) == 0

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

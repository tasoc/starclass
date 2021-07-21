
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
from contextlib import closing
from conftest import capture_run_cli
import starclass
from starclass.training_sets.testing_tset import testing_tset

AVALIABLE_CLASSIFIERS = list(starclass.classifier_list)
AVALIABLE_CLASSIFIERS.remove('meta')

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('classifier', AVALIABLE_CLASSIFIERS)
def test_classifiers_train_test(classifier):

	stcl = starclass.get_classifier(classifier)
	tset = testing_tset(tf=0.2, random_seed=42)

	with tempfile.TemporaryDirectory(prefix='starclass-testing-') as tmpdir:
		with stcl(tset=tset, features_cache=None, data_dir=tmpdir) as cl:
			print(cl.data_dir)

			# Run training:
			cl.train(tset)

			# Check that the features_names list is populated:
			# The classifier will have to provide a list (not the default None)
			print(cl.features_names)
			assert isinstance(cl.features_names, list)
			if classifier != 'slosh': # SLOSH is allowed to not have any feature names
				assert len(cl.features_names) > 0

			# Run testing phase:
			cl.test(tset)

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

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('classifier', AVALIABLE_CLASSIFIERS)
def test_run_training(PRIVATE_INPUT_DIR, classifier):

	tset = testing_tset(tf=0.2, random_seed=42)
	print(tset.StellarClasses)

	with tempfile.TemporaryDirectory(prefix='starclass-testing-') as tmpdir:
		logfile = os.path.join(tmpdir, 'training.log')
		todo_file = os.path.join(PRIVATE_INPUT_DIR, 'todo_run.sqlite')

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

		out, err, exitcode = capture_run_cli('run_starclass.py', [
			'--debug',
			'--classifier=' + classifier,
			'--trainingset=testing',
			'--level=L1',
			'--datadir=' + tmpdir,
			todo_file
		])
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

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common tests of all classifiers, excluding the Meta Classifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import tempfile
import os
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

	dd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'starclass', 'data', tset.level))
	with tempfile.TemporaryDirectory(dir=dd) as tmpdir:
		with stcl(tset=tset, features_cache=None, data_dir=os.path.basename(tmpdir)) as cl:
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
		with stcl(tset=tset, features_cache=None, data_dir=os.path.basename(tmpdir)) as cl:
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
def test_run_training(classifier):

	dd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'starclass', 'data', 'L1'))
	with tempfile.TemporaryDirectory(dir=dd, prefix='testing-') as tmpdir:
		output_dir = os.path.basename(tmpdir)
		logfile = os.path.join(tmpdir, 'training.log')

		out, err, exitcode = capture_run_cli('run_training.py', [
			'--classifier=' + classifier,
			'--trainingset=testing',
			'--level=L1',
			'--testfraction=0.2',
			'--log=' + logfile,
			'--log-level=info',
			f'--output={output_dir}'
		])
		assert exitcode == 0

		# Check that a log-file was indeed generated:
		assert os.path.isfile(logfile), "Log-file not generated"

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

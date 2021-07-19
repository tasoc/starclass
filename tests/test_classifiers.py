
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common tests of all classifiers, excluding the Meta Classifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import tempfile
import os
import conftest # noqa: F401
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
if __name__ == '__main__':
	pytest.main([__file__])

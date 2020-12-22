
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
			cl.train(tset)
			cl.test(tset)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

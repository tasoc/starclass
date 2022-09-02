#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of convenience functions.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import conftest # noqa: F401
import starclass

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('classifier', list(starclass.classifier_list) + [
	pytest.param('nonsense', marks=pytest.mark.xfail(raises=ValueError))
])
def test_convenience_get_classifier(classifier):
	stcl = starclass.get_classifier(classifier)
	assert issubclass(stcl, starclass.BaseClassifier)

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('tsetket', list(starclass.trainingset_list) + [
	pytest.param('nonsense', marks=pytest.mark.xfail(raises=ValueError))
])
def test_convenience_get_trainingset(tsetket):
	tsetclass = starclass.get_trainingset(tsetket)
	assert issubclass(tsetclass, starclass.training_sets.TrainingSet)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

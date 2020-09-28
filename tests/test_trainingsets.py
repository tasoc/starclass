#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of Training Sets.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import conftest # noqa: F401
import starclass.training_sets as tsets

#--------------------------------------------------------------------------------------------------
#@pytest.mark.skipif(not tsets.tset_available('keplerq9'), reason='TrainingSet not available')
def test_keplerq9():

	for testfraction in (0, 0.2):
		tset = tsets.keplerq9(tf=testfraction)
		print(tset)

		assert tset.key == 'keplerq9'
		assert tset.datalevel == 'corr'
		assert tset.testfraction == testfraction

	# Test-fractions which should all return in a ValueError:
	with pytest.raises(ValueError):
		tset = tsets.keplerq9(tf=1.2)

	with pytest.raises(ValueError):
		tset = tsets.keplerq9(tf=1.0)

	with pytest.raises(ValueError):
		tset = tsets.keplerq9(tf=-0.2)

	# KeplerQ9 does not support anything other than datalevel=corr
	with pytest.raises(ValueError):
		tset = tsets.keplerq9(datalevel='raw')
	with pytest.raises(ValueError):
		tset = tsets.keplerq9(datalevel='clean')

	# Calling with invalid datalevel should throw an error as well:
	with pytest.raises(ValueError):
		tset = tsets.keplerq9(datalevel='nonsense')

#--------------------------------------------------------------------------------------------------
#@pytest.mark.skipif(not tsets.tset_available('keplerq9v2'), reason='TrainingSet not available')
def test_keplerq9v2():

	for testfraction in (0, 0.2):
		tset = tsets.keplerq9v2(tf=testfraction)
		print(tset)

		assert tset.key == 'keplerq9v2'
		assert tset.datalevel == 'corr'
		assert tset.testfraction == testfraction

	# Test-fractions which should all return in a ValueError:
	with pytest.raises(ValueError):
		tset = tsets.keplerq9v2(tf=1.2)

	with pytest.raises(ValueError):
		tset = tsets.keplerq9v2(tf=1.0)

	with pytest.raises(ValueError):
		tset = tsets.keplerq9v2(tf=-0.2)

	# KeplerQ9 does not support anything other than datalevel=corr
	with pytest.raises(ValueError):
		tset = tsets.keplerq9v2(datalevel='raw')
	with pytest.raises(ValueError):
		tset = tsets.keplerq9v2(datalevel='clean')

	# Calling with invalid datalevel should throw an error as well:
	with pytest.raises(ValueError):
		tset = tsets.keplerq9v2(datalevel='nonsense')

#--------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not tsets.tset_available('keplerq9-linfit'), reason='TrainingSet not available')
def test_keplerq9linfit():

	for testfraction in (0, 0.2):
		tset = tsets.keplerq9linfit(tf=testfraction)
		print(tset)

		assert tset.key == 'keplerq9-linfit'
		assert tset.datalevel == 'corr'
		assert tset.testfraction == testfraction

	# Test-fractions which should all return in a ValueError:
	with pytest.raises(ValueError):
		tset = tsets.keplerq9linfit(tf=1.2)
	with pytest.raises(ValueError):
		tset = tsets.keplerq9linfit(tf=1.0)
	with pytest.raises(ValueError):
		tset = tsets.keplerq9linfit(tf=-0.2)

	# KeplerQ9 does not support anything other than datalevel=corr
	with pytest.raises(ValueError):
		tset = tsets.keplerq9linfit(datalevel='raw')
	with pytest.raises(ValueError):
		tset = tsets.keplerq9linfit(datalevel='clean')

	# Calling with invalid datalevel should throw an error as well:
	with pytest.raises(ValueError):
		tset = tsets.keplerq9linfit(datalevel='nonsense')

#--------------------------------------------------------------------------------------------------
@pytest.mark.skip()
@pytest.mark.parametrize('datalevel', ['corr', 'raw', 'clean'])
def test_tdasim(datalevel):

	for testfraction in (0, 0.2):
		tset = tsets.tda_simulations(datalevel=datalevel, tf=testfraction)
		print(tset)

		assert tset.key == 'tdasim'
		assert tset.datalevel == datalevel
		assert tset.testfraction == testfraction

	# Test-fractions which should all return in a ValueError:
	with pytest.raises(ValueError):
		tset = tsets.tda_simulations(datalevel=datalevel, tf=1.2)
	with pytest.raises(ValueError):
		tset = tsets.tda_simulations(datalevel=datalevel, tf=1.0)
	with pytest.raises(ValueError):
		tset = tsets.tda_simulations(datalevel=datalevel, tf=-0.2)

	# Calling with invalid datalevel should throw an error as well:
	with pytest.raises(ValueError):
		tset = tsets.tda_simulations(datalevel='nonsense')

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('tsetclass', [
	tsets.keplerq9v2,
	tsets.keplerq9,
	pytest.param(tsets.keplerq9linfit, marks=pytest.mark.skipif(not tsets.tset_available('keplerq9linfit'), reason='TrainingSet not available'))
])
def test_trainingset_labels(tsetclass):

	tset = tsetclass(tf=0)
	print(tset)
	lbls = tset.labels()
	lbls_test = tset.labels_test()
	print(tset.nobjects)
	print(len(lbls), len(lbls_test))

	assert len(lbls) == tset.nobjects
	assert len(lbls_test) == 0

	tset = tsetclass(tf=0.2)
	print(tset)
	lbls = tset.labels()
	lbls_test = tset.labels_test()
	print(tset.nobjects)
	print(len(lbls), len(lbls_test))

	assert len(lbls) + len(lbls_test) == tset.nobjects

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

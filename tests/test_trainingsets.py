#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of Training Sets.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import starclass.training_sets as tsets

#----------------------------------------------------------------------
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

#----------------------------------------------------------------------
@pytest.mark.skip()
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

#----------------------------------------------------------------------
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

#----------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

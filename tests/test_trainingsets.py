#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of Training Sets.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os
import shutil
import tempfile
import conftest # noqa: F401
import starclass.training_sets as tsets
from starclass import trainingset_available, get_trainingset

AVAILABLE_TSETS = [
	'keplerq9v3',
	pytest.param('keplerq9v3-instr', marks=pytest.mark.skip),
	pytest.param('keplerq9v2', marks=pytest.mark.skipif(not trainingset_available('keplerq9v2'), reason='TrainingSet not available')),
	pytest.param('keplerq9', marks=pytest.mark.skipif(not trainingset_available('keplerq9'), reason='TrainingSet not available')),
	pytest.param('keplerq9-linfit', marks=pytest.mark.skipif(not trainingset_available('keplerq9-linfit'), reason='TrainingSet not available')),
	pytest.param('tdasim', marks=pytest.mark.skip),
	pytest.param('tdasim-raw', marks=pytest.mark.skip),
	pytest.param('tdasim-clean', marks=pytest.mark.skip)
]

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('tsetkey', AVAILABLE_TSETS)
def test_trainingset(tsetkey):

	# Get training set class using conv. function:
	tsetclass = get_trainingset(tsetkey)

	for testfraction in (0, 0.2):
		tset = tsetclass(tf=testfraction)
		print(tset)

		assert tset.key == tsetkey
		assert tset.datalevel == 'corr'
		assert tset.testfraction == testfraction

	# Test-fractions which should all return in a ValueError:
	with pytest.raises(ValueError):
		tset = tsetclass(tf=1.2)
	with pytest.raises(ValueError):
		tset = tsetclass(tf=1.0)
	with pytest.raises(ValueError):
		tset = tsetclass(tf=-0.2)

	# Calling with invalid datalevel should throw an error as well:
	with pytest.raises(ValueError):
		tset = tsets.keplerq9linfit(datalevel='nonsense')

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
@pytest.mark.parametrize('tsetkey', AVAILABLE_TSETS)
def test_trainingset_generate_todolist(monkeypatch, tsetkey):

	# Get training set class using conv. function:
	tsetclass = get_trainingset(tsetkey)

	with tempfile.TemporaryDirectory(prefix='pytest-private-tsets-') as tmpdir:

		tset = tsetclass()
		input_folder = tset.input_folder
		print(input_folder)

		# Create a copy of the root fies of the trainings set (ignore that actual data)
		# in the temp. directory:
		tsetdir = os.path.join(tmpdir, tsetkey)
		os.makedirs(tsetdir)
		for f in os.listdir(input_folder):
			fpath = os.path.join(input_folder, f)
			if os.path.isdir(fpath) or f == 'todo.sqlite':
				continue
			shutil.copy(fpath, tsetdir)

		# Change the environment variable to the temp. dir:
		monkeypatch.setenv("STARCLASS_TSETS", tmpdir)
		print(os.environ['STARCLASS_TSETS'])

		# When we now initialize the trainingset it should run generate_todo automatically:
		tset = tsetclass()

		assert tset.input_folder == tsetdir
		assert os.path.isfile(os.path.join(tsetdir, 'todo.sqlite'))

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('tsetkey', AVAILABLE_TSETS)
def test_trainingset_folds(tsetkey):

	# Get training set class using conv. function:
	tsetclass = get_trainingset(tsetkey)
	tset = tsetclass()

	for k, fold in enumerate(tset.folds(n_splits=5, tf=0.2)):
		assert isinstance(fold, tsetclass)
		assert fold.crossval_folds == 5
		assert fold.fold == k + 1
		assert fold.testfraction == 0.2
		assert len(fold.train_idx) > 0
		assert len(fold.test_idx) > 0
		assert len(fold.train_idx) > len(fold.test_idx)
		assert len(fold.train_idx) < len(tset.train_idx)

	assert k == 4, "Not the correct number of folds"

#--------------------------------------------------------------------------------------------------
#@pytest.mark.skipif(not trainingset_available('keplerq9'), reason='TrainingSet not available')
def test_keplerq9():

	# KeplerQ9 does not support anything other than datalevel=corr
	with pytest.raises(ValueError):
		tsets.keplerq9(datalevel='raw')
	with pytest.raises(ValueError):
		tsets.keplerq9(datalevel='clean')

#--------------------------------------------------------------------------------------------------
#@pytest.mark.skipif(not trainingset_available('keplerq9v2'), reason='TrainingSet not available')
def test_keplerq9v2():

	# KeplerQ9 does not support anything other than datalevel=corr
	with pytest.raises(ValueError):
		tsets.keplerq9v2(datalevel='raw')
	with pytest.raises(ValueError):
		tsets.keplerq9v2(datalevel='clean')

#--------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not trainingset_available('keplerq9-linfit'), reason='TrainingSet not available')
def test_keplerq9linfit():

	# KeplerQ9 does not support anything other than datalevel=corr
	with pytest.raises(ValueError):
		tsets.keplerq9linfit(datalevel='raw')
	with pytest.raises(ValueError):
		tsets.keplerq9linfit(datalevel='clean')

#--------------------------------------------------------------------------------------------------
@pytest.mark.skip()
@pytest.mark.parametrize('datalevel', ['corr', 'raw', 'clean'])
def test_tdasim(datalevel):

	for testfraction in (0, 0.2):
		tset = tsets.tdasim(datalevel=datalevel, tf=testfraction)
		print(tset)

		assert tset.key == 'tdasim'
		assert tset.datalevel == datalevel
		assert tset.testfraction == testfraction

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

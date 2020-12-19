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
import types
import sqlite3
from contextlib import closing
import conftest # noqa: F401
import starclass.training_sets as tsets
from starclass import trainingset_available, get_trainingset

AVAILABLE_TSETS = [
	'keplerq9v3',
	'keplerq9v3-instr',
	pytest.param('keplerq9v2', marks=pytest.mark.skipif(not trainingset_available('keplerq9v2'), reason='TrainingSet not available')),
	pytest.param('keplerq9', marks=pytest.mark.skipif(not trainingset_available('keplerq9'), reason='TrainingSet not available')),
	pytest.param('tdasim', marks=pytest.mark.skip),
	pytest.param('tdasim-raw', marks=pytest.mark.skip),
	pytest.param('tdasim-clean', marks=pytest.mark.skip)
]

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('tsetkey', AVAILABLE_TSETS)
@pytest.mark.parametrize('linfit', [False, True])
def test_trainingset(tsetkey, linfit):

	# Get training set class using conv. function:
	tsetclass = get_trainingset(tsetkey)

	for testfraction in (0, 0.2):
		tset = tsetclass(tf=testfraction, linfit=linfit)
		print(tset)

		if linfit:
			assert tset.key == tsetkey + '-linfit'
		else:
			assert tset.key == tsetkey
		assert tset.level == 'L1'
		assert tset.datalevel == 'corr'
		assert tset.testfraction == testfraction
		assert len(tset) > 0

	# Invalid level should give ValueError:
	with pytest.raises(ValueError):
		tsetclass(level='nonsense')

	# Test-fractions which should all return in a ValueError:
	with pytest.raises(ValueError):
		tset = tsetclass(tf=1.2)
	with pytest.raises(ValueError):
		tset = tsetclass(tf=1.0)
	with pytest.raises(ValueError):
		tset = tsetclass(tf=-0.2)

	# Calling with invalid datalevel should throw an error as well:
	with pytest.raises(ValueError):
		tset = tsetclass(datalevel='nonsense')

	tset = tsetclass(tf=0, linfit=linfit)
	print(tset)
	lbls = tset.labels()
	lbls_test = tset.labels_test()
	print(tset.nobjects)
	print(len(lbls), len(lbls_test))

	assert len(lbls) == tset.nobjects
	assert len(lbls_test) == 0

	tset = tsetclass(tf=0.2, linfit=linfit)
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
	tset = tsetclass()
	input_folder = tset.input_folder
	print("Training Set input folder: %s" % input_folder)

	with tempfile.TemporaryDirectory(prefix='pytest-private-tsets-') as tmpdir:
		# Create a copy of the root fies of the trainings set (ignore that actual data)
		# in the temp. directory:
		tsetdir = os.path.join(tmpdir, os.path.basename(input_folder))
		print("New dummy input folder: %s" % tsetdir)
		os.makedirs(tsetdir)
		for f in os.listdir(input_folder):
			fpath = os.path.join(input_folder, f)
			if f.endswith('.sqlite'):
				continue
			if os.path.isdir(fpath):
				# NOTE: We are cheating, and creating an empty file with
				# the correct name, since the file is actually not
				# needed for building the todolist, it only needs to exist.
				os.makedirs(os.path.join(tsetdir, f))
				for subf in os.listdir(fpath):
					open(os.path.join(tsetdir, f, subf), 'w').close()
			else:
				shutil.copy(fpath, tsetdir)

		# Change the environment variable to the temp. dir:
		monkeypatch.setenv("STARCLASS_TSETS", tmpdir)
		print(os.environ['STARCLASS_TSETS'])

		# When we now initialize the trainingset it should run generate_todo automatically:
		tset = tsetclass()

		assert tset.input_folder == tsetdir
		assert os.path.isfile(os.path.join(tsetdir, tset._todo_name + '.sqlite'))

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('tsetkey', AVAILABLE_TSETS)
def test_trainingset_generate_todolist_insert(SHARED_INPUT_DIR, tsetkey):

	# Get training set class using conv. function:
	tsetclass = get_trainingset(tsetkey)
	tset = tsetclass()

	with tempfile.TemporaryDirectory(prefix='pytest-private-tsets-') as tmpdir:
		with closing(sqlite3.connect(os.path.join(tmpdir, 'todo.sqlite'))) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()

			tset.generate_todolist_structure(conn)

			with pytest.raises(ValueError):
				tset.generate_todolist_insert(cursor, priority=None)

			with pytest.raises(ValueError):
				tset.generate_todolist_insert(cursor, priority=1, lightcurve=None)

			lightcurve = os.path.join(SHARED_INPUT_DIR, 'tess00029281992-s01-c1800-dr01-v04-tasoc-cbv_lc.fits.gz')
			tset.generate_todolist_insert(cursor,
				priority=2187,
				starid=12345678,
				tmag=15.6,
				lightcurve=lightcurve,
				datasource='tpf',
				variance=3.14,
				rms_hour=2.71,
				ptp=42.0)

			# TODOLIST table:
			cursor.execute("SELECT * FROM todolist WHERE priority=2187;")
			row = cursor.fetchone()
			assert row['priority'] == 2187
			assert row['starid'] == 12345678
			assert row['tmag'] == 15.6
			assert row['datasource'] == 'tpf'
			assert row['camera'] == 1 # These are constant!
			assert row['ccd'] == 1 # These are constant!
			assert row['cbv_area'] == 111 # These are constant!
			assert row['status'] == 1 # These are constant!
			assert row['corr_status'] == 1 # These are constant!

			# DIAGNOSTICS_CORR table:
			cursor.execute("SELECT * FROM diagnostics_corr WHERE priority=2187;")
			row = cursor.fetchone()
			assert row['lightcurve'] == lightcurve
			assert row['variance'] == 3.14
			assert row['rms_hour'] == 2.71
			assert row['ptp'] == 42.0

			# DATAVALIDATION_CORR table:
			cursor.execute("SELECT * FROM datavalidation_corr WHERE priority=2187;")
			row = cursor.fetchone()
			assert row['approved'] == 1 # These are constant!
			assert row['dataval'] == 0 # These are constant!

			tset.generate_todolist_insert(cursor,
				priority=2188,
				lightcurve=lightcurve)

			# TODOLIST table:
			cursor.execute("SELECT * FROM todolist WHERE priority=2188;")
			row = cursor.fetchone()
			assert row['priority'] == 2188
			assert row['starid'] == 2188 # When not provided, will used priority
			assert row['tmag'] == -99
			assert row['datasource'] == 'ffi'

			# DIAGNOSTICS_CORR table:
			cursor.execute("SELECT * FROM diagnostics_corr WHERE priority=2188;")
			row = cursor.fetchone()
			assert row['lightcurve'] == lightcurve
			assert row['variance'] > 0
			assert row['rms_hour'] > 0
			assert row['ptp'] > 0

			# DATAVALIDATION_CORR table:
			cursor.execute("SELECT * FROM datavalidation_corr WHERE priority=2188;")
			row = cursor.fetchone()
			assert row['approved'] == 1 # These are constant!
			assert row['dataval'] == 0 # These are constant!

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('tsetkey', AVAILABLE_TSETS)
@pytest.mark.parametrize('linfit', [False, True])
def test_trainingset_features(tsetkey, linfit):

	# Get training set class using conv. function:
	tsetclass = get_trainingset(tsetkey)
	tset = tsetclass(tf=0.2, linfit=linfit)

	features = tset.features()
	assert isinstance(features, types.GeneratorType)

	features_test = tset.features_test()
	assert isinstance(features_test, types.GeneratorType)

	for tries in range(2):
		feat = next(features)
		print(feat)
		assert isinstance(feat, dict)
		assert 'lightcurve' in feat
		assert 'powerspectrum' in feat
		assert 'frequencies' in feat

		feat = next(features_test)
		print(feat)
		assert isinstance(feat, dict)
		assert 'lightcurve' in feat
		assert 'powerspectrum' in feat
		assert 'frequencies' in feat

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
		assert fold.level == tset.level
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

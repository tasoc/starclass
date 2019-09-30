#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from numpy import ceil
import numpy as np
import os
import requests
import zipfile
import shutil
import logging
import tempfile
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle as sklshuffle
from .. import BaseClassifier, TaskManager

#----------------------------------------------------------------------------------------------
class TrainingSet(object):

	def __init__(self, datalevel='corr', tf=0.0):

		self.datalevel = datalevel
		self.testfraction = tf

		if self.testfraction < 0 or self.testfraction >= 1:
			raise ValueError("Invalid testfraction provided")

		# Define cache location where we will save common features:
		self.features_cache = os.path.join(self.input_folder, 'features_cache_%s' % self.datalevel)
		os.makedirs(self.features_cache, exist_ok=True)

		# Generate TODO file if it is needed:
		sqlite_file = os.path.join(self.input_folder, 'todo.sqlite')
		if not os.path.exists(sqlite_file):
			self.generate_todolist()

		# Generate training/test indices
		self.train_idx = np.arange(self.nobjects, dtype=int) # Define here because it is needed by self.labels() used below
		if self.testfraction > 0:
			self.train_idx, self.test_idx = train_test_split(
				np.arange(self.nobjects),
				test_size=self.testfraction,
				random_state=42,
				stratify=self.labels()
			)

		# Cross Validation
		self.fold = 0
		self.crossval_folds = 0

	#----------------------------------------------------------------------------------------------
	def folds(self, n_splits=5, tf=0.2):
		"""

		"""

		labels_test = [lbl[0].value for lbl in self.labels()]

		# If keyword is true then split according to KFold cross-vadliation
		skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
		skf_splits = skf.split(self.train_idx, labels_test)

		# We are doing cross-validation, so we will return a copy
		# of the training-set where we have redefined the training- and test-
		# sets from the folding:
		for fold, (train_idx, test_idx) in enumerate(skf_splits):
			#newtset = deepcopy(self)
			newtset = self.__class__(datalevel=self.datalevel, tf=tf)
			newtset.train_idx = self.train_idx[train_idx]
			newtset.test_idx = self.train_idx[test_idx]
			newtset.crossval_folds = n_splits
			newtset.fold = fold + 1
			yield newtset

	#----------------------------------------------------------------------------------------------
	def tset_datadir(self, tset, url):

		logger = logging.getLogger(__name__)

		# Point this to the directory where the TDA simulations are stored
		INPUT_DIR = os.environ.get('STARCLASS_TSETS')
		if INPUT_DIR is None:
			INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
		elif not os.path.exists(INPUT_DIR) or not os.path.isdir(INPUT_DIR):
			raise IOError("The environment variable STARCLASS_TSETS is set, but points to a non-existent directory.")

		input_folder = os.path.join(INPUT_DIR, tset)

		if not os.path.exists(input_folder):
			logger.info("Downloading training set...")
			zip_tmp = os.path.join(input_folder, tset + '.zip')
			try:
				os.makedirs(input_folder)

				res = requests.get(url, stream=True)
				res.raise_for_status()
				total_size = int(res.headers.get('content-length', 0));
				block_size = 1024
				with open(zip_tmp, 'wb') as fid:
					for data in tqdm(res.iter_content(block_size), total=ceil(total_size/block_size), unit='KB', unit_scale=True):
						fid.write(data)

				# Extract ZIP file:
				with zipfile.ZipFile(zip_tmp, 'r') as zip:
					zip.extractall(input_folder)

			except:
				if os.path.exists(input_folder):
					shutil.rmtree(input_folder)
				raise

			finally:
				if os.path.exists(zip_tmp):
					os.remove(zip_tmp)

		return input_folder

	#----------------------------------------------------------------------------------------------
	def generate_todolist(self):
		raise NotImplementedError()

	#----------------------------------------------------------------------------------------------
	def features(self):

		# Create a temporary copy of the TODO-file that we are going to read from.
		# This is due to errors we have detected, where the database is unexpectively locked
		# when opened several times in parallel.
		try:
			with tempfile.NamedTemporaryFile(dir=self.input_folder, suffix='.sqlite', delete=False) as tmpdir:
				# Copy the original TODO-file to the new temp file:
				with open(os.path.join(self.input_folder, 'todo.sqlite'), 'rb') as fid:
					shutil.copyfileobj(fid, tmpdir)
				tmpdir.flush()

				with TaskManager(tmpdir.name, readonly=True, overwrite=False, cleanup=False) as tm:
					with BaseClassifier(tset_key=self.key, features_cache=self.features_cache) as stcl:
						for rowidx in self.train_idx:
							task = tm.get_task(priority=rowidx+1, change_classifier=False)
							if task is None: break

							# Lightcurve file to load:
							# We do not use the one from the database because in the simulations the
							# raw and corrected light curves are stored in different files.
							fname = os.path.join(self.input_folder, task['lightcurve'])
							yield stcl.load_star(task, fname)

		finally:
			if os.path.exists(tmpdir.name):
				os.remove(tmpdir.name)

	#----------------------------------------------------------------------------------------------
	def features_test(self):

		if self.testfraction <= 0:
			raise ValueError('features_test requires testfraction>0')

		# Create a temporary copy of the TODO-file that we are going to read from.
		# This is due to errors we have detected, where the database is unexpectively locked
		# when opened several times in parallel.
		try:
			with tempfile.NamedTemporaryFile(dir=self.input_folder, suffix='.sqlite', delete=False) as tmpdir:
				# Copy the original TODO-file to the new temp file:
				with open(os.path.join(self.input_folder, 'todo.sqlite'), 'rb') as fid:
					shutil.copyfileobj(fid, tmpdir)
				tmpdir.flush()

				with TaskManager(tmpdir.name, readonly=True, overwrite=False, cleanup=False) as tm:
					with BaseClassifier(tset_key=self.key, features_cache=self.features_cache) as stcl:
						for rowidx in self.test_idx:
							task = tm.get_task(priority=rowidx+1, change_classifier=False)
							if task is None: break

							# Lightcurve file to load:
							# We do not use the one from the database because in the simulations the
							# raw and corrected light curves are stored in different files.
							fname = os.path.join(self.input_folder, task['lightcurve'])
							yield stcl.load_star(task, fname)

		finally:
			if os.path.exists(tmpdir.name):
				os.remove(tmpdir.name)

	#----------------------------------------------------------------------------------------------
	def labels(self, level='L1'):
		raise NotImplementedError()

	#----------------------------------------------------------------------------------------------
	def labels_test(self, level='L1'):
		raise NotImplementedError()

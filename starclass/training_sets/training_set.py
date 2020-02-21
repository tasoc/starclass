#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Sets.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from bottleneck import nanmedian, nanvar
import os
import requests
import zipfile
import shutil
import logging
import tempfile
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from .. import BaseClassifier, TaskManager, utilities

#----------------------------------------------------------------------------------------------
class TrainingSet(object):

	def __init__(self, datalevel='corr', tf=0.0, random_seed=42):
		"""
		Parameters:
			datalevel (string, optional):
			tf (float, optional): Test-fraction. Default=0.
			random_seed (optional): Random seed. Default=42.
		"""

		# Basic checks of input:
		if tf < 0 or tf >= 1:
			raise ValueError("Invalid TESTFRACTION provided.")

		if datalevel not in ('corr', 'raw', 'clean'):
			raise ValueError("Invalid DATALEVEL provided.")

		# Store input:
		self.datalevel = datalevel
		self.testfraction = tf
		self.random_seed = random_seed

		# Define cache location where we will save common features:
		self.features_cache = os.path.join(self.input_folder, 'features_cache_%s' % self.datalevel)
		os.makedirs(self.features_cache, exist_ok=True)

		# Generate TODO file if it is needed:
		sqlite_file = os.path.join(self.input_folder, 'todo.sqlite')
		if not os.path.exists(sqlite_file):
			self.generate_todolist()

		# Generate training/test indices
		self.train_idx = np.arange(self.nobjects, dtype=int) # Define here because it is needed by self.labels() used below
		self.test_idx = np.array([], dtype=int)
		if self.testfraction > 0:
			self.train_idx, self.test_idx = train_test_split(
				np.arange(self.nobjects),
				test_size=self.testfraction,
				random_state=self.random_seed,
				stratify=self.labels()
			)
			# Have to sort as train_test_split shuffles and we don't want that
			self.train_idx = np.sort(self.train_idx)
			self.test_idx = np.sort(self.test_idx)
		# Cross Validation
		self.fold = 0
		self.crossval_folds = 0

	#----------------------------------------------------------------------------------------------
	def __str__(self):
		return "<TrainingSet({key:s}, {datalevel:s}, tf={tf:.2f})>".format(
			key=self.key,
			datalevel=self.datalevel,
			tf=self.testfraction
		)

	#----------------------------------------------------------------------------------------------
	def folds(self, n_splits=5, tf=0.2):
		"""
		Split training set object into stratified folds.

		Parameters:
			n_splits (integer, optional): Number of folds to split training set into. Default=5.
			tf (real, optional): Test-fraction, between 0 and 1, to split from each fold.

		Returns:
			Iterator of ``TrainingSet`` objects: Iterator of folds, which are also ``TrainingSet`` objects.
		"""

		logger = logging.getLogger(__name__)

		labels_test = [lbl[0].value for lbl in self.labels()]

		# If keyword is true then split according to KFold cross-validation
		skf = StratifiedKFold(n_splits=n_splits, random_state=self.random_seed, shuffle=True)
		skf_splits = skf.split(self.train_idx, labels_test)

		# We are doing cross-validation, so we will return a copy
		# of the training-set where we have redefined the training- and test-
		# sets from the folding:
		for fold, (train_idx, test_idx) in enumerate(skf_splits):
			# Write some debug information:
			logger.debug("Fold %d: Training set=%d, Test set=%d", fold+1, len(train_idx), len(test_idx))

			# Set tf to be zero here so the training set isn't further split
			# as want to run all the data through CV
			newtset = self.__class__(datalevel=self.datalevel, tf=0.0)

			# Set testfraction to value from CV i.e. 1/n_splits
			newtset.testfraction = tf
			newtset.train_idx = self.train_idx[train_idx]
			newtset.test_idx = self.train_idx[test_idx]
			newtset.crossval_folds = n_splits
			newtset.fold = fold + 1
			yield newtset

	#----------------------------------------------------------------------------------------------
	def tset_datadir(self, tset, url):
		"""
		Setup TrainingSet data directory. If the directory doesn't already exist,

		Parameters:
			tset (string): Name of TrainingSet folder.
			url (string): URL from where to download the training-set if it doesn't already exist.

		Returns:
			string: Path to directory where training set is stored.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		# Point this to the directory where the TDA simulations are stored
		INPUT_DIR = os.environ.get('STARCLASS_TSETS')
		if INPUT_DIR is None:
			INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
		elif not os.path.exists(INPUT_DIR) or not os.path.isdir(INPUT_DIR):
			raise IOError("The environment variable STARCLASS_TSETS is set, but points to a non-existent directory.")

		input_folder = os.path.join(INPUT_DIR, tset)

		tqdm_settings = {
			'unit': 'KB',
			'unit_scale': True,
			'disable': not logger.isEnabledFor(logging.INFO)
		}

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
					for data in tqdm(res.iter_content(block_size), total=np.ceil(total_size/block_size), **tqdm_settings):
						fid.write(data)

				# Extract ZIP file:
				logger.info("Unpacking training set...")
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
	def generate_todolist_structure(self, conn):

		cursor = conn.cursor()
		cursor.execute("PRAGMA foreign_keys=ON;")

		cursor.execute("""CREATE TABLE todolist (
			priority INTEGER PRIMARY KEY NOT NULL,
			starid BIGINT NOT NULL,
			datasource TEXT NOT NULL DEFAULT 'ffi',
			camera INTEGER NOT NULL,
			ccd INTEGER NOT NULL,
			method TEXT DEFAULT NULL,
			tmag REAL,
			status INTEGER DEFAULT NULL,
			corr_status INTEGER DEFAULT NULL,
			cbv_area INTEGER NOT NULL
		);""")
		cursor.execute("CREATE INDEX status_idx ON todolist (status);")
		cursor.execute("CREATE INDEX corr_status_idx ON todolist (corr_status);")
		cursor.execute("CREATE INDEX starid_idx ON todolist (starid);")

		cursor.execute("""CREATE TABLE diagnostics_corr (
			priority INTEGER PRIMARY KEY NOT NULL,
			lightcurve TEXT,
			elaptime REAL,
			worker_wait_time REAL,
			variance REAL,
			rms_hour REAL,
			ptp REAL,
			errors TEXT,
			FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
		);""")

		cursor.execute("""CREATE TABLE datavalidation_corr (
			priority INTEGER PRIMARY KEY NOT NULL,
			approved BOOLEAN NOT NULL,
			dataval INTEGER NOT NULL,
			FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
		);""")
		cursor.execute("CREATE INDEX datavalidation_corr_approved_idx ON datavalidation_corr (approved);")

		conn.commit()

	#----------------------------------------------------------------------------------------------
	def generate_todolist_insert(self, cursor, priority=None, starid=None, tmag=None,
		lightcurve=None, datasource=None, variance=None, rms_hour=None, ptp=None):

		if priority is None:
			raise ValueError("priority is required")
		if starid is None:
			starid = priority

		# Try to load the lightcurve using the BaseClassifier method.
		# This will ensure that the lightcurve can actually be read by the system.
		if not all([datasource, variance, rms_hour, ptp]):
			with BaseClassifier(tset_key=self.key, features_cache=None) as bc:
				print(os.path.join(self.input_folder, lightcurve))
				fake_task = {
					'priority': priority,
					'starid': starid
				}
				features = bc.load_star(fake_task, os.path.join(self.input_folder, lightcurve))
				lc = features['lightcurve']

		elaptime = np.random.normal(3.14, 0.5)
		if tmag is None:
			tmag = -99
		if variance is None:
			variance = nanvar(lc.flux, ddof=1)
		if rms_hour is None:
			rms_hour = utilities.rms_timescale(lc)
		if ptp is None:
			ptp = nanmedian(np.abs(np.diff(lc.flux)))

		if datasource is None:
			if (lc.time[1] - lc.time[0])*86400 > 1000:
				datasource = 'ffi'
			else:
				datasource = 'tpf'

		#camera = 1
		#if ecllat < 6+24:
		#	camera = 1
		#elif ecllat < 6+2*24:
		#	camera = 2
		#elif ecllat < 6+3*24:
		#	camera = 3
		#else:
		#	camera = 4

		cursor.execute("INSERT INTO todolist (priority,starid,tmag,datasource,status,corr_status,camera,ccd,cbv_area) VALUES (?,?,?,?,1,1,1,1,111);", (
			priority,
			starid,
			tmag,
			datasource
		))
		cursor.execute("INSERT INTO diagnostics_corr (priority,lightcurve,elaptime,variance,rms_hour,ptp) VALUES (?,?,?,?,?,?);", (
			priority,
			lightcurve,
			elaptime,
			variance,
			rms_hour,
			ptp
		))
		cursor.execute("INSERT INTO datavalidation_corr (priority,approved,dataval) VALUES (?,1,0);", (priority,))

	#----------------------------------------------------------------------------------------------
	def features(self):
		"""
		Iterator of features for training.

		Returns:
			Iterator: Iterator of dicts containing features to be used for training.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

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
			if os.path.exists(tmpdir.name + '-journal'):
				os.remove(tmpdir.name + '-journal')

	#----------------------------------------------------------------------------------------------
	def features_test(self):
		"""
		Iterator of features for testing.

		Returns:
			Iterator: Iterator of dicts containing features to be used for testing.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		if self.testfraction <= 0:
			raise ValueError('features_test requires testfraction > 0')

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
			if os.path.exists(tmpdir.name + '-journal'):
				os.remove(tmpdir.name + '-journal')

	#----------------------------------------------------------------------------------------------
	def labels(self, level='L1'):
		raise NotImplementedError()

	#----------------------------------------------------------------------------------------------
	def labels_test(self, level='L1'):
		raise NotImplementedError()

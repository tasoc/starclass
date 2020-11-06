#!/usr/bin/env python3
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
import sqlite3
from contextlib import closing
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from .. import BaseClassifier, TaskManager, utilities
from ..StellarClasses import StellarClassesLevel1, StellarClassesLevel2

#--------------------------------------------------------------------------------------------------
class TrainingSet(object):
	"""
	Generic Training Set.

	Attributes:
		testfraction (float):
		random_seed (int):
		features_cache (str):
		train_idx (ndarray):
		test_idx (ndarray):
		crossval_folds (int):
		fold (int):
	"""

	def __init__(self, level='L1', datalevel='corr', tf=0.0, random_seed=42):
		"""
		Parameters:
			datalevel (string, optional):
			tf (float, optional): Test-fraction. Default=0.
			random_seed (optional): Random seed. Default=42.
		"""

		if not hasattr(self, 'key'):
			raise Exception("Training set class does not have 'key' definied.")

		# Basic checks of input:
		if level not in ('L1', 'L2'):
			raise ValueError("Invalid LEVEL")

		if tf < 0 or tf >= 1:
			raise ValueError("Invalid TESTFRACTION provided.")

		if datalevel not in ('corr', 'raw', 'clean'):
			raise ValueError("Invalid DATALEVEL provided.")

		# Store input:
		self.level = level
		self.datalevel = datalevel
		self.testfraction = tf
		self.random_seed = random_seed

		# Assign StellarClasses Enum depending on
		# the classification level we are running:
		if not hasattr(self, 'StellarClasses'):
			self.StellarClasses = {
				'L1': StellarClassesLevel1,
				'L2': StellarClassesLevel2
			}[self.level]

		# Define cache location where we will save common features:
		self.features_cache = os.path.join(self.input_folder, 'features_cache_%s' % self.datalevel)
		os.makedirs(self.features_cache, exist_ok=True)

		# Generate TODO file if it is needed:
		sqlite_file = os.path.join(self.input_folder, 'todo.sqlite')
		if not os.path.isfile(sqlite_file):
			self.generate_todolist()

		# Generate training/test indices
		# Define here because it is needed by self.labels() used below
		if hasattr(self, '_valid_indicies'):
			self.train_idx = self._valid_indicies
		else:
			self.train_idx = np.arange(self.nobjects, dtype=int)
		self.test_idx = np.array([], dtype=int)
		if self.testfraction > 0:
			self.train_idx, self.test_idx = train_test_split(
				self.train_idx,
				test_size=self.testfraction,
				random_state=self.random_seed,
				stratify=self.labels()
			)

		# Cross Validation
		self.fold = 0
		self.crossval_folds = 0

	#----------------------------------------------------------------------------------------------
	def __str__(self):
		str_fold = '' if self.fold == 0 else ', fold={0:d}/{1:d}'.format(self.fold, self.crossval_folds)
		return "<TrainingSet({key:s}, {datalevel:s}, tf={tf:.2f}{fold:s})>".format(
			key=self.key,
			datalevel=self.datalevel,
			tf=self.testfraction,
			fold=str_fold
		)

	#----------------------------------------------------------------------------------------------
	def __len__(self):
		return self.nobjects

	#----------------------------------------------------------------------------------------------
	def folds(self, n_splits=5, tf=0.2):
		"""
		Split training set object into stratified folds.

		Parameters:
			n_splits (int, optional): Number of folds to split training set into. Default=5.
			tf (float, optional): Test-fraction, between 0 and 1, to split from each fold.

		Returns:
			Iterator of :class:`TrainingSet` objects: Iterator of folds, which are also
				:class:`TrainingSet` objects.
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
			newtset = self.__class__(level=self.level, datalevel=self.datalevel, tf=0.0)

			# Set testfraction to value from CV i.e. 1/n_splits
			newtset.testfraction = tf
			newtset.train_idx = self.train_idx[train_idx]
			newtset.test_idx = self.train_idx[test_idx]
			newtset.crossval_folds = n_splits
			newtset.fold = fold + 1
			yield newtset

	#----------------------------------------------------------------------------------------------
	@classmethod
	def find_input_folder(cls):
		"""
		Find the folder containing the data for the training set.

		This is a class method, so it can be called without having to initialize the training set.
		"""
		if not hasattr(cls, 'key'):
			raise Exception("Training set class does not have 'key' definied.")

		# Point this to the directory where the training set data are stored
		INPUT_DIR = os.environ.get('STARCLASS_TSETS')
		if INPUT_DIR is None:
			INPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
		elif not os.path.exists(INPUT_DIR) or not os.path.isdir(INPUT_DIR):
			raise IOError("The environment variable STARCLASS_TSETS is set, but points to a non-existent directory.")

		datadir = cls.key if not hasattr(cls, 'datadir') else cls.datadir
		return os.path.join(INPUT_DIR, datadir)

	#----------------------------------------------------------------------------------------------
	def tset_datadir(self, url):
		"""
		Setup TrainingSet data directory. If the directory doesn't already exist,

		Parameters:
			url (string): URL from where to download the training-set if it doesn't already exist.

		Returns:
			string: Path to directory where training set is stored.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		if not hasattr(self, 'key'):
			raise Exception("Training set class does not have 'key' definied.")

		logger = logging.getLogger(__name__)
		tqdm_settings = {
			'unit': 'B',
			'unit_scale': True,
			'unit_divisor': 1024,
			'disable': not logger.isEnabledFor(logging.INFO)
		}

		# Find folder where training set is stored:
		input_folder = self.find_input_folder()

		if not os.path.exists(input_folder):
			logger.info("Step 1: Downloading %s training set...", self.key)
			zip_tmp = os.path.join(input_folder, self.key + '.zip')
			try:
				os.makedirs(input_folder)

				res = requests.get(url, stream=True)
				res.raise_for_status()
				total_size = int(res.headers.get('content-length', 0))
				block_size = 1024
				with tqdm(total=total_size, **tqdm_settings) as pbar:
					with open(zip_tmp, 'wb') as fid:
						for data in res.iter_content(block_size):
							datasize = fid.write(data)
							pbar.update(datasize)

				# Extract ZIP file:
				logger.info("Step 2: Unpacking %s training set...", self.key)
				with zipfile.ZipFile(zip_tmp, 'r') as myzip:
					for fileName in tqdm(myzip.namelist(), disable=not logger.isEnabledFor(logging.INFO)):
						myzip.extract(fileName, input_folder)

			except: # noqa: E722, pragma: no cover
				if os.path.exists(input_folder):
					shutil.rmtree(input_folder)
				raise

			finally:
				if os.path.exists(zip_tmp):
					os.remove(zip_tmp)

		return input_folder

	#----------------------------------------------------------------------------------------------
	def generate_todolist(self):
		"""
		Generate todo.sqlite file in training set directory.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		logger = logging.getLogger(__name__)

		sqlite_file = os.path.join(self.input_folder, 'todo.sqlite')
		with closing(sqlite3.connect(sqlite_file)) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()

			# Create the basic file structure of a TODO-list:
			self.generate_todolist_structure(conn)

			logger.info("Step 3: Reading file and extracting information...")
			pri = 0

			diagnostics = np.genfromtxt(os.path.join(self.input_folder, 'diagnostics.txt'),
				delimiter=',', comments='#', dtype=None, encoding='utf-8')

			for k, star in tqdm(enumerate(self.starlist), total=len(self.starlist)):
				# Get starid:
				starname = star[0]
				starclass = star[1]
				if starname.startswith('constant_'):
					starid = -10000 - int(starname[9:])
				elif starname.startswith('fakerrlyr_'):
					starid = -20000 - int(starname[10:])
				else:
					starid = int(starname)
					starname = '{0:09d}'.format(starid)

				# Path to lightcurve:
				lightcurve = starclass + '/' + starname + '.txt'

				# Load diagnostics from file, to speed up the process:
				variance, rms_hour, ptp = diagnostics[k]

				pri += 1
				self.generate_todolist_insert(cursor,
					priority=pri,
					starid=starid,
					lightcurve=lightcurve,
					datasource='ffi',
					variance=variance,
					rms_hour=rms_hour,
					ptp=ptp)

			conn.commit()
			cursor.close()

		logger.info("%s training set successfully built.", self.key)

	#----------------------------------------------------------------------------------------------
	def generate_todolist_structure(self, conn):
		"""
		Generate overall database structure for todo.sqlite.

		Parameters:
			conn (sqlite3.connection): Connection to SQLite file.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

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
	def generate_todolist_insert(self, cursor, priority=None, lightcurve=None, starid=None,
		tmag=None, datasource=None, variance=None, rms_hour=None, ptp=None):
		"""
		Insert an entry in the todo.sqlite file.

		Parameters:
			cursor (sqlite3.Cursor): Cursor in SQLite file.
			priority (int):
			lightcurve (str):
			starid (int, optional):
			tmag (float, optional): TESS Magnitude.
			datasource (str, optional):
			variance (float, optional):
			rms_hour (float, optional):
			ptp (float, optional):

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		if priority is None:
			raise ValueError("PRIORITY is required.")
		if lightcurve is None:
			raise ValueError("LIGHTCURVE is required.")
		if starid is None:
			starid = priority

		# Try to load the lightcurve using the BaseClassifier method.
		# This will ensure that the lightcurve can actually be read by the system.
		if not all([datasource, variance, rms_hour, ptp]):
			with BaseClassifier(tset=self, features_cache=None) as bc:
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

				# Make sure overwrite=False, or else previous results will be deleted,
				# meaning there would be no results for the MetaClassifier to work with
				with TaskManager(tmpdir.name, overwrite=False, cleanup=False, classes=self.StellarClasses) as tm:
					# NOTE: This does not propergate the 'data_dir' keyword to the BaseClassifier,
					#       But since we are not doing anything other than loading data,
					#       this should not cause any problems.
					with BaseClassifier(tset=self, features_cache=self.features_cache) as stcl:
						for rowidx in self.train_idx:
							task = tm.get_task(priority=rowidx+1, change_classifier=False)

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

				# Make sure overwrite=False, or else previous results will be deleted,
				# meaning there would be no results for the MetaClassifier to work with
				with TaskManager(tmpdir.name, overwrite=False, cleanup=False, classes=self.StellarClasses) as tm:
					# NOTE: This does not propergate the 'data_dir' keyword to the BaseClassifier,
					#       But since we are not doing anything other than loading data,
					#       this should not cause any problems.
					with BaseClassifier(tset=self, features_cache=self.features_cache) as stcl:
						for rowidx in self.test_idx:
							task = tm.get_task(priority=rowidx+1, change_classifier=False)

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
	def labels(self):
		"""
		Labels of training-set.

		Returns:
			tuple: Tuple of labels associated with features in :meth:`features`.
				Each element is itself a tuple of enums of :class:`StellarClasses`.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		# Create list of all the classes for each star:
		lookup = []
		for rowidx in self.train_idx:
			row = self.starlist[rowidx, :]
			labels = row[1].strip().split(';')
			lbls = [self.StellarClasses[lbl.strip()] for lbl in labels]
			lookup.append(tuple(set(lbls)))

		return tuple(lookup)

	#----------------------------------------------------------------------------------------------
	def labels_test(self):
		"""
		Labels of test-set.

		Returns:
			tuple: Tuple of labels associated with features in :meth:`features_test`.
				Each element is itself a tuple of enums of :class:`StellarClasses`.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		# Create list of all the classes for each star:
		lookup = []
		for rowidx in self.test_idx:
			row = self.starlist[rowidx, :]
			labels = row[1].strip().split(';')
			lbls = [self.StellarClasses[lbl.strip()] for lbl in labels]
			lookup.append(tuple(set(lbls)))

		return tuple(lookup)

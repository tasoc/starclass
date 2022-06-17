#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Sets.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from bottleneck import nanvar
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
from .. import BaseClassifier, TaskManager, utilities, io
from ..StellarClasses import StellarClassesLevel1, StellarClassesLevel2
from ..todolist import todolist_structure, todolist_insert, todolist_cleanup

#--------------------------------------------------------------------------------------------------
class TrainingSet(object):
	"""
	Generic Training Set.

	Attributes:
		key (str): Unique identifier for training set.
		linfit (bool): Indicating if linfit mechanism is enabled.
		testfraction (float): Test-fraction.
		StellarClasses (enum): Enum of the classes associated with this training set.
		random_seed (int): Random seed in use.
		features_cache (str): Path to directory where cache of extracted features is being stored.
		train_idx (ndarray):
		test_idx (ndarray):
		crossval_folds (int): Number of cross-validation folds the training set has
			been split into. If ``0`` the training set has not been split.
		fold (int): The current cross-validation fold. This is ``0`` in the original training set.
	"""

	# Name of the TODO-file used by this training set:
	_todo_name = 'todo'

	def __init__(self, level='L1', datalevel='corr', tf=0.0, linfit=False, random_seed=42):
		"""
		Initialize TrainingSet.

		Parameters:
			level (str): Level of the classification. Choises are ``'L1'`` and ``'L2'``.
				Default is level 1.
			tf (float): Test-fraction. Default=0.
			linfit (bool): Should linfit be enabled for the trainingset?
				If ``linfit`` is enabled, lightcurves will be detrended using a linear
				trend before passed on to have frequencies extracted.
				See :meth:`BaseClassifier.calc_features` for details.
			random_seed (int): Random seed. Default=42.
			datalevel (str): Deprecated.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		if not hasattr(self, 'key'):
			raise RuntimeError("Training set class does not have 'key' definied.")

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
		self.linfit = linfit

		# Assign StellarClasses Enum depending on
		# the classification level we are running:
		if not hasattr(self, 'StellarClasses'):
			self.StellarClasses = {
				'L1': StellarClassesLevel1,
				'L2': StellarClassesLevel2
			}[self.level]

		# Define cache location where we will save common features:
		features_cache_name = 'features_cache_' + self.datalevel
		if self.linfit:
			self.key += '-linfit'
			self._todo_name += '-linfit'
			features_cache_name += '_linfit'
		self.features_cache = os.path.join(self.input_folder, features_cache_name)
		os.makedirs(self.features_cache, exist_ok=True)

		# Generate TODO file if it is needed:
		self.todo_file = os.path.join(self.input_folder, self._todo_name + '.sqlite')
		if not os.path.isfile(self.todo_file):
			self.generate_todolist()

		#with closing(sqlite3.connect(self.todo_file)) as conn:
		#	conn.row_factory = sqlite3.Row
		#	cursor = conn.cursor()
		#	cursor.execute("SELECT COUNT(*) FROM todolist;")
		#	self.nobjects = int(cursor.fetchone()[0])

		# Generate training/test indices
		# Define here because it is needed by self.labels() used below
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

		self.fake_metaclassifier = False

	#----------------------------------------------------------------------------------------------
	def __str__(self):
		str_fold = '' if self.fold == 0 else f', fold={self.fold:d}/{self.crossval_folds:d}'
		return f"<TrainingSet({self.key:s}, {self.datalevel:s}, tf={self.testfraction:.2f}{str_fold:s})>"

	#----------------------------------------------------------------------------------------------
	def __len__(self):
		return len(self.train_idx)

	#----------------------------------------------------------------------------------------------
	def folds(self, n_splits=5):
		"""
		Split training set object into stratified folds.

		Parameters:
			n_splits (int, optional): Number of folds to split training set into. Default=5.

		Returns:
			Iterator of :class:`TrainingSet` objects: Iterator of folds, which are also
				:class:`TrainingSet` objects.
		"""

		logger = logging.getLogger(__name__)

		# FIXME: Use BaseClassifier.parse_labels
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
			newtset = self.__class__(
				level=self.level,
				datalevel=self.datalevel,
				linfit=self.linfit,
				random_seed=self.random_seed,
				tf=0.0)

			# Transfer settings not set during initialization:
			newtset.fake_metaclassifier = self.fake_metaclassifier

			# Set testfraction to value from CV i.e. 1/n_splits
			newtset.testfraction = 1/n_splits
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
			raise RuntimeError("Training set class does not have 'key' definied.")

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
			raise RuntimeError("Training set class does not have 'key' definied.")

		logger = logging.getLogger(__name__)
		tqdm_settings = {
			'unit': 'B',
			'unit_scale': True,
			'unit_divisor': 1024,
			'disable': None if logger.isEnabledFor(logging.INFO) else True
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

		try:
			with closing(sqlite3.connect(self.todo_file)) as conn:
				conn.row_factory = sqlite3.Row
				cursor = conn.cursor()

				# Create the basic file structure of a TODO-list:
				todolist_structure(conn)

				logger.info("Step 3: Reading file and extracting information...")
				pri = 0

				diagnostics_file = os.path.join(self.input_folder, 'diagnostics.txt')
				diagnostics = None
				if os.path.isfile(diagnostics_file):
					diagnostics = np.genfromtxt(diagnostics_file,
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

					# Check that the file actually exists:
					if not os.path.exists(os.path.join(self.input_folder, lightcurve)):
						raise FileNotFoundError(lightcurve)

					# Load diagnostics from file, to speed up the process:
					if diagnostics is not None:
						variance, rms_hour, ptp = diagnostics[k]
					else:
						# Load the lightcurve using the load_lightcurve method.
						# This will ensure that the lightcurve can actually be read by the system.
						lc = io.load_lightcurve(os.path.join(self.input_folder, lightcurve))

						variance = nanvar(lc.flux, ddof=1)
						rms_hour = utilities.rms_timescale(lc)
						ptp = utilities.ptp(lc)

						#if datasource is None:
						#	if (lc.time[1] - lc.time[0])*86400 > 1000:
						#		datasource = 'ffi'
						#	else:
						#		datasource = 'tpf'

					elaptime = np.random.normal(3.14, 0.5)

					pri += 1
					todolist_insert(cursor,
						priority=pri,
						starid=starid,
						lightcurve=lightcurve,
						datasource='ffi',
						variance=variance,
						rms_hour=rms_hour,
						ptp=ptp,
						elaptime=elaptime)

				conn.commit()
				todolist_cleanup(conn, cursor)
				cursor.close()

		except: # noqa: E722, pragma: no cover
			if os.path.exists(self.todo_file):
				os.remove(self.todo_file)
			raise

		logger.info("%s training set successfully built.", self.key)

	#----------------------------------------------------------------------------------------------
	def features(self):
		"""
		Iterator of features for training.

		Returns:
			Iterator: Iterator of dicts containing features to be used for training.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		cl = 'meta' if self.fake_metaclassifier else None

		# Create a temporary copy of the TODO-file that we are going to read from.
		# This is due to errors we have detected, where the database is unexpectively locked
		# when opened several times in parallel.
		try:
			with tempfile.NamedTemporaryFile(dir=self.input_folder, suffix='.sqlite', delete=False) as tmpdir:
				# Copy the original TODO-file to the new temp file:
				with open(self.todo_file, 'rb') as fid:
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
							task = tm.get_task(priority=rowidx+1, classifier=cl, change_classifier=False)

							# Lightcurve file to load:
							# We do not use the one from the database because in the simulations the
							# raw and corrected light curves are stored in different files.
							yield stcl.load_star(task)

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

		cl = 'meta' if self.fake_metaclassifier else None

		# Create a temporary copy of the TODO-file that we are going to read from.
		# This is due to errors we have detected, where the database is unexpectively locked
		# when opened several times in parallel.
		try:
			with tempfile.NamedTemporaryFile(dir=self.input_folder, suffix='.sqlite', delete=False) as tmpdir:
				# Copy the original TODO-file to the new temp file:
				with open(self.todo_file, 'rb') as fid:
					shutil.copyfileobj(fid, tmpdir)
				tmpdir.flush()

				# Make sure overwrite=False, or else previous results will be deleted,
				# meaning there would be no results for the MetaClassifier to work with
				with TaskManager(tmpdir.name, overwrite=False, cleanup=False, classes=self.StellarClasses) as tm:
					for rowidx in self.test_idx:
						task = tm.get_task(priority=rowidx+1, classifier=cl, change_classifier=False)

						# Lightcurve file to load:
						# We do not use the one from the database because in the simulations the
						# raw and corrected light curves are stored in different files.
						yield task

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

	#----------------------------------------------------------------------------------------------
	def clear_cache(self):
		"""
		Clear features cache.

		This will delete the features cache directory in the training-set data directory,
		and delete all MOAT cache tables in the training-set.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		# Delete the features cache directory:
		if os.path.exists(self.features_cache):
			shutil.rmtree(self.features_cache)

		# Delete the MOAT tables from the training-set todo-file:
		with TaskManager(self.todo_file, overwrite=False, classes=self.StellarClasses) as tm:
			tm.moat_clear()

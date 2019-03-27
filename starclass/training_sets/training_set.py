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
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from .. import BaseClassifier, TaskManager

#----------------------------------------------------------------------------------------------
class TrainingSet(object):

	def __init__(self, classifier=None, datalevel='corr', tf=0.0, KFoldCV=False):

		self.classifier = classifier
		self.datalevel = datalevel
		self.testfraction = tf
		self.KFoldCV = KFoldCV

		if self.testfraction < 0 or self.testfraction > 1:
			raise ValueError("Invalid testfraction provided")

		self.features_cache = os.path.join(self.input_folder, 'features_cache_%s' % self.datalevel)

		if not os.path.exists(self.features_cache):
			os.makedirs(self.features_cache)

		# Generate TODO file if it is needed:
		sqlite_file = os.path.join(self.input_folder, 'todo.sqlite')
		if not os.path.exists(sqlite_file):
			self.generate_todolist()

		# Generate training/test indices
		if self.testfraction > 0:
			fitlabels = self.parse_labels(self.labels(train_test_split=False))
			self.train_idx, self.test_idx = train_test_split(
				np.arange(self.nobjects),
				test_size=self.testfraction,
				random_state=42,
				stratify=fitlabels,
			)

		# Cross Validation
		if self.KFoldCV == True:
			# If keyword is true then split according to KFold cross-vadliation
			skf = StratifiedKFold(n_splits=5, random_state=42)
			if self.testfraction > 0:
				# Have to pass labels back into original format so sklearn can
				# properly interpret them to provide the indices
				fitlabels = self.parse_labels(self.labels(train_test_split=True))
				self.skf = skf.split(self.train_idx, fitlabels)
			else:
				fitlabels = self.parse_labels(self.labels(train_test_split=False))
				self.skf = skf.split(np.arange(self.nobjects),
									 fitlabels)

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

		rowidx = -1
		with TaskManager(self.input_folder, overwrite=True) as tm:
			with BaseClassifier(features_cache=self.features_cache) as stcl:
				while True:
					task = tm.get_task(classifier=self.classifier, change_classifier=False)
					if task is None: break
					tm.start_task(task)
					rowidx += 1

					# Lightcurve file to load:
					# We do not use the one from the database because in the simulations the
					# raw and corrected light curves are stored in different files.
					fname = os.path.join(self.input_folder, task['lightcurve'])

					if (self.testfraction > 0) and (self.KFoldCV == False):
						if rowidx in self.train_idx:
							yield stcl.load_star(task, fname)
					elif (self.KFoldCV == True) and (self.testfraction > 0):
						if rowidx in self.cvtrain_idx:
							yield stcl.load_star(task, fname)
					elif (self.KFoldCV == True):
						if rowidx in self.cvtrain_idx:
							yield stcl.load_star(task, fname)
					else:
						yield stcl.load_star(task, fname)

	#----------------------------------------------------------------------------------------------
	def features_test(self):

		# Ignore this error if KFoldCV flag set since we set test_idx manually
		if (self.testfraction <= 0) and (self.KFoldCV == False):
			raise ValueError('features_test requires testfraction>0')
		else:
			rowidx = -1
			with TaskManager(self.input_folder, overwrite=True) as tm:
				with BaseClassifier(features_cache=self.features_cache) as stcl:
					while True:
						task = tm.get_task(classifier=self.classifier, change_classifier=False)
						if task is None: break
						tm.start_task(task)
						rowidx += 1

						if (self.KFoldCV == True):
							if rowidx in self.cvtest_idx:
								# Lightcurve file to load:
								# We do not use the one from the database because in the simulations the
								# raw and corrected light curves are stored in different files.
								fname = os.path.join(self.input_folder, task['lightcurve'])
								yield stcl.load_star(task, fname)
						else:
							if rowidx in self.test_idx:
								# Lightcurve file to load:
								# We do not use the one from the database because in the simulations the
								# raw and corrected light curves are stored in different files.
								fname = os.path.join(self.input_folder, task['lightcurve'])
								yield stcl.load_star(task, fname)


	#----------------------------------------------------------------------------------------------
	def labels(self, level='L1'):
		raise NotImplementedError()

	#----------------------------------------------------------------------------------------------
	def labels_test(self, level='L1'):
		raise NotImplementedError()

	def parse_labels(self,labels,removeduplicates=False):
		"""
		"""
		fitlabels = []
		for lbl in labels:
			if removeduplicates:
				#is it multi-labelled? In which case, what takes priority?
				#or duplicate it once for each label
				if len(lbl)>1:#Priority order loosely based on signal clarity
					if StellarClasses.ECLIPSE in lbl:
						fitlabels.append('transit/eclipse')
					elif StellarClasses.RRLYR_CEPHEID in lbl:
						fitlabels.append('RRLyr/Ceph')
					elif StellarClasses.CONTACT_ROT in lbl:
						fitlabels.append('contactEB/spots')
					elif StellarClasses.DSCT_BCEP in lbl:
						fitlabels.append('dSct/bCep')
					elif StellarClasses.GDOR_SPB in lbl:
						fitlabels.append('gDor/spB')
					elif StellarClasses.SOLARLIKE in lbl:
						fitlabels.append('solar')
					else:
						fitlabels.append(lbl[0].value)
				else:
					#then convert to str
					fitlabels.append(lbl[0].value)
			else:
				fitlabels.append(lbl[0].value)
		return np.array(fitlabels)

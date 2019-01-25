#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from numpy import ceil
import os
import requests
import zipfile
import shutil
import logging
from tqdm import tqdm

#----------------------------------------------------------------------------------------------
class TrainingSet(object):

	def __init__(self, classifier=None, datalevel='corr', tf=0.0):

		self.classifier = classifier
		self.datalevel = datalevel
		self.testfraction = tf

		self.features_cache = os.path.join(self.input_folder, 'features_cache_%s' % self.datalevel)

		if not os.path.exists(self.features_cache):
			os.makedirs(self.features_cache)

		sqlite_file = os.path.join(self.input_folder, 'todo.sqlite')
		if not os.path.exists(sqlite_file):
			self.generate_todolist()


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

	def generate_todolist(self):
		raise NotImplementedError()

	def training_set_features(self):
		raise NotImplementedError()
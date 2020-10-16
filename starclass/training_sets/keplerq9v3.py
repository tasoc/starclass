#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kepler Q9 Training Set (version 3).

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
import numpy as np
from contextlib import closing
import sqlite3
import logging
from tqdm import tqdm
from . import TrainingSet

#--------------------------------------------------------------------------------------------------
class keplerq9v3(TrainingSet):
	"""
	Kepler Q9 Training Set (version 3).

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	# Class constants:
	key = 'keplerq9v3'

	def __init__(self, *args, datalevel='corr', **kwargs):

		if datalevel != 'corr':
			raise ValueError("The KeplerQ9v3 training set only as corrected data. Please specify datalevel='corr'.")

		# Point this to the directory where the TDA simulations are stored
		self.input_folder = self.tset_datadir('https://tasoc.dk/pipeline/starclass_trainingsets/keplerq9v3.zip')

		# Find the number of training sets:
		self.starlist = np.genfromtxt(os.path.join(self.input_folder, 'targets.txt'),
			dtype='str',
			delimiter=',',
			comments='#',
			encoding='utf-8')

		# Remove the instrumental class from this trainingset:
		indx = [star[1] != 'INSTRUMENT' for star in self.starlist]
		self.starlist = self.starlist[indx]

		# Count the number of objects in trainingset:
		self.nobjects = self.starlist.shape[0]

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

	#----------------------------------------------------------------------------------------------
	def generate_todolist(self):

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
				if starclass == 'INSTRUMENT':
					continue
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

#--------------------------------------------------------------------------------------------------
class keplerq9v3_instr(TrainingSet):
	"""
	Kepler Q9 Training Set (version 3).

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	# Class constants:
	key = 'keplerq9v3-instr'
	datadir = 'keplerq9v3'

	def __init__(self, *args, datalevel='corr', **kwargs):

		if datalevel != 'corr':
			raise ValueError("The KeplerQ9v3 training set only as corrected data. Please specify datalevel='corr'.")

		# Point this to the directory where the TDA simulations are stored
		self.input_folder = self.tset_datadir('https://tasoc.dk/pipeline/starclass_trainingsets/keplerq9v3.zip')

		# Find the number of training sets:
		self.starlist = np.genfromtxt(os.path.join(self.input_folder, 'targets.txt'),
			dtype='str',
			delimiter=',',
			comments='#',
			encoding='utf-8')

		# Count the number of objects in trainingset:
		self.nobjects = self.starlist.shape[0]

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

	#----------------------------------------------------------------------------------------------
	def generate_todolist(self):

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

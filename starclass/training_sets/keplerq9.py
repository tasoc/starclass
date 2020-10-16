#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kepler Q9 Training Set.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
import numpy as np
from contextlib import closing
import sqlite3
import logging
from tqdm import tqdm
from . import TrainingSet

#----------------------------------------------------------------------------------------------
class keplerq9(TrainingSet):
	"""
	Kepler Q9 Training Set.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	# Class constants:
	key = 'keplerq9'

	def __init__(self, *args, datalevel='corr', **kwargs):

		if datalevel != 'corr':
			raise ValueError("The KeplerQ9 training set only as corrected data. Please specify datalevel='corr'.")

		# Point this to the directory where the TDA simulations are stored
		self.input_folder = self.tset_datadir('https://tasoc.dk/pipeline/starclass_trainingsets/keplerq9.zip')

		# Find the number of training sets:
		self.starlist = np.genfromtxt(os.path.join(self.input_folder, 'targets.txt'),
			dtype='str',
			delimiter=',',
			comments='#',
			encoding='utf-8')
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
				delimiter=',', dtype=None, encoding='utf-8')

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

				# Path to lightcurve:
				lightcurve = starclass + '/' + starname + '.txt'

				# Load diagnostics from file, to speed up the process:
				variance, rms_hour, ptp = diagnostics[k]

				#data = np.loadtxt(os.path.join(self.input_folder, lightcurve))
				#if data[-1,0] - data[0,0] > 27.4:
				#	raise Exception("Okay, didn't we agree that this should be only one sector?!")

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

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
from .. import StellarClasses
from . import TrainingSet

#----------------------------------------------------------------------------------------------
class keplerq9(TrainingSet):
	"""
	Kepler Q9 Training Set.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, *args, datalevel='corr', **kwargs):

		if datalevel != 'corr':
			raise ValueError("The KeplerQ9 training set only as corrected data. Please specify datalevel='corr'.")

		# Key for this training-set:
		self.key = 'keplerq9'

		# Point this to the directory where the TDA simulations are stored
		self.input_folder = self.tset_datadir('keplerq9', 'https://tasoc.dk/pipeline/starclass_trainingsets/keplerq9.zip')

		# Find the number of training sets:
		data = np.genfromtxt(os.path.join(self.input_folder, 'targets.txt'), dtype=None, delimiter=',', encoding='utf-8')
		self.nobjects = data.shape[0]

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

			logger.info("Step 1: Reading file and extracting information...")
			pri = 0
			starlist = np.genfromtxt(os.path.join(self.input_folder, 'targets.txt'), delimiter=',', dtype=None, encoding='utf-8')
			diagnostics = np.genfromtxt(os.path.join(self.input_folder, 'diagnostics.txt'), delimiter=',', dtype=None, encoding='utf-8')
			for k, star in tqdm(enumerate(starlist), total=len(starlist)):
				# Get starid:
				starname = star[0]
				starclass = star[1]
				if starname.startswith('constant_'):
					starid = -1
				elif starname.startswith('fakerrlyr_'):
					starid = -1
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

		logger.info("DONE.")

	#----------------------------------------------------------------------------------------------
	def labels(self, level='L1'):

		logger = logging.getLogger(__name__)

		data = np.genfromtxt(os.path.join(self.input_folder, 'targets.txt'), dtype=None, delimiter=',', encoding='utf-8')

		# Translation of Mikkel's identifiers into the broader:
		translate = {
			'SOLARLIKE': StellarClasses.SOLARLIKE,
			'ECLIPSE': StellarClasses.ECLIPSE,
			'RRLYR_CEPHEID': StellarClasses.RRLYR_CEPHEID,
			'GDOR_SPB': StellarClasses.GDOR_SPB,
			'DSCT_BCEP': StellarClasses.DSCT_BCEP,
			'CONTACT_ROT': StellarClasses.CONTACT_ROT,
			'APERIODIC': StellarClasses.APERIODIC,
			'CONSTANT': StellarClasses.CONSTANT
		}

		# Create list of all the classes for each star:
		lookup = []
		for rowidx,row in enumerate(data):
			#starid = int(row[0][4:])
			labels = row[1].strip().split(';')
			lbls = []
			for lbl in labels:
				lbl = lbl.strip()
				c = translate.get(lbl.strip())
				if c is None:
					logger.error("Unknown label: %s", lbl)
				else:
					lbls.append(c)

			if self.testfraction > 0:
				if rowidx in self.train_idx:
					lookup.append(tuple(set(lbls)))
			else:
				lookup.append(tuple(set(lbls)))

		return tuple(lookup)

	#----------------------------------------------------------------------------------------------
	def labels_test(self, level='L1'):

		logger = logging.getLogger(__name__)

		if self.testfraction <= 0:
			return []
		else:
			data = np.genfromtxt(os.path.join(self.input_folder, 'targets.txt'), dtype=None, delimiter=',', encoding='utf-8')

			# Translation of Mikkel's identifiers into the broader:
			translate = {
				'SOLARLIKE': StellarClasses.SOLARLIKE,
				'ECLIPSE': StellarClasses.ECLIPSE,
				'RRLYR_CEPHEID': StellarClasses.RRLYR_CEPHEID,
				'GDOR_SPB': StellarClasses.GDOR_SPB,
				'DSCT_BCEP': StellarClasses.DSCT_BCEP,
				'CONTACT_ROT': StellarClasses.CONTACT_ROT,
				'APERIODIC': StellarClasses.APERIODIC,
				'CONSTANT': StellarClasses.CONSTANT
			}

			# Create list of all the classes for each star:
			lookup = []
			for rowidx, row in enumerate(data):
				#starid = int(row[0][4:])
				labels = row[1].strip().split(';')
				lbls = []
				for lbl in labels:
					lbl = lbl.strip()
					c = translate.get(lbl.strip())
					if c is None:
						logger.error("Unknown label: %s", lbl)
					else:
						lbls.append(c)

				if rowidx in self.test_idx:
					lookup.append(tuple(set(lbls)))
			return tuple(lookup)

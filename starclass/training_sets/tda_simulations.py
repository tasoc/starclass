#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
import numpy as np
import sqlite3
from contextlib import closing
import logging
from tqdm import tqdm
from .. import StellarClasses
from . import TrainingSet

#----------------------------------------------------------------------------------------------
class tda_simulations(TrainingSet):

	def __init__(self, *args, **kwargs):

		# Key for this training-set:
		datalevel = kwargs.get('datalevel')
		if datalevel == 'raw':
			self.key = 'tdasim-raw'
			self.input_folder = self.tset_datadir('tdasim-raw', 'https://tasoc.dk/pipeline/starclass_trainingsets/tdasim-raw.zip')
		elif datalevel == 'corr':
			self.key = 'tdasim'
			self.input_folder = self.tset_datadir('tdasim', 'https://tasoc.dk/pipeline/starclass_trainingsets/tdasim.zip')
		elif datalevel == 'clean':
			self.key = 'tdasim-clean'
			self.input_folder = self.tset_datadir('tdasim-clean', 'https://tasoc.dk/pipeline/starclass_trainingsets/tdasim-clean.zip')

		data = np.genfromtxt(os.path.join(self.input_folder, 'Data_Batch_TDA4_r1.txt'), dtype=None, delimiter=',', usecols=(0,10), encoding='utf-8')
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
			starlist = np.genfromtxt(os.path.join(self.input_folder, 'Data_Batch_TDA4_r1.txt'), delimiter=',', dtype=None, encoding='utf-8')
			#diagnostics = np.genfromtxt(os.path.join(self.input_folder, 'diagnostics.txt'), delimiter=',', dtype=None, encoding='utf-8')
			for k, star in tqdm(enumerate(starlist), total=len(starlist)):
				# Get starid:
				starname = star[0]
				starid = int(starname[4:])

				# Path to lightcurve:
				if self.datalevel == 'raw':
					lightcurve = 'sysnoise/Star%d.sysnoise' % starid
				elif self.datalevel == 'corr':
					lightcurve = 'noisy/Star%d.noisy' % starid
				elif self.datalevel == 'clean':
					lightcurve = 'clean/Star%d.clean' % starid

				# Extract the camera from the lattitude:
				tmag = star[1]
				#ecllat = star[4]
				#ecllon = star[5]

				# Load diagnostics from file, to speed up the process:
				#variance, rms_hour, ptp = diagnostics[k]

				#data = np.loadtxt(os.path.join(self.input_folder, lightcurve))
				#sector = np.floor(data_sysnoise[:,0] / 27.4) + 1
				#sectors = [int(s) for s in np.unique(sector)]
				#if data[-1,0] - data[0,0] > 27.4:
				#	raise Exception("Okay, didn't we agree that this should be only one sector?!")

				pri += 1
				self.generate_todolist_insert(cursor,
					priority=pri,
					starid=starid,
					lightcurve=lightcurve,
					tmag=tmag)
				#datasource='ffi',
				#variance=variance,
				#rms_hour=rms_hour,
				#ptp=ptp

			conn.commit()
			cursor.close()

		logger.info("DONE.")

	#----------------------------------------------------------------------------------------------
	def labels(self, level='L1'):

		logger = logging.getLogger(__name__)

		data = np.genfromtxt(os.path.join(self.input_folder, 'Data_Batch_TDA4_r1.txt'), dtype=None, delimiter=', ', usecols=(0,10), encoding='utf-8')

		# Translation of Mikkel's identifiers into the broader
		# classes we have defined in StellarClasses:
		if level == 'L1':
			translate = {
				'Solar-like': StellarClasses.SOLARLIKE,
				'Transit': StellarClasses.ECLIPSE,
				'Eclipse': StellarClasses.ECLIPSE, # short period EBs should be CONTACT_ROT, not ECLIPSE
				'multi': StellarClasses.ECLIPSE,
				'MMR': StellarClasses.ECLIPSE,
				'RR Lyrae': StellarClasses.RRLYR_CEPHEID,
				'RRab': StellarClasses.RRLYR_CEPHEID,
				'RRc': StellarClasses.RRLYR_CEPHEID,
				'RRd': StellarClasses.RRLYR_CEPHEID,
				'Cepheid': StellarClasses.RRLYR_CEPHEID,
				'FM': StellarClasses.RRLYR_CEPHEID,
				'1O': StellarClasses.RRLYR_CEPHEID,
				'1O2O': StellarClasses.RRLYR_CEPHEID,
				'FM1O': StellarClasses.RRLYR_CEPHEID,
				'Type II': StellarClasses.RRLYR_CEPHEID,
				'Anomaleous': StellarClasses.RRLYR_CEPHEID,
				'SPB': StellarClasses.GDOR_SPB,
				'dsct': StellarClasses.DSCT_BCEP,
				'bumpy': StellarClasses.GDOR_SPB,
				'gDor': StellarClasses.GDOR_SPB,
				'bCep': StellarClasses.DSCT_BCEP,
				'roAp': StellarClasses.RAPID,
				'sdBV': StellarClasses.RAPID,
				'Flare': StellarClasses.TRANSIENT,
				'Spots': StellarClasses.CONTACT_ROT,
				'LPV': StellarClasses.APERIODIC,
				'MIRA': StellarClasses.APERIODIC,
				'SR': StellarClasses.APERIODIC,
				'Constant': StellarClasses.CONSTANT
			}
		elif level == 'L2':
			translate = {
				'Solar-like': StellarClasses.SOLARLIKE,
				'Transit': StellarClasses.ECLIPSE,
				'Eclipse': StellarClasses.ECLIPSE,
				'multi': StellarClasses.ECLIPSE,
				'MMR': StellarClasses.ECLIPSE,
				'RR Lyrae': StellarClasses.RRLYR,
				'RRab': StellarClasses.RRLYR,
				'RRc': StellarClasses.RRLYR,
				'RRd': StellarClasses.RRLYR,
				'Cepheid': StellarClasses.CEPHEID,
				'FM': StellarClasses.CEPHEID,
				'1O': StellarClasses.CEPHEID,
				'1O2O': StellarClasses.CEPHEID,
				'FM1O': StellarClasses.CEPHEID,
				'Type II': StellarClasses.CEPHEID,
				'Anomaleous': StellarClasses.CEPHEID,
				'SPB': StellarClasses.SPB,
				'dsct': StellarClasses.DSCT,
				'bumpy': StellarClasses.DSCT, # This is not right - Should we make a specific class for these?
				'gDor': StellarClasses.GDOR,
				'bCep': StellarClasses.BCEP,
				'roAp': StellarClasses.ROAP,
				'sdBV': StellarClasses.SDB,
				'Flare': StellarClasses.TRANSIENT,
				'Spots': StellarClasses.SPOTS,
				'LPV': StellarClasses.LPV,
				'MIRA': StellarClasses.LPV,
				'SR': StellarClasses.LPV,
				'Constant': StellarClasses.CONSTANT
			}

		# Create list of all the classes for each star:
		lookup = []
		for rowidx,row in enumerate(data):
			#starid = int(row[0][4:])
			labels = row[1].strip().split(';')
			lbls = []
			for lbl in labels:
				lbl = lbl.strip()
				if lbl == 'gDor+dSct hybrid' or lbl == 'dSct+gDor hybrid':
					if level == 'L1':
						lbls.append(StellarClasses.DSCT_BCEP)
						lbls.append(StellarClasses.GDOR_SPB)
					elif level == 'L2':
						lbls.append(StellarClasses.DSCT)
						lbls.append(StellarClasses.GDOR)
				elif lbl == 'bCep+SPB hybrid':
					if level == 'L1':
						lbls.append(StellarClasses.DSCT_BCEP)
						lbls.append(StellarClasses.GDOR_SPB)
					elif level == 'L2':
						lbls.append(StellarClasses.BCEP)
						lbls.append(StellarClasses.SPB)
				else:
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

		if self.testfraction == 0:
			return []
		else:
			data = np.genfromtxt(os.path.join(self.input_folder, 'Data_Batch_TDA4_r1.txt'), dtype=None, delimiter=', ', usecols=(0,10), encoding='utf-8')

			# Translation of Mikkel's identifiers into the broader
			# classes we have defined in StellarClasses:
			if level == 'L1':
				translate = {
					'Solar-like': StellarClasses.SOLARLIKE,
					'Transit': StellarClasses.ECLIPSE,
					'Eclipse': StellarClasses.ECLIPSE, # short period EBs should be CONTACT_ROT, not ECLIPSE
					'multi': StellarClasses.ECLIPSE,
					'MMR': StellarClasses.ECLIPSE,
					'RR Lyrae': StellarClasses.RRLYR_CEPHEID,
					'RRab': StellarClasses.RRLYR_CEPHEID,
					'RRc': StellarClasses.RRLYR_CEPHEID,
					'RRd': StellarClasses.RRLYR_CEPHEID,
					'Cepheid': StellarClasses.RRLYR_CEPHEID,
					'FM': StellarClasses.RRLYR_CEPHEID,
					'1O': StellarClasses.RRLYR_CEPHEID,
					'1O2O': StellarClasses.RRLYR_CEPHEID,
					'FM1O': StellarClasses.RRLYR_CEPHEID,
					'Type II': StellarClasses.RRLYR_CEPHEID,
					'Anomaleous': StellarClasses.RRLYR_CEPHEID,
					'SPB': StellarClasses.GDOR_SPB,
					'dsct': StellarClasses.DSCT_BCEP,
					'bumpy': StellarClasses.GDOR_SPB,
					'gDor': StellarClasses.GDOR_SPB,
					'bCep': StellarClasses.DSCT_BCEP,
					'roAp': StellarClasses.RAPID,
					'sdBV': StellarClasses.RAPID,
					'Flare': StellarClasses.TRANSIENT,
					'Spots': StellarClasses.CONTACT_ROT,
					'LPV': StellarClasses.APERIODIC,
					'MIRA': StellarClasses.APERIODIC,
					'SR': StellarClasses.APERIODIC,
					'Constant': StellarClasses.CONSTANT
				}
			elif level == 'L2':
				translate = {
					'Solar-like': StellarClasses.SOLARLIKE,
					'Transit': StellarClasses.ECLIPSE,
					'Eclipse': StellarClasses.ECLIPSE,
					'multi': StellarClasses.ECLIPSE,
					'MMR': StellarClasses.ECLIPSE,
					'RR Lyrae': StellarClasses.RRLYR,
					'RRab': StellarClasses.RRLYR,
					'RRc': StellarClasses.RRLYR,
					'RRd': StellarClasses.RRLYR,
					'Cepheid': StellarClasses.CEPHEID,
					'FM': StellarClasses.CEPHEID,
					'1O': StellarClasses.CEPHEID,
					'1O2O': StellarClasses.CEPHEID,
					'FM1O': StellarClasses.CEPHEID,
					'Type II': StellarClasses.CEPHEID,
					'Anomaleous': StellarClasses.CEPHEID,
					'SPB': StellarClasses.SPB,
					'dsct': StellarClasses.DSCT,
					'bumpy': StellarClasses.DSCT, # This is not right - Should we make a specific class for these?
					'gDor': StellarClasses.GDOR,
					'bCep': StellarClasses.BCEP,
					'roAp': StellarClasses.ROAP,
					'sdBV': StellarClasses.SDB,
					'Flare': StellarClasses.TRANSIENT,
					'Spots': StellarClasses.SPOTS,
					'LPV': StellarClasses.LPV,
					'MIRA': StellarClasses.LPV,
					'SR': StellarClasses.LPV,
					'Constant': StellarClasses.CONSTANT
				}

			# Create list of all the classes for each star:
			lookup = []
			for rowidx,row in enumerate(data):
				#starid = int(row[0][4:])
				labels = row[1].strip().split(';')
				lbls = []
				for lbl in labels:
					lbl = lbl.strip()
					if lbl == 'gDor+dSct hybrid' or lbl == 'dSct+gDor hybrid':
						if level == 'L1':
							lbls.append(StellarClasses.DSCT_BCEP)
							lbls.append(StellarClasses.GDOR_SPB)
						elif level == 'L2':
							lbls.append(StellarClasses.DSCT)
							lbls.append(StellarClasses.GDOR)
					elif lbl == 'bCep+SPB hybrid':
						if level == 'L1':
							lbls.append(StellarClasses.DSCT_BCEP)
							lbls.append(StellarClasses.GDOR_SPB)
						elif level == 'L2':
							lbls.append(StellarClasses.BCEP)
							lbls.append(StellarClasses.SPB)
					else:
						c = translate.get(lbl.strip())
						if c is None:
							logger.error("Unknown label: %s", lbl)
						else:
							lbls.append(c)

				if rowidx in self.train_idx:
					lookup.append(tuple(set(lbls)))

		return tuple(lookup)

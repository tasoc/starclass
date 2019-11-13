#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
import numpy as np
from bottleneck import nanmedian, nansum
import sqlite3
import logging
from tqdm import tqdm
from .. import StellarClasses
from . import TrainingSet

#----------------------------------------------------------------------------------------------
class keplerq9(TrainingSet):

	def __init__(self, *args, datalevel='corr', **kwargs):

		if datalevel != 'corr':
			raise Exception("The KeplerQ9 training set only as corrected data. Please specify datalevel='corr'.")

		# Key for this training-set:
		self.key = 'keplerq9'

		# Point this to the directory where the TDA simulations are stored
		self.input_folder = self.tset_datadir('keplerq9', 'https://tasoc.dk/starclass_tsets/keplerq9.zip')

		# Find the number of training sets:
		data = np.genfromtxt(os.path.join(self.input_folder, 'targets.txt'), dtype=None, delimiter=',', encoding='utf-8')
		self.nobjects = data.shape[0]

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super(self.__class__, self).__init__(*args, **kwargs)

	#----------------------------------------------------------------------------------------------
	def generate_todolist(self):

		logger = logging.getLogger(__name__)

		sqlite_file = os.path.join(self.input_folder, 'todo.sqlite')
		with sqlite3.connect(sqlite_file) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()

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

			cursor.execute("""CREATE TABLE diagnostics_corr (
				priority INTEGER PRIMARY KEY NOT NULL,
				lightcurve TEXT,
				elaptime REAL,
				worker_wait_time REAL,
				variance REAL,
				rms_hour REAL,
				ptp REAL,
				errors TEXT
			);""")

			# Create the same indicies as is available in the real todolists:
			cursor.execute("CREATE UNIQUE INDEX priority_idx ON todolist (priority);")
			cursor.execute("CREATE INDEX status_idx ON todolist (status);")
			cursor.execute("CREATE INDEX corr_status_idx ON todolist (corr_status);")
			cursor.execute("CREATE INDEX starid_idx ON todolist (starid);")
			conn.commit()

			logger.info("Step 1: Reading file and extracting information...")
			pri = 0
			starlist = np.genfromtxt(os.path.join(self.input_folder, 'targets.txt'), delimiter=',', dtype=None, encoding='utf-8')
			for k, star in tqdm(enumerate(starlist), total=len(starlist)):
				#print(star)

				# Get starid:
				starname = star[0]
				starclass = star[1]
				if starname.startswith('constant_'):
					starid = -1
				elif starname.startswith('fakerrlyr_'):
					starid = -1
				else:
					starid = int(starname)


				data = np.loadtxt(os.path.join(self.input_folder, starclass, '%s.txt' % starname))

				if (data[1,0] - data[0,0])*86400 > 1000:
					datasource = 'ffi'
				else:
					datasource = 'tpf'

				# Extract the camera from the lattitude:
				tmag = -99
				ecllat = 0
				if ecllat < 6+24:
					camera = 1
				elif ecllat < 6+2*24:
					camera = 2
				elif ecllat < 6+3*24:
					camera = 3
				else:
					camera = 4

				#sector = np.floor(data_sysnoise[:,0] / 27.4) + 1
				#sectors = [int(s) for s in np.unique(sector)]
				if data[-1,0] - data[0,0] > 27.4:
					raise Exception("Okay, didn't we agree that this should be only one sector?!")

				#indx = (sector == s)
				#data_sysnoise_sector = data_sysnoise[indx, :]
				#data_noisy_sector = data_noisy[indx, :]
				#data_clean_sector = data_clean[indx, :]

				lightcurve = starclass + '/' + starname + '.txt'

				#lightcurve = 'Star%d-sector%02d' % (starid, s)

				# Save files cut up into sectors:
				#np.savetxt(os.path.join(self.input_folder, 'sysnoise_by_sectors', lightcurve + '.sysnoise'), data_sysnoise_sector, fmt=('%.8f', '%.18e', '%.18e', '%d'), delimiter='  ')
				#np.savetxt(os.path.join(self.input_folder, 'noisy_by_sectors', lightcurve + '.noisy'), data_noisy_sector, fmt=('%.8f', '%.18e', '%.18e', '%d'), delimiter='  ')
				#np.savetxt(os.path.join(self.input_folder, 'clean_by_sectors', lightcurve + '.clean'), data_clean_sector, fmt=('%.9f', '%.18e', '%.18e', '%d'), delimiter='  ')

				#sqlite_file = os.path.join(self.input_folder, 'todo-sector%02d.sqlite' % s)

				pri += 1
				elaptime = np.random.normal(3.14, 0.5)
				mean_flux = nanmedian(data[:,1])
				variance = nansum((data[:,1] - mean_flux)**2) / (data.shape[0] - 1)
				rms_hour = None
				ptp = None

				cursor.execute("INSERT INTO todolist (priority,starid,tmag,datasource,status,corr_status,camera,ccd,cbv_area) VALUES (?,?,?,?,1,1,?,0,0);", (
					pri,
					starid,
					tmag,
					datasource,
					camera
				))
				cursor.execute("INSERT INTO diagnostics_corr (priority,lightcurve,elaptime,variance,rms_hour,ptp) VALUES (?,?,?,?,?,?);", (
					pri,
					lightcurve,
					elaptime,
					variance,
					rms_hour,
					ptp
				))

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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import six
import os.path
import numpy as np
from bottleneck import nanmedian, nansum
import sqlite3
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from .. import StellarClasses, BaseClassifier, TaskManager
from . import TrainingSet

#----------------------------------------------------------------------------------------------
class keplerq9(TrainingSet):

	def __init__(self, *args, **kwargs):

		if kwargs.get('datalevel') != 'corr':
			raise Exception("The KeplerQ9 training set only as corrected data. Please specify datalevel='corr'.")

		# Point this to the directory where the TDA simulations are stored
		self.input_folder = self.tset_datadir('keplerq9', 'https://tasoc.dk/starclass_tsets/keplerq9.zip')

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super(self.__class__, self).__init__(*args, **kwargs)

		# Generate training/test indices
		if self.testfraction > 0:
			data = np.genfromtxt(os.path.join(self.input_folder, 'targets.txt'), dtype=None, delimiter=',', encoding='utf-8')
			nobjects = data.shape[0]
			self.train_idx, self.test_idx = train_test_split(np.arange(nobjects), test_size=self.testfraction, random_state=42, stratify=data[:,1])

	#----------------------------------------------------------------------------------------------
	def generate_todolist(self):

		logger = logging.getLogger(__name__)

		# Make sure some directories exist:
		#os.makedirs(os.path.join(self.input_folder, 'sysnoise_by_sectors'), exist_ok=True)
		#os.makedirs(os.path.join(self.input_folder, 'noisy_by_sectors'), exist_ok=True)
		#os.makedirs(os.path.join(self.input_folder, 'clean_by_sectors'), exist_ok=True)

		sqlite_file = os.path.join(self.input_folder, 'todo.sqlite')
		with sqlite3.connect(sqlite_file) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()

			cursor.execute("""CREATE TABLE todolist (
				priority BIGINT PRIMARY KEY NOT NULL,
				starid BIGINT NOT NULL,
				datasource TEXT NOT NULL DEFAULT 'ffi',
				camera INT NOT NULL,
				ccd INT NOT NULL,
				method TEXT DEFAULT NULL,
				tmag REAL,
				status INT DEFAULT NULL,
				cbv_area INT NOT NULL
			);""")

			cursor.execute("""CREATE TABLE diagnostics (
				priority BIGINT PRIMARY KEY NOT NULL,
				starid BIGINT NOT NULL,
				lightcurve TEXT,
				elaptime REAL NOT NULL,
				mean_flux DOUBLE PRECISION,
				variance DOUBLE PRECISION,
				variability DOUBLE PRECISION,
				mask_size INT,
				pos_row REAL,
				pos_column REAL,
				contamination REAL,
				stamp_resizes INT,
				errors TEXT,
				eclon DOUBLE PRECISION,
				eclat DOUBLE PRECISION
			);""")

			# Create the same indicies as is available in the real todolists:
			cursor.execute("CREATE UNIQUE INDEX priority_idx ON todolist (priority);")
			cursor.execute("CREATE INDEX starid_datasource_idx ON todolist (starid, datasource);") # FIXME: Should be "UNIQUE", but something is weird in ETE-6?!
			cursor.execute("CREATE INDEX status_idx ON todolist (status);")
			cursor.execute("CREATE INDEX starid_idx ON todolist (starid);")
			cursor.execute("CREATE INDEX variability_idx ON diagnostics (variability);")
			conn.commit()

			logger.info("Step 1: Reading file and extracting information...")
			pri = 0
			starlist = np.genfromtxt(os.path.join(self.input_folder, 'targets.txt'), delimiter=',', dtype=None, encoding='utf-8')
			for k, star in tqdm(enumerate(starlist), total=len(starlist)):
				#print(star)

				# Get starid:
				starname = star[0]
				starclass = star[1]
				if not isinstance(starname, six.string_types): starname = starname.decode("utf-8") # For Python 3
				if not isinstance(starclass, six.string_types): starclass = starclass.decode("utf-8") # For Python 3
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
				ecllon = 0
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
				mean_flux = nanmedian(data[:,1])
				variance = nansum((data[:,1] - mean_flux)**2) / (data.shape[0] - 1)

				# This could be done in the photometry code as well:
				time = data[:,0]
				flux = data[:,1] #/ mean_flux
				indx = np.isfinite(flux)
				p = np.polyfit(time[indx], flux[indx], 3)
				variability = np.nanstd(flux - np.polyval(p, time))

				elaptime = np.random.normal(3.14, 0.5)
				Npixels = np.interp(tmag, np.array([8.0, 9.0, 10.0, 12.0, 14.0, 16.0]), np.array([350.0, 200.0, 125.0, 100.0, 50.0, 40.0]))

				cursor.execute("INSERT INTO todolist (priority,starid,tmag,datasource,status,camera,ccd,cbv_area) VALUES (?,?,?,?,1,?,0,0);", (
					pri,
					starid,
					tmag,
					datasource,
					camera
				))
				cursor.execute("INSERT INTO diagnostics (priority,starid,lightcurve,elaptime,mean_flux,variance,variability,mask_size,pos_row,pos_column,contamination,stamp_resizes,eclon,eclat) VALUES (?,?,?,?,?,?,?,?,0,0,0.0,0,?,?);", (
					pri,
					starid,
					lightcurve,
					elaptime,
					mean_flux,
					variance,
					variability,
					int(Npixels),
					ecllon,
					ecllat
				))

			"""
			logger.info("Step 2: Figuring out where targets are on CCDs...")
			cursor.execute("SELECT MIN(eclon) AS min_eclon, MAX(eclon) AS max_eclon FROM diagnostics;")
			row = cursor.fetchone()
			eclon_min = row['min_eclon']
			eclon_max = row['max_eclon']

			cursor.execute("SELECT todolist.priority, camera, eclat, eclon FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority;")
			results = cursor.fetchall()
			for row in tqdm(results, total=len(results)):
				frac_lon = (row['eclon']-eclon_min) / (eclon_max-eclon_min)
				offset = (row['camera']-1)*24 + 6.0
				frac_lat = (row['eclat']-offset) / 24.0
				if frac_lon <= 0.5 and frac_lat <= 0.5:
					ccd = 1
				elif frac_lon > 0.5 and frac_lat <= 0.5:
					ccd = 2
				elif frac_lon <= 0.5 and frac_lat > 0.5:
					ccd = 3
				elif frac_lon > 0.5 and frac_lat > 0.5:
					ccd = 4
				else:
					raise Exception("WHAT?")

				pos_column = 4096 * frac_lon
				if pos_column > 2048: pos_column -= 2048
				pos_row = 4096 * frac_lat
				if pos_row > 2048: pos_row -= 2048

				cbv_area = 100*camera + 10*ccd
				if pos_row > 1024 and pos_column > 1024:
					cbv_area += 4
				elif pos_row > 1024 and pos_column <= 1024:
					cbv_area += 3
				elif pos_row <= 1024 and pos_column > 1024:
					cbv_area += 2
				else:
					cbv_area += 1

				cursor.execute("UPDATE todolist SET ccd=?, cbv_area=? WHERE priority=?;", (ccd, cbv_area, row['priority']))
				cursor.execute("UPDATE diagnostics SET pos_column=?, pos_row=? WHERE priority=?;", (pos_column, pos_row, row['priority']))
			"""

			conn.commit()
			cursor.close()

		logger.info("DONE.")

	#----------------------------------------------------------------------------------------------
	def training_set_features(self):

		rowidx = -1
		with TaskManager(self.input_folder, overwrite=True) as tm:
			with BaseClassifier(features_cache=os.path.join(self.input_folder, 'features_cache_%s' % self.datalevel)) as stcl:
				while True:
					task = tm.get_task(classifier=self.classifier)
					if task is None: break
					tm.start_task(task)
					rowidx += 1

					# Lightcurve file to load:
					# We do not use the one from the database because in the simulations the
					# raw and corrected light curves are stored in different files.
					fname = os.path.join(self.input_folder, task['lightcurve'])

					if self.testfraction > 0:
						if rowidx in self.train_idx:
							yield stcl.load_star(task, fname)
					else:
						yield stcl.load_star(task, fname)

	#----------------------------------------------------------------------------------------------
	def training_set_labels(self, level='L1'):

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
	def training_set_features_test(self):

		if self.testfraction == 0:
			raise ValueError('training_set_features_test requires testfraction>0')
		else:

			rowidx = -1
			with TaskManager(self.input_folder, overwrite=True) as tm:
				with BaseClassifier(features_cache=os.path.join(self.input_folder, 'features_cache_%s' % self.datalevel)) as stcl:
					while True:
						task = tm.get_task(classifier=self.classifier)
						if task is None: break
						tm.start_task(task)
						rowidx += 1

						# Lightcurve file to load:
						# We do not use the one from the database because in the simulations the
						# raw and corrected light curves are stored in different files.
						fname = os.path.join(self.input_folder, task['lightcurve'])

						if self.testfraction > 0:
							if rowidx in self.test_idx:
								yield stcl.load_star(task, fname)
						else:
							yield stcl.load_star(task, fname)

	#----------------------------------------------------------------------------------------------
	def training_set_labels_test(self, level='L1'):

		logger = logging.getLogger(__name__)

		if self.testfraction == 0:
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

				if rowidx in self.test_idx:
					lookup.append(tuple(set(lbls)))

			return tuple(lookup)
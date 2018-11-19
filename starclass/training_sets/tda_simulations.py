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
from .. import StellarClasses, BaseClassifier

# Point this to the directory where the TDA simulations are stored
# URL: https://tasoc.dk/wg0/SimData
# The directories "sysnoise", "noisy" and "clean" should exist in this directory
INPUT_DIR = r'/Users/fingeza/Documents/tda_simulated_data/'

#----------------------------------------------------------------------------------------------
class tda_simulations(object):

	def __init__(self, datalevel='corr'):
		self.input_folder = INPUT_DIR
		self.datalevel = datalevel

		self.features_cache = os.path.join(INPUT_DIR, 'features_cache_%s' % datalevel)

		if not os.path.exists(self.features_cache):
			os.makedirs(self.features_cache)

		sqlite_file = os.path.join(INPUT_DIR, 'todo.sqlite')
		if not os.path.exists(sqlite_file):
			self.generate_todolist()


	def generate_todolist(self):

		logger = logging.getLogger(__name__)

		# Make sure some directories exist:
		#os.makedirs(os.path.join(INPUT_DIR, 'sysnoise_by_sectors'), exist_ok=True)
		#os.makedirs(os.path.join(INPUT_DIR, 'noisy_by_sectors'), exist_ok=True)
		#os.makedirs(os.path.join(INPUT_DIR, 'clean_by_sectors'), exist_ok=True)

		sqlite_file = os.path.join(INPUT_DIR, 'todo.sqlite')
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
			starlist = np.genfromtxt(os.path.join(INPUT_DIR, 'Data_Batch_TDA4_r1.txt'), delimiter=',', dtype=None)
			for k, star in tqdm(enumerate(starlist), total=len(starlist)):
				#print(star)

				# Get starid:
				starname = star[0]
				if not isinstance(starname, six.string_types): starname = starname.decode("utf-8") # For Python 3
				starid = int(starname[4:])


				#data_sysnoise = np.loadtxt(os.path.join(INPUT_DIR, 'sysnoise', 'Star%d.sysnoise' % starid))
				#data_noisy = np.loadtxt(os.path.join(INPUT_DIR, 'noisy', 'Star%d.noisy' % starid))
				data_clean = np.loadtxt(os.path.join(INPUT_DIR, 'clean', 'Star%d.clean' % starid))

				# Just because Mikkel can not be trusted:
				#if star[2] == 1800:
				if (data_sysnoise[1,0] - data_sysnoise[0,0])*86400 > 1000:
					datasource = 'ffi'
				else:
					datasource = 'tpf'

				# Extract the camera from the lattitude:
				tmag = star[1]
				ecllat = star[4]
				ecllon = star[5]
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
				if data_sysnoise[-1,0] - data_sysnoise[0,0] > 27.4:
					raise Exception("Okay, didn't we agree that this should be only one sector?!")

				#indx = (sector == s)
				#data_sysnoise_sector = data_sysnoise[indx, :]
				#data_noisy_sector = data_noisy[indx, :]
				#data_clean_sector = data_clean[indx, :]

				lightcurve = 'Star%d' % (starid)
				#lightcurve = 'Star%d-sector%02d' % (starid, s)

				# Save files cut up into sectors:
				#np.savetxt(os.path.join(INPUT_DIR, 'sysnoise_by_sectors', lightcurve + '.sysnoise'), data_sysnoise_sector, fmt=('%.8f', '%.18e', '%.18e', '%d'), delimiter='  ')
				#np.savetxt(os.path.join(INPUT_DIR, 'noisy_by_sectors', lightcurve + '.noisy'), data_noisy_sector, fmt=('%.8f', '%.18e', '%.18e', '%d'), delimiter='  ')
				#np.savetxt(os.path.join(INPUT_DIR, 'clean_by_sectors', lightcurve + '.clean'), data_clean_sector, fmt=('%.9f', '%.18e', '%.18e', '%d'), delimiter='  ')

				#sqlite_file = os.path.join(INPUT_DIR, 'todo-sector%02d.sqlite' % s)

				pri += 1
				mean_flux = nanmedian(data_sysnoise[:,1])
				variance = nansum((data_sysnoise[:,1] - mean_flux)**2) / (data_sysnoise.shape[0] - 1)

				# This could be done in the photometry code as well:
				time = data_sysnoise[:,0]
				flux = data_sysnoise[:,1] / mean_flux
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
					'sysnoise/' + lightcurve + '.sysnoise',
					elaptime,
					mean_flux,
					variance,
					variability,
					int(Npixels),
					ecllon,
					ecllat
				))

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

			conn.commit()
			cursor.close()

		logger.info("DONE.")

	#----------------------------------------------------------------------------------------------
	def training_set_features(self):

		todo_file = os.path.join(INPUT_DIR, 'todo.sqlite')
		data = np.genfromtxt(os.path.join(INPUT_DIR, 'Data_Batch_TDA4_r1.txt'), dtype=None, delimiter=', ', usecols=(0,10), encoding='utf-8')

		with sqlite3.connect(todo_file) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()

			with BaseClassifier(features_cache=os.path.join(INPUT_DIR, 'features_cache_%s' % self.datalevel)) as stcl:
				for row in data:
					starid = int(row[0][4:])

					# Get task info from database:
					cursor.execute("SELECT * FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE todolist.starid=?;", (starid, ))
					task = dict(cursor.fetchone())

					# Lightcurve file to load:
					# We do not use the one from the database because in the simulations the
					# raw and corrected light curves are stored in different files.
					if self.datalevel == 'raw':
						fname = os.path.join(INPUT_DIR, 'sysnoise', 'Star%d.sysnoise' % starid) # # These are the lightcurves INCLUDING SYSTEMATIC NOISE
					elif self.datalevel == 'corr':
						fname = os.path.join(INPUT_DIR, 'noisy', 'Star%d.noisy' % starid) # # These are the lightcurves WITHOUT SYSTEMATIC NOISE

					yield stcl.load_star(task, fname)

	#----------------------------------------------------------------------------------------------
	def training_set_labels(self, level='L1'):

		logger = logging.getLogger(__name__)

		data = np.genfromtxt(os.path.join(INPUT_DIR, 'Data_Batch_TDA4_r1.txt'), dtype=None, delimiter=', ', usecols=(0,10), encoding='utf-8')

		# Translation of Mikkel's identifiers into the broader
		# classes we have defined in StellarClasses:
		if level == 'L1':
			translate = {
				'Solar-like': StellarClasses.SOLARLIKE,
				'Transit': StellarClasses.ECLIPSE,
				'Eclipse': StellarClasses.ECLIPSE, #short period EBs should be CONTACT_ROT, not ECLIPSE
				'multi': StellarClasses.ECLIPSE,
				'MMR': StellarClasses.ECLIPSE,
				'RR Lyrae': StellarClasses.RRLYR_CEPHEID,
				'RRab': StellarClasses.RRLYR_CEPHEID,
				'RRc': StellarClasses.RRLYR_CEPHEID,
				'RRd':  StellarClasses.RRLYR_CEPHEID,
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
				'RRd':  StellarClasses.RRLYR,
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
		for row in data:
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

			lookup.append(tuple(set(lbls)))

		return tuple(lookup)
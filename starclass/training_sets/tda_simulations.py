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
#from bottleneck import nanmedian, nanvar
#from lightkurve import LightCurve
from tqdm import tqdm
from ..StellarClasses import StellarClasses, StellarClassesLevel2
from . import TrainingSet
#from ..utilities import rms_timescale

#--------------------------------------------------------------------------------------------------
def _generate_todolist(self):

	logger = logging.getLogger(__name__)

	sqlite_file = os.path.join(self.input_folder, 'todo.sqlite')
	with closing(sqlite3.connect(sqlite_file)) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()

		# Create the basic file structure of a TODO-list:
		self.generate_todolist_structure(conn)

		logger.info("Step 3: Reading file and extracting information...")
		starlist = np.genfromtxt(os.path.join(self.input_folder, 'Data_Batch_TDA4_r1.txt'),
			delimiter=',', dtype=None, encoding='utf-8')
		diagnostics = np.genfromtxt(os.path.join(self.input_folder, 'diagnostics.txt'),
			delimiter=',', dtype=None, encoding='utf-8')

		pri = 0
		for k, star in tqdm(enumerate(starlist), total=len(starlist)):
			# Get starid:
			starname = star[0]
			starid = int(starname[4:])

			# Path to lightcurve:
			if self.key == 'tdasim-raw':
				lightcurve = 'sysnoise/Star%d.sysnoise' % starid
			elif self.key == 'tdasim':
				lightcurve = 'noisy/Star%d.noisy' % starid
			elif self.key == 'tdasim-clean':
				lightcurve = 'clean/Star%d.clean' % starid

			# Extract the camera from the lattitude:
			tmag = star[1]
			#ecllat = star[4]
			#ecllon = star[5]

			# Load diagnostics from file, to speed up the process:
			variance, rms_hour, ptp = diagnostics[k]

			#data = np.loadtxt(os.path.join(self.input_folder, lightcurve))
			#sector = np.floor(data_sysnoise[:,0] / 27.4) + 1
			#sectors = [int(s) for s in np.unique(sector)]
			#if data[-1,0] - data[0,0] > 27.4:
			#	raise Exception("Okay, didn't we agree that this should be only one sector?!")

			#data[:,0] -= data[0,0]
			#m = nanmedian(data[:,1])
			#data[:,1] = 1e6*(data[:,1]/m - 1)
			#data[:,2] = 1e6*data[:,2]/m

			# Calculate diagnostics:
			#lc = LightCurve(time=data[:,0], flux=data[:,1], flux_err=data[:,2])
			#variance = nanvar(data[:,1], ddof=1)
			#rms_hour = rms_timescale(lc, timescale=3600/86400)
			#ptp = nanmedian(np.abs(np.diff(data[:,1])))

			# Add target to TODO-list:
			#diag.write("{variance:.16e},{rms_hour:.16e},{ptp:.16e}\n".format(
			#	variance=variance,
			#	rms_hour=rms_hour,
			#	ptp=ptp
			#))

			pri += 1
			self.generate_todolist_insert(cursor,
				priority=pri,
				starid=starid,
				lightcurve=lightcurve,
				tmag=tmag,
				datasource='ffi',
				variance=variance,
				rms_hour=rms_hour,
				ptp=ptp)

		conn.commit()
		cursor.close()

	logger.info("%s training set successfully built.", self.key)

#--------------------------------------------------------------------------------------------------
def _labels(self, level='L1'):

	logger = logging.getLogger(__name__)

	data = np.genfromtxt(os.path.join(self.input_folder, 'Data_Batch_TDA4_r1.txt'),
		dtype=None, delimiter=', ', usecols=(0,10), encoding='utf-8')

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
			#'roAp': StellarClasses.RAPID,
			#'sdBV': StellarClasses.RAPID,
			#'Flare': StellarClasses.TRANSIENT,
			'Spots': StellarClasses.CONTACT_ROT,
			'LPV': StellarClasses.APERIODIC,
			'MIRA': StellarClasses.APERIODIC,
			'SR': StellarClasses.APERIODIC,
			'Constant': StellarClasses.CONSTANT
		}
	elif level == 'L2':
		translate = {
			'Solar-like': StellarClassesLevel2.SOLARLIKE,
			'Transit': StellarClassesLevel2.ECLIPSE,
			'Eclipse': StellarClassesLevel2.ECLIPSE,
			'multi': StellarClassesLevel2.ECLIPSE,
			'MMR': StellarClassesLevel2.ECLIPSE,
			'RR Lyrae': StellarClassesLevel2.RRLYR,
			'RRab': StellarClassesLevel2.RRLYR,
			'RRc': StellarClassesLevel2.RRLYR,
			'RRd': StellarClassesLevel2.RRLYR,
			'Cepheid': StellarClassesLevel2.CEPHEID,
			'FM': StellarClassesLevel2.CEPHEID,
			'1O': StellarClassesLevel2.CEPHEID,
			'1O2O': StellarClassesLevel2.CEPHEID,
			'FM1O': StellarClassesLevel2.CEPHEID,
			'Type II': StellarClassesLevel2.CEPHEID,
			'Anomaleous': StellarClassesLevel2.CEPHEID,
			'SPB': StellarClassesLevel2.SPB,
			'dsct': StellarClassesLevel2.DSCT,
			'bumpy': StellarClassesLevel2.DSCT, # This is not right - Should we make a specific class for these?
			'gDor': StellarClassesLevel2.GDOR,
			'bCep': StellarClassesLevel2.BCEP,
			'roAp': StellarClassesLevel2.ROAP,
			'sdBV': StellarClassesLevel2.SDB,
			'Flare': StellarClassesLevel2.FLARE,
			'Spots': StellarClassesLevel2.SPOTS,
			'LPV': StellarClassesLevel2.LPV,
			'MIRA': StellarClassesLevel2.LPV,
			'SR': StellarClassesLevel2.LPV,
			'Constant': StellarClassesLevel2.CONSTANT
		}

	# Create list of all the classes for each star:
	lookup = []
	for rowidx in self.train_idx:
		row = data[rowidx, :]
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
					lbls.append(StellarClassesLevel2.DSCT)
					lbls.append(StellarClassesLevel2.GDOR)
			elif lbl == 'bCep+SPB hybrid':
				if level == 'L1':
					lbls.append(StellarClasses.DSCT_BCEP)
					lbls.append(StellarClasses.GDOR_SPB)
				elif level == 'L2':
					lbls.append(StellarClassesLevel2.BCEP)
					lbls.append(StellarClassesLevel2.SPB)
			else:
				c = translate.get(lbl.strip())
				if c is None:
					logger.error("Unknown label: %s", lbl)
				else:
					lbls.append(c)

		lookup.append(tuple(set(lbls)))

	return tuple(lookup)

#--------------------------------------------------------------------------------------------------
def _labels_test(self, level='L1'):

	logger = logging.getLogger(__name__)

	if self.testfraction == 0:
		return []
	else:
		data = np.genfromtxt(os.path.join(self.input_folder, 'Data_Batch_TDA4_r1.txt'),
			dtype=None, delimiter=', ', usecols=(0,10), encoding='utf-8')

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
				#'roAp': StellarClasses.RAPID,
				#'sdBV': StellarClasses.RAPID,
				#'Flare': StellarClasses.TRANSIENT,
				'Spots': StellarClasses.CONTACT_ROT,
				'LPV': StellarClasses.APERIODIC,
				'MIRA': StellarClasses.APERIODIC,
				'SR': StellarClasses.APERIODIC,
				'Constant': StellarClasses.CONSTANT
			}
		elif level == 'L2':
			translate = {
				'Solar-like': StellarClassesLevel2.SOLARLIKE,
				'Transit': StellarClassesLevel2.ECLIPSE,
				'Eclipse': StellarClassesLevel2.ECLIPSE,
				'multi': StellarClassesLevel2.ECLIPSE,
				'MMR': StellarClassesLevel2.ECLIPSE,
				'RR Lyrae': StellarClassesLevel2.RRLYR,
				'RRab': StellarClassesLevel2.RRLYR,
				'RRc': StellarClassesLevel2.RRLYR,
				'RRd': StellarClassesLevel2.RRLYR,
				'Cepheid': StellarClassesLevel2.CEPHEID,
				'FM': StellarClassesLevel2.CEPHEID,
				'1O': StellarClassesLevel2.CEPHEID,
				'1O2O': StellarClassesLevel2.CEPHEID,
				'FM1O': StellarClassesLevel2.CEPHEID,
				'Type II': StellarClassesLevel2.CEPHEID,
				'Anomaleous': StellarClassesLevel2.CEPHEID,
				'SPB': StellarClassesLevel2.SPB,
				'dsct': StellarClassesLevel2.DSCT,
				'bumpy': StellarClassesLevel2.DSCT, # This is not right - Should we make a specific class for these?
				'gDor': StellarClassesLevel2.GDOR,
				'bCep': StellarClassesLevel2.BCEP,
				'roAp': StellarClassesLevel2.ROAP,
				'sdBV': StellarClassesLevel2.SDB,
				'Flare': StellarClassesLevel2.FLARE,
				'Spots': StellarClassesLevel2.SPOTS,
				'LPV': StellarClassesLevel2.LPV,
				'MIRA': StellarClassesLevel2.LPV,
				'SR': StellarClassesLevel2.LPV,
				'Constant': StellarClassesLevel2.CONSTANT
			}

		# Create list of all the classes for each star:
		lookup = []
		for rowidx in self.test_idx:
			row = data[rowidx, :]
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
						lbls.append(StellarClassesLevel2.DSCT)
						lbls.append(StellarClassesLevel2.GDOR)
				elif lbl == 'bCep+SPB hybrid':
					if level == 'L1':
						lbls.append(StellarClasses.DSCT_BCEP)
						lbls.append(StellarClasses.GDOR_SPB)
					elif level == 'L2':
						lbls.append(StellarClassesLevel2.BCEP)
						lbls.append(StellarClassesLevel2.SPB)
				else:
					c = translate.get(lbl.strip())
					if c is None:
						logger.error("Unknown label: %s", lbl)
					else:
						lbls.append(c)

			lookup.append(tuple(set(lbls)))

	return tuple(lookup)

#--------------------------------------------------------------------------------------------------
class tdasim(TrainingSet):
	# Class constants:
	key = 'tdasim'

	def __init__(self, *args, **kwargs):

		self.input_folder = self.tset_datadir('https://tasoc.dk/pipeline/starclass_trainingsets/tdasim.zip')

		data = np.genfromtxt(os.path.join(self.input_folder, 'Data_Batch_TDA4_r1.txt'),
			dtype=None, delimiter=',', usecols=(0,10), encoding='utf-8')
		self.nobjects = data.shape[0]

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

	def generate_todolist(self):
		return _generate_todolist(self)

	def labels(self, *args, **kwargs):
		return _labels(self, *args, **kwargs)

	def labels_test(self, *args, **kwargs):
		return _labels_test(self, *args, **kwargs)

#--------------------------------------------------------------------------------------------------
class tdasim_raw(TrainingSet):
	# Class constants:
	key = 'tdasim-raw'

	def __init__(self, *args, **kwargs):

		self.input_folder = self.tset_datadir('https://tasoc.dk/pipeline/starclass_trainingsets/tdasim-raw.zip')

		data = np.genfromtxt(os.path.join(self.input_folder, 'Data_Batch_TDA4_r1.txt'),
			dtype=None, delimiter=',', usecols=(0,10), encoding='utf-8')
		self.nobjects = data.shape[0]

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

	def generate_todolist(self):
		return _generate_todolist(self)

	def labels(self, *args, **kwargs):
		return _labels(self, *args, **kwargs)

	def labels_test(self, *args, **kwargs):
		return _labels_test(self, *args, **kwargs)

#--------------------------------------------------------------------------------------------------
class tdasim_clean(TrainingSet):
	# Class constants:
	key = 'tdasim-clean'

	def __init__(self, *args, **kwargs):

		# Key for this training-set:
		self.input_folder = self.tset_datadir('https://tasoc.dk/pipeline/starclass_trainingsets/tdasim-clean.zip')

		data = np.genfromtxt(os.path.join(self.input_folder, 'Data_Batch_TDA4_r1.txt'),
			dtype=None, delimiter=',', usecols=(0,10), encoding='utf-8')
		self.nobjects = data.shape[0]

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

	def generate_todolist(self):
		return _generate_todolist(self)

	def labels(self, *args, **kwargs):
		return _labels(self, *args, **kwargs)

	def labels_test(self, *args, **kwargs):
		return _labels_test(self, *args, **kwargs)

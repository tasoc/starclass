#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kepler Q9 Training Set (version 3).

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
import numpy as np
from . import TrainingSet
from ..StellarClasses import StellarClassesLevel1Instr

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
		self._valid_indicies = np.arange(self.starlist.shape[0], dtype=int)
		indx = [star[1] != 'INSTRUMENT' for star in self.starlist]
		self._valid_indicies = self._valid_indicies[indx]

		# Count the number of objects in trainingset:
		self.nobjects = len(self._valid_indicies)

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

#--------------------------------------------------------------------------------------------------
class keplerq9v3_instr(TrainingSet):
	"""
	Kepler Q9 Training Set (version 3) including instrumental class.

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

		# Pin the StellarClasses Enum to special values for this training set:
		self.StellarClasses = StellarClassesLevel1Instr

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kepler Q9 Training Set (version 3).

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

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

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

	def load_targets(self):
		# Remove the instrumental class from this trainingset:
		starlist = super().load_targets()
		indx = (starlist['starclass'] != 'INSTRUMENT')
		return starlist[indx]

#--------------------------------------------------------------------------------------------------
class keplerq9v3_instr(TrainingSet):
	"""
	Kepler Q9 Training Set (version 3) including instrumental class.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	# Class constants:
	key = 'keplerq9v3-instr'
	_todo_name = 'todo-instr'
	datadir = 'keplerq9v3'

	def __init__(self, *args, datalevel='corr', **kwargs):

		if datalevel != 'corr':
			raise ValueError("The KeplerQ9v3 training set only as corrected data. Please specify datalevel='corr'.")

		# Point this to the directory where the TDA simulations are stored
		self.input_folder = self.tset_datadir('https://tasoc.dk/pipeline/starclass_trainingsets/keplerq9v3.zip')

		# Pin the StellarClasses Enum to special values for this training set:
		self.StellarClasses = StellarClassesLevel1Instr

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

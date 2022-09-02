#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kepler Q9 Training Set.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

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

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

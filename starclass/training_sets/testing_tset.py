
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Set only used for unit-testing.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from sklearn.model_selection import train_test_split
from .training_set import TrainingSet

#--------------------------------------------------------------------------------------------------
class testing_tset(TrainingSet):
	"""
	Training Set only used for unit-testing.

	Very down-scaled version of the Kepler Q9 Training Set (version 3).

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	# Class constants:
	key = 'testing'
	datadir = 'keplerq9v3'
	_todo_name = 'todo-testing'

	def __init__(self, *args, datalevel='corr', **kwargs):

		if datalevel != 'corr':
			raise ValueError("The KeplerQ9v3 training set only as corrected data. Please specify datalevel='corr'.")

		# Point this to the directory where the TDA simulations are stored
		self.input_folder = self.tset_datadir('https://tasoc.dk/pipeline/starclass_trainingsets/keplerq9v3.zip')

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

	def load_targets(self):

		starlist = super().load_targets()

		indx = (starlist['starclass'] != 'INSTRUMENT')
		starlist = starlist[indx]

		# TODO: Doesn't work with multiple labels
		_, idx = train_test_split(
			np.arange(len(starlist), dtype=int),
			test_size=200/len(starlist),
			stratify=starlist['starclass'],
			random_state=2187
		)
		starlist = starlist[idx]

		return starlist

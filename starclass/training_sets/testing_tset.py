
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Set only used for unit-testing.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import os.path
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
	key = 'testtset'
	datadir = 'keplerq9v3'
	_todo_name = 'todo-testing'

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

		self.starlist = self.starlist[self.starlist[:, 1] != 'INSTRUMENT', :]
		_, idx = train_test_split(
			np.arange(len(self.starlist)),
			test_size=200/len(self.starlist),
			stratify=self.starlist[:, 1],
			random_state=2187
		)
		self.starlist = self.starlist[idx]

		# Initialize parent
		# NOTE: We do this after setting the input_folder, as it depends on that being set:
		super().__init__(*args, **kwargs)

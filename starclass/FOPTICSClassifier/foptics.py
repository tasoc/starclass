#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import numpy as np
from .. import BaseClassifier, StellarClasses

class FOPTICSClassifier(BaseClassifier):


	def __init__(self, *args, **kwargs):
		"""
		Initialization for the class.

		"""

		# Initialize parent:
		super(self.__class__, self).__init__(*args, **kwargs)

		print("make clean -f Makefile_FOPTICS")
		print("make -f Makefile_FOPTICS")


	def train(self, tset):
		INPUT_FILE = os.path.abspath('input.dat')
		OUTPUT_FILE = 'output.txt'
		num_attributes = 4
		undefined_distance = 1.0
		min_neighbors = 20

		print(INPUT_FILE)
		with open(INPUT_FILE, 'w') as fid:
			for feat in tset.features():
				fid.write("Priority{priority:04d} {freq1:e} {amp1:e} {freq2:e} {amp2:e}\n".format(**feat))

		print("foptics %s %s %d %f %d" % (INPUT_FILE, OUTPUT_FILE, num_attributes, undefined_distance, min_neighbors))


	def do_classify(self, features):

		pass

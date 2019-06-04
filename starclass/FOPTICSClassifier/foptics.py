#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import logging
import numpy as np
from scipy.stats import skew
import subprocess
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

	def calculate_additional_features(self, feat):
		feat['phasediff_21'] = feat['phase2'] - feat['phase1']
		feat['multi_per'] = np.log10(feat['amp1']/feat['amp2'])
		feat['power'] = np.log10( (feat['amp1']**2 + feat['amp1']**2 + feat['amp1']**2 + feat['amp1']**2) / feat['amp1']**2 - 1 )
		feat['skew'] = skew(feat['lightcurve'].flux, bias=False, nan_policy='omit')
		return feat

	def train(self, tset):
	
		logger = logging.getLogger(__name__)
	
		INPUT_FILE = os.path.abspath(os.path.join(self.features_cache, 'foptics_input.dat'))
		OUTPUT_FILE = os.path.abspath(os.path.join(self.features_cache, 'foptics_output.dat'))
		num_attributes = 7
		undefined_distance = 1.0
		min_neighbors = 20

		logger.debug(INPUT_FILE)
		if not os.path.exists(INPUT_FILE):
			with open(INPUT_FILE, 'w') as fid:
				for feat in tset.features():
					feat = self.calculate_additional_features(feat)
					fid.write("Priority{priority:05d} {freq1:e} {amp1:e} {amp2:e} {phasediff_21:e} {multi_per:e} {power:e} {skew:e}\n".format(**feat))

		# Construct command to be issued, calling the SLSCLEAN program:
		cmd = 'foptics "%s" "%s" %d %f %d' % (
			os.path.dirname(INPUT_FILE),
			OUTPUT_FILE,
			num_attributes,
			undefined_distance,
			min_neighbors
		)
		logger.debug("Running command: %s", cmd)

		# Call the FOPTICS program in a subprocess:
		p = subprocess.Popen(cmd, cwd=os.path.dirname(__file__), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
		ret = p.communicate()
		logger.debug(ret[0])
		if p.returncode != 0:
			raise Exception(ret[1])
		elif ret[1] != '':
			raise Exception(ret[1])
			
		

	def do_classify(self, features):

		features = calculate_additional_features(self, features)

		pass

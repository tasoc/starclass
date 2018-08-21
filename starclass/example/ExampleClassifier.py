#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An example classifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import logging
from .. import BaseClassifier, StellarClasses

class ExampleClassifier(BaseClassifier):
	"""
	An example classifier.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, *args, **kwargs):
		"""
		Initialize the classifier object.
		"""
		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(*args, **kwargs)
		
		# Here you could do other things that needs doing
		# when the classifier is loaded in.

		# Load stuff:
		self.something = np.load('my_classifier.npy')
		
		
	def do_classify(self, lightcurve, features):
		"""
		My classification that will be run on each lightcurve
		
		Parameters:
			lightcurve (``lightkurve.TessLightCurve`` object): Lightcurve.
			features (dict): Dictionary of other features.
			
		Returns:
			dict: Dictionary of stellar classifications.
		"""
		
		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)
		
		# Do the magic:
		logger.info("We are staring the magic...")
		self.something.doit(lightcurve, features)
		
		# Dummy result where the target is 98% a solar-like
		# and 2% classical pulsator:
		result = {
			StellarClasses.SOLARLIKE: 0.98,
			StellarClasses.CLASSICAL: 0.02
		}
		
		# If something went wrong:
		logger.warning("This is a warning")
		logger.error("This is an error")
		
		return result

		
	def train(self, lightcurves, labels):
		# Do all the stuff needed to train the classifier here
		pass
		
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An example classifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import numpy as np
from .. import BaseClassifier

#--------------------------------------------------------------------------------------------------
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
		super().__init__(*args, **kwargs)

		# Here you could do other things that needs doing
		# when the classifier is loaded in.

		# Load stuff:
		self.something = np.load('my_classifier.npy')

	#----------------------------------------------------------------------------------------------
	def do_classify(self, features):
		"""
		My classification that will be run on each lightcurve.

		Parameters:
			features (dict): Dictionary of features.
				Of particular interest should be the `lightcurve` (``lightkurve.TessLightCurve`` object) and
				`powerspectum` which contains the lightcurve and power density spectrum respectively.

		Returns:
			dict: Dictionary of stellar classifications.
		"""

		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		# Do the magic:
		logger.info("We are staring the magic...")
		self.something.doit(features['lightcurve'], features)

		# Dummy result where the target is 98% a solar-like
		# and 2% classical pulsator (delta Scuti/beta Cep):
		result = {
			self.StellarClasses.SOLARLIKE: 0.98,
			self.StellarClasses.DSCT_BCEP: 0.02
		}

		# If something went wrong:
		logger.warning("This is a warning")
		logger.error("This is an error")

		return result

	#----------------------------------------------------------------------------------------------
	def train(self, features, labels):
		"""
		Train the classifier.

		Parameters:
			features (iterator of dicts): Iterator of features-dictionaries similar to those in ``do_classify``.
			labels (iterator of lists): For each feature, provides a list of the assigned known ``StellarClasses`` identifications.
		"""
		# Do all the stuff needed to train the classifier here

		my_classifier = do_the_training(features, labels)
		np.save('my_classifier.npy', my_classifier)

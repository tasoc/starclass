#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An example classifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import numpy as np
from .. import BaseClassifier
from ..exceptions import UntrainedClassifierError

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

		# Define names of features used:
		self.features_names = ['rms', 'ptp']

		# Optional: Remove if not applicable.
		# Link to the internal classifier model,
		# which can be used for calculating feature importances:
		self._classifier_model = self.something

	#----------------------------------------------------------------------------------------------
	def do_classify(self, features):
		"""
		My classification that will be run on each lightcurve.

		Parameters:
			features (dict): Dictionary of features.
				Of particular interest should be the `lightcurve` (``lightkurve.TessLightCurve`` object) and
				`powerspectum` which contains the lightcurve and power density spectrum respectively.

		Returns:
			tuple:
			- dict: Dictionary of stellar classifications.
			- list: Features used for classification.

		Raises:
			UntrainedClassifierError: If classifier has not been trained.
		"""

		# Start a logger that should be used to output e.g. debug information:
		logger = logging.getLogger(__name__)

		if not self.trained: # This needs to be defined somehow!
			raise UntrainedClassifierError("Classifier has not been trained")

		# Do the magic:
		logger.info("We are staring the magic...")
		featarray = [features['rms'], features['ptp']]
		self.something.predict(featarray)

		# Dummy result where the target is 98% a solar-like
		# and 2% classical pulsator (delta Scuti/beta Cep):
		result = {
			self.StellarClasses.SOLARLIKE: 0.98,
			self.StellarClasses.DSCT_BCEP: 0.02
		}

		# If something went wrong:
		logger.warning("This is a warning")
		logger.error("This is an error")

		return result, featarray

	#----------------------------------------------------------------------------------------------
	def train(self, features, labels):
		"""
		Train the classifier.

		Parameters:
			features (iterator of dicts): Iterator of features-dictionaries similar to those in ``do_classify``.
			labels (iterator of lists): For each feature, provides a list of the assigned known ``StellarClasses`` identifications.
		"""
		# Do all the stuff needed to train the classifier here

		my_classifier = do_the_training(features, labels) # noqa: F821
		np.save('my_classifier.npy', my_classifier)

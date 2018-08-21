#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The basic stellar classifier class for the TASOC pipeline.
All other specific stellar classification algorithms will inherit from BaseClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import os.path
import logging
import enum
from lightkurve import TessLightCurveFile

__docformat__ = 'restructuredtext'

class BaseClassifier(object):
	"""
	The basic stellar classifier class for the TASOC pipeline.
	All other specific stellar classification algorithms will inherit from BaseClassifier.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, plot=False):
		"""
		Initialize the classifier object.

		Parameters:
			plot (boolean, optional): Create plots as part of the output. Default is ``False``.
		"""

		# Store the input:
		self.plot = plot
		
	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()
	
	def close(self):
		"""Close the classifier."""
		pass
	
	def classify(self, lightcurve, features):
		"""
		Classify a star from the lightcurve and other features.

		Will run the :py:func:`do_classify` method and
		check some of the output and calculate various
		performance metrics.

		Parameters:
			lightcurve (``lightkurve.TessLightCurve`` object): Lightcurve.
			features (dict): Dictionary of other features.
			
		Returns:
			dict: Dictionary of classifications

		See Also:
			:py:func:`do_classify`

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		res = self.do_classify(lightcurve, features)
		# Check results
		return res
		
	def do_classify(self, lightcurve, features):
		"""
		This method should be overwritten by child classes.

		Raises:
			NotImplementedError
		"""
		raise NotImplementedError()
		
	def load_star(self, starid):
		fname = os.path.join()
		
		lightcurve = TessLightCurveFile(fname, default_mask)
		features = {}
		return lightcurve, features
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The basic stellar classifier class for the TASOC pipeline.
All other specific stellar classification algorithms will inherit from BaseClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import os.path
from lightkurve import TessLightCurveFile

class BaseClassifier(object):

	def __init__(self):
		pass
		
	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()
	
	def close(self):
		pass
	
	def classify(self, features, lightcurve):
		res = self.do_classify(features, lightcurve)
		# Check results
		return res
		
	def do_classify(self, features, lightcurve):
		raise NotImplementedError()
		
	def load_star(self, starid):
		fname = os.path.join()
		
		lightcurve = TessLightCurveFile(fname, default_mask)
		return features, lightcurve
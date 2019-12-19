#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of BaseClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from starclass import BaseClassifier

#----------------------------------------------------------------------
def test_baseclassifier_import():
	with BaseClassifier() as cl:
		assert(cl.__class__.__name__ == 'BaseClassifier')

#----------------------------------------------------------------------
if __name__ == '__main__':
	test_baseclassifier_import()

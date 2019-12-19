#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of MetaClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from starclass import MetaClassifier

#----------------------------------------------------------------------
def test_metaclassifier_import():
	with MetaClassifier() as cl:
		assert(cl.__class__.__name__ == 'MetaClassifier')

#----------------------------------------------------------------------
if __name__ == '__main__':
	test_metaclassifier_import()

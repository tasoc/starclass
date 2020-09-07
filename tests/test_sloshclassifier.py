#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of SLOSHClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import conftest # noqa: F401
from starclass import SLOSHClassifier

#----------------------------------------------------------------------
def test_sloshclassifier_import():
	with SLOSHClassifier() as cl:
		assert(cl.__class__.__name__ == 'SLOSHClassifier')

#----------------------------------------------------------------------
if __name__ == '__main__':
	test_sloshclassifier_import()

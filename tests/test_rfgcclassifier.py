#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of RFGCClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import conftest # noqa: F401
from starclass import RFGCClassifier

#----------------------------------------------------------------------
def test_rfgcclassifier_import():
	with RFGCClassifier() as cl:
		assert(cl.__class__.__name__ == 'RFGCClassifier')

#----------------------------------------------------------------------
if __name__ == '__main__':
	test_rfgcclassifier_import()

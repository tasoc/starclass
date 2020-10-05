#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of SortingHatClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import conftest # noqa: F401
from starclass import SortingHatClassifier

#----------------------------------------------------------------------
def test_sortinghatclassifier_import():
	with SortingHatClassifier() as cl:
		assert(cl.__class__.__name__ == 'SortingHatClassifier')

#----------------------------------------------------------------------
if __name__ == '__main__':
	test_sortinghatclassifier_import()

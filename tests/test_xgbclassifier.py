#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of XGBClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import conftest # noqa: F401
from starclass import XGBClassifier

#--------------------------------------------------------------------------------------------------
def test_xgbclassifier_import():
	with XGBClassifier() as cl:
		assert(cl.__class__.__name__ == 'XGBClassifier')

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

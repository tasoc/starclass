#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of MetaClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import conftest # noqa: F401
from starclass import MetaClassifier

#--------------------------------------------------------------------------------------------------
def test_metaclassifier_import():
	with MetaClassifier() as cl:
		assert(cl.__class__.__name__ == 'MetaClassifier')

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

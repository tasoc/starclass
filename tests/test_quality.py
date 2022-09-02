#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of starclass.quality.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import numpy as np
import conftest # noqa: F401
from starclass.quality import CorrectorQualityFlags, TESSQualityFlags

#--------------------------------------------------------------------------------------------------
def test_CorrectorQuality():
	"""Test of CorrectorQualityFlags"""

	# Quality flags:
	quality = np.zeros(10, dtype='int32')
	print(quality)

	# When quality is all zeros, filter should give all True:
	assert all(CorrectorQualityFlags.filter(quality))

	# Assign a random flag to one of the qualities and see if it filters:
	quality[3] = CorrectorQualityFlags.ManualExclude | CorrectorQualityFlags.JumpMultiplicativeLinear
	indx = CorrectorQualityFlags.filter(quality, CorrectorQualityFlags.ManualExclude)
	assert not indx[3]
	assert sum(indx) == len(quality) - 1

	# Test binary representation:
	rep = CorrectorQualityFlags.binary_repr(quality)
	print(rep)
	assert len(rep) == len(quality)
	assert len(rep[3]) == 32
	assert rep[3] == '00000000000000000000000001000010'
	assert rep[0] == '00000000000000000000000000000000'

	rep = CorrectorQualityFlags.binary_repr(2051)
	print(rep)
	assert rep == '00000000000000000000100000000011'

	# Test decoding to strings:
	dc = CorrectorQualityFlags.decode(quality[3])
	print(dc)
	assert len(dc) == 2
	assert 'Manual exclude' in dc

	# A default bitmask should be set:
	dmask = CorrectorQualityFlags.DEFAULT_BITMASK
	print(dmask)

	# Test HARDEST_BITMASK:
	hrep = CorrectorQualityFlags.binary_repr(CorrectorQualityFlags.HARDEST_BITMASK)
	assert hrep == '11111111111111111111111111111111'

	#for flag in CorrectorQualityFlags:
	#	print(flag)
	#	rep = CorrectorQualityFlags.binary_repr(quality)
	#	assert len(rep) == 32

#--------------------------------------------------------------------------------------------------
def test_TESSQuality():
	"""Test of TESSQualityFlags"""

	# Quality flags:
	quality = np.zeros(10, dtype='int32')
	print(quality)

	# When quality is all zeros, filter should give all True:
	assert all(TESSQualityFlags.filter(quality))

	# Assign a random flag to one of the qualities and see if it filters:
	quality[3] = TESSQualityFlags.ManualExclude | TESSQualityFlags.CoarsePoint
	indx = TESSQualityFlags.filter(quality, TESSQualityFlags.ManualExclude)
	assert not indx[3]
	assert sum(indx) == len(quality) - 1

	# Test binary representation:
	rep = TESSQualityFlags.binary_repr(quality)
	print(rep)
	assert len(rep) == len(quality)
	assert len(rep[3]) == 32
	assert rep[3] == '00000000000000000000000010000100'
	assert rep[0] == '00000000000000000000000000000000'

	rep = TESSQualityFlags.binary_repr(2051)
	print(rep)
	assert rep == '00000000000000000000100000000011'

	# Test decoding to strings:
	dc = TESSQualityFlags.decode(quality[3])
	print(dc)
	assert len(dc) == 2
	assert 'Manual exclude' in dc

	# A default bitmask should be set:
	dmask = TESSQualityFlags.DEFAULT_BITMASK
	print(dmask)

	# Test HARDEST_BITMASK:
	hrep = TESSQualityFlags.binary_repr(TESSQualityFlags.HARDEST_BITMASK)
	assert hrep == '11111111111111111111111111111111'

	#for flag in TESSQualityFlags:
	#	print(flag)
	#	indx = TESSQualityFlags.filter(quality, flag)
	#	rep = CorrectorQualityFlags.binary_repr(quality)
	#	assert len(rep) == 32

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

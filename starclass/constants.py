#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants used throughout package.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

#--------------------------------------------------------------------------------------------------
#: List of available classifiers.
classifier_list = ('rfgc', 'slosh', 'xgb', 'sortinghat', 'meta')

#--------------------------------------------------------------------------------------------------
#: List of available training sets.
trainingset_list = (
	'keplerq9v3',
	'keplerq9v3-instr',
	'keplerq9v3-long',
	'keplerq9v3-long-instr',
	'keplerq9v2',
	'keplerq9',
	'tdasim',
	'tdasim-raw',
	'tdasim-clean',
	'testing' # Special trainingset only used for testing
)

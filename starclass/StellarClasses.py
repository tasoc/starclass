#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enum of all the possible stellar classes.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import enum

__docformat__ = 'restructuredtext'

class StellarClasses(enum.Enum):
	"""
	Enum of all the possible stellar classes.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	SOLARLIKE = 'solar'
	TRANSECLIPSE = 'eclipse'
	RRLYRCEPH = 'rrl/ceph'
	GDOR = 'gdor'
	DSCT = 'dsct'
	RAPID = 'rapid'
	TRANSIENT = 'transient'
	CONTACT_ROT = 'contact/rot'
	APERIODIC = 'aperiodic'
	CONSTANT = 'constant'
	

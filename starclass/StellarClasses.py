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

	# General classes (Level 1)
	SOLARLIKE = 'solar'
	ECLIPSE = 'transit/eclipse'
	RRLYR_CEPHEID = 'RRLyr/Ceph'
	DSCT_BCEP = 'dSct/bCep'
	GDOR_SPB = 'gDor/spB'
	TRANSIENT = 'transient'
	CONTACT_ROT = 'contactEB/spots'
	APERIODIC = 'aperiodic'
	CONSTANT = 'constant'
	RAPID = 'rapid'

	# Level 2 classes
	RRLYR = 'RR Lyrae'
	CEPHEID = 'Cepheid'
	GDOR = 'gamma Doradus'
	SPB = 'Slowly pulsating B star'
	DSCT = 'delta Scuti'
	BCEP = 'beta Cepheid'
	LPV = 'Long period variable'
	SPOTS = 'Spot modulation'
	ROAP = 'roAp'
	SDB = 'sdB'
	WD = 'White Dwarf'

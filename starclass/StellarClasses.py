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

	TRANSIT = 'transit'

	RRLYR = 'RR Lyrae'
	CEPHEID = 'Cepheid'

	DSCT = 'delta Scuti'
	BCEP = 'beta Cepheid'

	GDOR = 'gamma Doradus'
	SPB = 'Slowly pulsating B star'

	# Rapid pulsators:
	ROAP = 'roAp'
	SDB = 'sdB'
	WD = 'White Dwarf'

	TRANSIENT = 'transient'

	SPOTS = 'Spot modulation'

	LPV = 'Long period variable'

	CONSTANT = 'constant'

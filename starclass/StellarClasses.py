#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enum of all the possible stellar classes.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import enum

__docformat__ = 'restructuredtext'

#--------------------------------------------------------------------------------------------------
class StellarClassesLevel1(enum.Enum):
	"""
	Enum of all the possible Level-1 stellar classes.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	# General classes (Level 1)
	SOLARLIKE = 'solar'
	ECLIPSE = 'transit/eclipse'
	RRLYR_CEPHEID = 'RRLyr/Ceph'
	DSCT_BCEP = 'dSct/bCep'
	GDOR_SPB = 'gDor/spB'
	#TRANSIENT = 'transient'
	CONTACT_ROT = 'contactEB/spots'
	APERIODIC = 'aperiodic'
	CONSTANT = 'constant'
	#RAPID = 'rapid'

#--------------------------------------------------------------------------------------------------
class StellarClassesLevel1Instr(enum.Enum):
	"""
	Enum of all the possible Level-1 stellar classes, including
	additional instrumental class.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	# General classes (Level 1) with additional instrumental class
	SOLARLIKE = 'solar'
	ECLIPSE = 'transit/eclipse'
	RRLYR_CEPHEID = 'RRLyr/Ceph'
	DSCT_BCEP = 'dSct/bCep'
	GDOR_SPB = 'gDor/spB'
	#TRANSIENT = 'transient'
	CONTACT_ROT = 'contactEB/spots'
	APERIODIC = 'aperiodic'
	CONSTANT = 'constant'
	#RAPID = 'rapid'
	INSTRUMENT = 'instrumental'

#--------------------------------------------------------------------------------------------------
class StellarClassesLevel2(enum.Enum):
	"""
	Enum of all the possible Level-2 stellar classes.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

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
	FLARE = 'Flare'

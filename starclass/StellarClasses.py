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

    #General classes
    SOLARLIKE = 'solar'
    ECLIPSE = 'trans/eclipse'
    RRLYR_CEPHEID = 'RRLyr/Ceph'
    DSCT_BCEP = 'dSct/bCep'
    GDOR_SPB = 'gDor/spB'
    TRANSIENT = 'transient'
    CONTACT_ROT = 'contactEB/spots'
    APERIODIC = 'aperiodic'
    CONSTANT = 'constant'

    #if we do them:
    RAPID = 'rapid'

    #Level 2 classes
    LPV = 'Long period variable'
    SPOTS = 'Spot modulation'

    GDOR = 'gamma Doradus'
    SPB = 'Slowly pulsating B star'

    DSCT = 'delta Scuti'
    BCEP = 'beta Cepheid'

    RRLYR = 'RR Lyrae'
    CEPHEID = 'Cepheid'

    # Rapid pulsators:
    ROAP = 'roAp'
    SDB = 'sdB'
    WD = 'White Dwarf'
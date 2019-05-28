#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import six.moves.cPickle as pickle
import gzip

PICKLE_DEFAULT_PROTOCOL = 4 #: Default protocol to use for saving pickle files.

def savePickle(fname, obj):
	"""
	Save an object to file using pickle.

	Parameters:
		fname (string): File name to save to. If the name ends in '.gz' the file
			will be automatically gzipped.
		obj (object): Any pickalble object to be saved to file.
	"""

	if fname.endswith('.gz'):
		o = gzip.open
	else:
		o = open

	with o(fname, 'wb') as fid:
		pickle.dump(obj, fid, protocol=PICKLE_DEFAULT_PROTOCOL)


def loadPickle(fname):
	"""
	Load an object from file using pickle.

	Parameters:
		fname (string): File name to save to. If the name ends in '.gz' the file
			will be automatically unzipped.
		obj (object): Any pickalble object to be saved to file.

	Returns:
		object: The unpickled object from the file.
	"""

	if fname.endswith('.gz'):
		o = gzip.open
	else:
		o = open

	with o(fname, 'rb') as fid:
		return pickle.load(fid)

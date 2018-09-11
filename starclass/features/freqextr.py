#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
from six.moves import range
import numpy as np
import tempfile
import subprocess
import logging
import os.path

def freqextr(lightcurve, numfreq=6, hifac=1, ofac=4):
	"""
	Extract the highest amplitude frequencies from the timeseries.

	Parameters:
		lightcurve (): Lightcurve to extract frequencies for.
		numfreq (integer, optional): Number of frequencies to extract.
		hifac (integer, optional): Nyquist factor.
		ofac (integer, optional): Oversampling factor used for initial search for peaks in power spectrum.

	Returns:
		dict: Features

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# Normalize the lightcurve and convert to ppm:
	lc = (lightcurve.remove_nans().normalize() - 1.0) * 1e6

	# Create temp file to hold the timeseries:
	with tempfile.NamedTemporaryFile(prefix='slsclean_', mode='w') as fid:
		fname = fid.name

		# Write timeseries to temp file:
		for i in range(len(lc.time)):
			fid.write("%.16e %.16e\n" % (lc.time[i], lc.flux[i]))

		# Construct command to be issued, calling the SLSCLEAN program:
		cmd = 'slsclean "{inp:s}" -hifac {hifac:d} -ofac {ofac:d} -nmax {numfreq:d}'.format(
			inp=fname,
			hifac=hifac,
			ofac=ofac,
			numfreq=numfreq
		)
		logger.debug("Running command: %s", cmd)

		# Call the SLSCLEAN program in a subprocess:
		p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
		ret = p.communicate()
		logger.debug(ret[0])
		if p.returncode != 0:
			raise Exception(ret)
		elif ret[1] != '':
			raise Exception(ret[1])

		# Load results from the output file:
		freqs, amps, phases = np.loadtxt(fname + '.slscleanlog', comments='#', usecols=(1, 3, 5), unpack=True, ndmin=2)

		# Make sure to clean up after ourselves:
		os.remove(fname + '.slscleanlog')
		os.remove(fname + '.slsclean')

	# Restructure lists into dictionary of features:
	features = {}
	for k in range(numfreq):
		if k >= len(freqs):
			features['freq' + str(k+1)] = np.nan
			features['amp' + str(k+1)] = np.nan
			features['phase' + str(k+1)] = np.nan
		else:
			features['freq' + str(k+1)] = freqs[k]
			features['amp' + str(k+1)] = amps[k]
			features['phase' + str(k+1)] = phases[k]

	return features

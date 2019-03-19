#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
from six.moves import range
from tempfile import mkdtemp
import numpy as np
import subprocess
import logging
import os.path
import warnings
import shutil

def freqextr(lightcurve, numfreq=6, hifac=1, ofac=4):
	"""
	Extract the highest amplitude frequencies from the timeseries.

	Parameters:
		lightcurve (``lightkurve.LightCurve`` object): Lightcurve to extract frequencies for.
		numfreq (integer, optional): Number of frequencies to extract.
		hifac (integer, optional): Nyquist factor.
		ofac (integer, optional): Oversampling factor used for initial search for peaks in power spectrum.

	Returns:
		dict: Features

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# Normalize the lightcurve and convert to ppm:
	#lc = (lightcurve.remove_nans().normalize() - 1.0) * 1e6
	lc = lightcurve

	# Create temp directory to hold the timeseries:
	# We should really use tempfile.TemporaryDirectory instead, but it seems
	# to have some issues with sometimes not being able to clean up after
	# itself on some operating systems. Therefore we are using our own
	# version here which attempts at doing a more robust cleanup
	tmpdir = mkdtemp(prefix='slsclean_')
	try:
		# Name of timeseries file:
		fname = os.path.join(tmpdir, 'tmp_timeseries')

		# Write timeseries to temp file:
		with open(fname, 'w') as fid:
			for i in range(len(lc.time)):
				fid.write("%.16e %.16e\n" % (lc.time[i], lc.flux[i]))

		# Construct command to be issued, calling the SLSCLEAN program:
		cmd = 'slsclean "{inp:s}" -hifac {hifac:d} -ofac {ofac:d} -nmax {numfreq:d} -nots'.format(
			inp=fname,
			hifac=hifac,
			ofac=ofac,
			numfreq=numfreq
		)
		logger.debug("Running command: %s", cmd)

		# Call the SLSCLEAN program in a subprocess:
		p = subprocess.Popen(cmd, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
		ret = p.communicate()
		logger.debug(ret[0])
		if p.returncode != 0:
			raise Exception(ret)
		elif ret[1] != '':
			raise Exception(ret[1])

		# Load results from the output file:
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore', message='loadtxt: Empty input file: ', category=UserWarning)
			data = np.loadtxt(fname + '.slscleanlog', comments='#', usecols=(1, 3, 5), unpack=True, ndmin=2).reshape(-1, 3)

	# Once we are done, try to cleanup the temp directory
	# Sometimes this fails so we allow it to try again a few times
	finally:
		for retries in range(5):
			try:
				shutil.rmtree(tmpdir)
				break
			except:
				logger.warning("Couldnt delete temp dir on %d try", retries+1)

	# Restructure lists into dictionary of features:
	features = {}
	for k in range(numfreq):
		if k >= data.shape[0]:
			features['freq' + str(k+1)] = np.nan
			features['amp' + str(k+1)] = np.nan
			features['phase' + str(k+1)] = np.nan
		else:
			features['freq' + str(k+1)] = data[k, 0]
			features['amp' + str(k+1)] = data[k, 1]
			features['phase' + str(k+1)] = data[k, 2]

	return features

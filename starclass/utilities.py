#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import pickle
import gzip
from bottleneck import nanmedian, nanmean, allnan
from scipy.stats import binned_statistic
import astropy.units as u

# Constants:
mad_to_sigma = 1.482602218505602 #: Conversion constant from MAD to Sigma. Constant is 1/norm.ppf(3/4)

PICKLE_DEFAULT_PROTOCOL = 4 #: Default protocol to use for saving pickle files.

#--------------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------
def rms_timescale(lc, timescale=3600/86400):
	"""
	Compute robust RMS on specified timescale. Using MAD scaled to RMS.

	Parameters:
		lc (``lightkurve.TessLightCurve`` object): Timeseries to calculate RMS for.
		timescale (float, optional): Timescale to bin timeseries before calculating RMS. Default=1 hour.

	Returns:
		float: Robust RMS on specified timescale.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	time = np.asarray(lc.time)
	flux = np.asarray(lc.flux)
	if len(flux) == 0 or allnan(flux):
		return np.nan
	if len(time) == 0 or allnan(time):
		raise ValueError("Invalid time-vector specified. No valid timestamps.")

	time_min = np.nanmin(time)
	time_max = np.nanmax(time)
	if not np.isfinite(time_min) or not np.isfinite(time_max) or time_max - time_min <= 0:
		raise ValueError("Invalid time-vector specified")

	# Construct the bin edges seperated by the timescale:
	bins = np.arange(time_min, time_max, timescale)
	bins = np.append(bins, time_max)

	# Bin the timeseries to one hour:
	indx = np.isfinite(flux)
	flux_bin, _, _ = binned_statistic(time[indx], flux[indx], nanmean, bins=bins)

	# Compute robust RMS value (MAD scaled to RMS)
	return mad_to_sigma * nanmedian(np.abs(flux_bin - nanmedian(flux_bin)))

#--------------------------------------------------------------------------------------------------
def get_periods(featdict, nfreqs, time, in_days=True):
	"""
		Cuts frequency data down to desired number of frequencies (in umHz) and optionally transforms them into periods in days.
	"""
	# convert to c/d (1*u.uHz).to(u.cycle/u.d, equivalencies=[(u.cycle/u.s, u.Hz)]).to(u.d))
	#periods = featdict['frequencies'].loc['harmonic',0].loc['num',:nfreqs]['frequency']

	tab = featdict['frequencies']
	periods = tab[tab['harmonic'] == 0][:nfreqs]['frequency'].quantity
	if in_days:
		periods = (1./(periods)).to(u.day)

	is_nan = np.isnan(periods)
	periods[np.where(is_nan)] = np.max(time)-np.min(time)
	n_usedfreqs = nfreqs - np.sum(is_nan)

	return periods, n_usedfreqs

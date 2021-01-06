#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from bottleneck import nanmedian, nanmean, allnan
from scipy.stats import binned_statistic
import astropy.units as u

# Constants:
mad_to_sigma = 1.482602218505602 #: Conversion constant from MAD to Sigma. Constant is 1/norm.ppf(3/4)

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
def get_periods(featdict, nfreqs, time, in_days=True, ignore_harmonics=False):
	"""
	Cuts frequency data down to desired number of frequencies (in uHz) and optionally
	transforms them into periods in days.

	Parameters:
		featdict (dict):
		nfreq (int): Number of frequencies/periods to extract
		time (ndarray):
		in_days (bool, optional): Return periods in days instead of frequencies in uHz.
		ignore_harmonics (bool, optional): Sort frequency table by amplitude (i.e. ignore into
			harmonic structure).

	Returns:
		tuple:
			- periods:
			- n_usedfreqs (int): Number of true periods/frequencies that are used.
			- usedfreqs: Indices of the used periods/frequencies in the astropy table.

	.. codeauthor:: Jeroen Audenaert <jeroen.audenaert@kuleuven.be>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	tab = featdict['frequencies']
	tab = tab[~np.isnan(tab['amplitude'])]
	if ignore_harmonics:
		tab.sort('amplitude', reverse=True)
		selection = tab[:min(len(tab), nfreqs)]
	else:
		selection = tab[tab['harmonic'] == 0][:nfreqs]

	periods = selection['frequency'].quantity
	usedfreqs = selection[['num', 'harmonic']]

	if in_days:
		periods = (1/periods).to(u.day)

	per = (np.max(time) - np.min(time)) * u.day
	gap = nfreqs - len(periods)
	if gap > 0:
		if in_days:
			for i in range(gap):
				periods = periods.insert(len(periods), per)
		else:
			for i in range(gap):
				periods = periods.insert(len(periods), (1/per).to(u.uHz))

	n_usedfreqs = len(usedfreqs)

	return periods.value, n_usedfreqs, usedfreqs

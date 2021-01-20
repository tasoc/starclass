#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract frequencies from timeseries.

.. codeauthor:: Kristine Kousholt Mikkelsen <201505068@post.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import logging
import itertools
from bottleneck import median
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.table import Table
from .powerspectrum import powerspectrum

#--------------------------------------------------------------------------------------------------
# TODO: Replace with ps.ls.model?
def model(x, a, b, freq):
	omegax = 0.1728 * np.pi * freq * x # Strange factor is 2 * 86400 * 1e-6
	return a * np.sin(omegax) + b * np.cos(omegax)

#--------------------------------------------------------------------------------------------------
def freqextr(lightcurve, n_peaks=6, n_harmonics=0, hifac=1, ofac=4, snrlim=None, snr_width=None,
	faplim=1-0.9973, devlim=0.5, conseclim=10, harmonics_list=None, Noptimize=10, optim_max_diff=10,
	initps=None):
	r"""
	Extract frequencies from timeseries.

	The program will perform iterative sine-wave fitting (CLEAN or pre-whitening) using
	a sum of harmonic functions of the following form:

	.. math::
		\sum_{i=1}^{N_\mathrm{peaks}} A_i \sin(2\pi\nu_i t + \delta_i)
		= \sum_{i=1}^{N_\mathrm{peaks}} \alpha_i\sin(2\pi\nu_i t) + \beta_i\cos(2\pi\nu_i t) \, ,

	where :math:`\nu_i`, :math:`A_i` and :math:`\delta_i` denoted the frequency, amplitude
	and phase of the oscillation.

	If ``n_harmonic`` is greater than zero, the routine will additionally for each extracted peak
	extract peaks at the given number of harmonics for each peak. Default is to :math:`2\nu_i`,
	:math:`3\nu_i` etc., but this can be controlled by the ``harmonics_list`` input.

	At each iteration, an optimization loop is entered which will go back and re-optimize the
	previously found peaks in an attempt at minimizing influences between close frequencies.
	The details of this optimization can be controlled by the parameters ``Noptimize`` and
	``optim_max_diff``.

	Parameters:
		lightcurve (:class:`lightkurve.LightCurve`): Lightcurve to extract frequencies for.
		n_peaks (int, optional): Number of frequencies to extract.
		n_harmonics (int, optional): Number of harmonics to extract for each frequency.
		hifac (float, optional): Nyquist factor.
		ofac (int, optional): Oversampling factor used for initial search for peaks
			in power spectrum.
		snrlim (float, optional): Limit on local signal-to-noise ratio above which peaks are
			considered significant. If set to `None` no limit is enforced. Default is to not
			enforce a limit.
		snr_width (float, optional): Width in uHz around each peak to estimate signal-to-noise from.
			Default is 15 frequency steps on either side of the peak.
		faplim (float, optional): False alarm probability limit. Peaks with a f.a.p. below this
			limit are considerd significant. If set to `None` no limit is enforced.
			Default is 1-0.9973=0.0027.
		devlim (float, optional): If set to `None` no limit is enforced. Default is 50%.
		conseclim (int, optional): Stop after this number of consecutive failed peaks.
			Default is 10.
		Noptimize (int, optional): At each iteration re-optimize. If put to -1, all peaks will
			be optimized at each iteration. Default is 10.
		optim_max_diff (float, optional): Maximal difference in uHz between frequencies to be
			optimized. Any frequencies futher away than this value from the extracted peak
			will not be optimized in that iteration. If set to ``None`` no limit is enforced.
			Default is 10 uHz. Please note that this does not take the spectral windowfunction
			into account, so this value may have to	be increased in cases where the windowfunction
			has significant side-lobes.
		initps (:class:`powerspectrum`, optional): Initial powerspectrum. Should be a powerspectrum
			calculated from the provided lightcurve. This can be provided if the powerspectrum
			has already been calculated. If not provided, it is calculated from the provided
			lightcurve.

	Returns:
		:class:`astropy.table.Table`: Table of extracted oscillations.

	Note:
		If the height of the peak of one of the harmonics are close to being insignificant,
		the harmonic may not be found as an harmonic, but will be found later as a peak in it self.

	.. codeauthor:: Kristine Kousholt Mikkelsen <201505068@post.au.dk>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# Default value for different parameters
	# TODO: Add these as inputs
	estimate_noise = True

	if initps is not None and not isinstance(initps, powerspectrum):
		raise ValueError("Initial powerspectrum is invalid")

	if Noptimize is None:
		Noptimize = 0

	# If no list of harmonics is given, do the simple one:
	if harmonics_list is None:
		harmonics_list = np.arange(2, n_harmonics+2)
	elif len(harmonics_list) < n_harmonics:
		raise ValueError("List of harmonics is too short")

	# Constants:
	power_median_to_mean = (1 - 1/9)**-3
	mean_noise = 1

	# Store original lightcurve and powerspectrum for later use:
	original_lightcurve = lightcurve.copy()
	if initps is None:
		original_ps = powerspectrum(original_lightcurve)
	else:
		original_ps = initps
	f_max = original_ps.nyquist*hifac*1e6
	df = original_ps.df*1e6

	# Defaults that depend on the power spectrum parameters:
	if snr_width is None:
		snr_width = 15*df

	# Create lists for frequencies, alpha, beta and deviations:
	# Create as 2D array, which as main-frequency number for rows, and harmonic number for columns.
	nu = np.full((n_peaks, n_harmonics+1), np.nan)
	alpha = np.full((n_peaks, n_harmonics+1), np.nan)
	beta = np.full((n_peaks, n_harmonics+1), np.nan)
	deviation = np.full((n_peaks, n_harmonics+1), np.nan)

	# The first powerspectrum has already been calculated:
	ps = original_ps.copy()

	for i in range(n_peaks):
		logger.debug("-"*72)

		# Calculate the powerspectrum and find the index of the largest power value
		if i > 0:
			ps = powerspectrum(lightcurve)
		frequency, power = ps.powerspectrum(oversampling=ofac, nyquist_factor=hifac, scale='power')

		# Estimate a frequency-dependent noise-floor by binning the power spectrum.
		if estimate_noise:
			# Create bins to estimate noise level in:
			#bins = np.logspace(np.floor(np.log10(df)), np.ceil(np.log10(f_max)), 20)
			bins = np.linspace(df, f_max, 20)

			# Calculate the median in the bins.
			# Make sure we have at least 20 frequencies in each bin,
			# otherwise combine adjacent bins until this is the case:
			for _ in range(100):
				mean_noise, bins, binindx = binned_statistic(frequency, power, bins=bins, statistic=median)

				redo = False
				for k, num in enumerate(np.bincount(binindx)):
					if num < 20:
						bins = np.delete(bins, k)
						redo = True
						break

				if not redo:
					break

			bins = bins[:-1] + 0.5*(bins[1:] - bins[:-1])

			indx = np.isfinite(mean_noise)
			if np.sum(indx) > 2:
				mean_noise_func = interp1d(bins[indx], mean_noise[indx], kind='linear', fill_value='extrapolate', assume_sorted=True)
				mean_noise = power_median_to_mean * mean_noise_func(frequency)
				mean_noise = np.clip(mean_noise, 0, None)
				mean_noise += 1 # Add one to avoid DivideByZero errors - only used for finding max
			else:
				mean_noise = 1

			#plt.figure()
			#plt.plot(frequency, np.sqrt(power/mean_noise), 'k-', lw=0.5)
			#plt.plot(frequency, power, 'b')
			#plt.plot(frequency, mean_noise,'k-')
			#plt.title(i)
			#plt.show()

		# Finds the frequency of the largest peak:
		pmax_index = np.argmax(power / mean_noise)
		fsearch = frequency[pmax_index]
		if pmax_index > 0 and pmax_index < len(power)-1:
			fsearch = [frequency[pmax_index-1], fsearch, frequency[pmax_index+1]]
		nu[i,0] = ps.optimize_peak(fsearch)
		alpha[i,0], beta[i,0] = ps.alpha_beta(nu[i,0])
		logger.debug('Fundamental frequency: %f', nu[i,0])

		# Stop if significance becomes too low (lombscargle significance)
		if faplim is not None:
			FAP = ps.false_alarm_probability(nu[i,0])
			if FAP > faplim:
				logger.debug("Stopped from FAP")
				nu[i,0] = np.nan
				alpha[i,0] = np.nan
				beta[i,0] = np.nan
				break

		# Stop if significance becomes too low (SNR ratio)
		if snrlim is not None:
			# Calculate SNR by estimating noise level locally around peak:
			# TODO: Subtract peak first?
			noise = np.sqrt(power_median_to_mean * median(power[(frequency > (nu[i,0] - snr_width)) & (frequency < (nu[i,0] + snr_width))]))
			amp = np.sqrt(alpha[i,0]**2 + beta[i,0]**2)
			snr = amp / noise
			logger.debug("SNR: %f", snr)

			#plt.figure()
			#plt.plot(frequency, np.sqrt(power), 'k-', lw=0.5)
			#plt.plot(frequency, np.sqrt(mean_noise), 'r-', lw=0.5)
			#plt.plot(frequency[pmax_index], np.sqrt(power[pmax_index]), 'go')
			#plt.plot(nu[i,0], ps.powerspectrum(nu[i,0]*1e-6, scale='amplitude')[1], 'ro')
			#plt.axhline(noise)
			#plt.axvline(nu[i,0] - snr_width)
			#plt.axvline(nu[i,0] + snr_width)

			if snr < snrlim:
				logger.debug("Stopped from SNR")
				nu[i,0] = np.nan
				alpha[i,0] = np.nan
				beta[i,0] = np.nan
				break

		# Check how the extracted peak compares with the original powerspectrum
		if devlim is not None:
			atemp, btemp = original_ps.alpha_beta(nu[i,0])
			deviation[i,0] = (alpha[i,0]**2 + beta[i,0]**2) / (atemp**2 + btemp**2)

		# Stops if there are to many consecutive failed peaks
		if devlim is not None and conseclim is not None:
			# Stop numpy from warning us that deviation contains NaN
			with np.errstate(invalid='ignore'):
				deviation_large = (deviation > 1/devlim) | (deviation < devlim)
			if np.all( deviation_large[max(i-conseclim, 0):(i+1), 0] ): # Only checking main peaks right now!
				logger.debug('Stopped due to too many consecutive failed peaks')
				break

		# Removes the largest peak from the data:
		lightcurve -= model(lightcurve.time, alpha[i,0], beta[i,0], nu[i,0])

		# Loop through all harmonics:
		for h in range(1, n_harmonics+1):
			n_harmonic = harmonics_list[h-1]
			# Don't find harmonics outside frequency range:
			if n_harmonic*nu[i,0] > f_max:
				break

			# Updates the flux and optimize to find the correct frequency
			ps = powerspectrum(lightcurve)

			# Checks the significance of the harmonics. If it is too low NaN is returned in amplitude, frequency and phase for the given harmonic
			nu[i,h] = ps.optimize_peak(n_harmonic*nu[i,0])

			# Stop if significance becomes too low (lombscargle significance)
			if faplim is not None:
				FAP = ps.false_alarm_probability(nu[i,h])
				logger.debug('harmonic %d: %f %f', h, nu[i,h], FAP)
				if FAP > faplim:
					logger.debug("Harmonic rejected from FAP")
					nu[i,h] = np.nan
					alpha[i,h] = np.nan
					beta[i,h] = np.nan
					continue

			# Stop if significance becomes too low (SNR ratio):
			if snrlim is not None:
				# Calculate SNR by estimating noise level locally around peak:
				# TODO: Subtract peak first?
				noise = np.sqrt(power_median_to_mean * median(power[(frequency > (nu[i,0] - snr_width)) & (frequency < (nu[i,0] + snr_width))]))
				amp = np.sqrt(alpha[i,0]**2 + beta[i,0]**2)
				snr = amp / noise
				logger.debug("SNR: %f", snr)

				#plt.figure()
				#plt.plot(frequency, np.sqrt(power), 'k-', lw=0.5)
				#plt.plot(frequency, np.sqrt(mean_noise), 'r-', lw=0.5)
				#plt.plot(frequency[pmax_index], np.sqrt(power[pmax_index]), 'go')
				#plt.plot(nu[i,0], ps.powerspectrum(nu[i,0]*1e-6, scale='amplitude')[1], 'ro')
				#plt.axhline(noise)
				#plt.axvline(nu[i,0] - snr_width)
				#plt.axvline(nu[i,0] + snr_width)

				if snr < snrlim:
					logger.debug("Stopped from SNR")
					nu[i,0] = np.nan
					alpha[i,0] = np.nan
					beta[i,0] = np.nan
					break

			# Removes the harmonic peak from the data:
			alpha[i,h], beta[i,h] = ps.alpha_beta(nu[i,h])
			lightcurve -= model(lightcurve.time, alpha[i,h], beta[i,h], nu[i,h])

			# Check how the extracted peak compares with the original powerspectrum and stops
			# if there are to many consecutive failed peaks
			if devlim is not None:
				atemp, btemp = original_ps.alpha_beta(nu[i,h])
				deviation[i,h] = (alpha[i,h]**2 + beta[i,h]**2) / (atemp**2 + btemp**2)

		# Optimize the Noptimize nearest peaks
		if i != 0 and Noptimize != 0:
			for h in range(n_harmonics+1):

				# Sort to find nearest frequencies to optimize
				Nopt = Noptimize + 1
				if (i+1)*(n_harmonics+1) < Nopt or Noptimize == -1:
					Nopt = (i+1)*(n_harmonics+1)

				nusort = np.abs(nu - nu[i,h])
				nusort = nusort.ravel()
				order = np.argsort(nusort) # sort nusort and find the list of indexes

				# Create an index of which peaks should be optimized:
				indx_optim = np.zeros_like(order, dtype='bool')
				indx_optim[1:Nopt] = True

				# Only optimize a peak if it is closer than the set limit.
				# NOTE: Be careful as this doesn't take the window function into account
				if optim_max_diff is not None:
					with np.errstate(invalid='ignore'):
						indx_optim &= (nusort[order] < optim_max_diff)

				# Pick out the peaks that should be optimized:
				order = order[indx_optim]
				order = list(zip(*np.unravel_index(order, (n_peaks, n_harmonics+1))))
				logger.debug("Optimizing %d peaks", len(order))

				for j in order:
					if np.isfinite(alpha[j]): # and deviation[j] < 1/devlim and deviation[j] > devlim:
						# Add the oscillation:
						lightcurve += model(lightcurve.time, alpha[j], beta[j], nu[j])
						ps = powerspectrum(lightcurve)

						# Find the frequency of maximum power and find alpha and beta again
						nu[j] = ps.optimize_peak(nu[j])
						alpha[j], beta[j] = ps.alpha_beta(nu[j])

						# Recalculate the deviation
						if devlim is not None:
							atemp, btemp = original_ps.alpha_beta(nu[j])
							deviation[j] = (alpha[j]**2 + beta[j]**2)/(atemp**2 + btemp**2)

						# Remove the oscillation again:
						lightcurve -= model(lightcurve.time, alpha[j], beta[j], nu[j])

	# Remove anything that in the end was marked with a large deviation:
	if devlim is not None:
		for i in range(n_peaks):
			if deviation[i,0] > 1/devlim or deviation[i,0] < devlim:
				# If main peak is rejected, then also reject all harmonics
				nu[i,:] = np.nan
				alpha[i,:] = np.nan
				beta[i,:] = np.nan
			else:
				for j in range(1, n_harmonics+1):
					if deviation[i,j] > 1/devlim or deviation[i,j] < devlim:
						nu[i,j] = np.nan
						alpha[i,j] = np.nan
						beta[i,j] = np.nan

	# Calculate amplitude and phase from alpha and beta:
	amp = np.sqrt(alpha**2 + beta**2)
	phase = np.arctan2(beta, alpha)

	# Make sure the found peaks are ordered by the amplitude of the main peak:
	amp[np.isnan(amp)] = -np.inf
	indx = np.argsort(amp[:,0])[::-1]
	nu = nu[indx,:]
	amp = amp[indx,:]
	phase = phase[indx,:]
	alpha = alpha[indx,:]
	beta = beta[indx,:]
	deviation = deviation[indx,:]
	amp[~np.isfinite(amp)] = np.nan

	# Gather into table:
	num, harmonic = np.meshgrid(range(1, n_peaks+1), range(n_harmonics+1))
	tab = Table(
		data=[
			num.flatten(order='F'),
			harmonic.flatten(order='F'),
			nu.flatten(),
			amp.flatten(),
			phase.flatten(),
			alpha.flatten(),
			beta.flatten(),
			deviation.flatten()
		],
		names=['num', 'harmonic', 'frequency', 'amplitude', 'phase', 'alpha', 'beta', 'deviation'],
		dtype=['int32', 'int32', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64'])

	# Add units to columns:
	tab['frequency'].unit = u.uHz
	tab['amplitude'].unit = lightcurve.flux_unit
	tab['phase'].unit = u.rad
	tab['alpha'].unit = lightcurve.flux_unit
	tab['beta'].unit = lightcurve.flux_unit

	# Add index to peak number and harmonic for easy lookup:
	# TODO: Use table indicies - Problem with Pickle
	#tab.add_index('num')

	# Add meta data to table on how the list was created:
	tab.meta['n_peaks'] = n_peaks
	tab.meta['n_harmonics'] = n_harmonics
	tab.meta['harmonics_list'] = harmonics_list
	tab.meta['hifac'] = hifac
	tab.meta['ofac'] = ofac
	tab.meta['snrlim'] = snrlim
	tab.meta['snr_width'] = snr_width * u.uHz
	tab.meta['faplim'] = faplim
	tab.meta['devlim'] = devlim
	tab.meta['conseclim'] = conseclim

	return tab

#--------------------------------------------------------------------------------------------------
def freqextr_table_from_dict(feat, n_peaks=None, n_harmonics=None, flux_unit=None):
	"""
	Reconstruct freqextr table from features dict.

	Please note that not all information can be recreated from features dictionaries.
	The ``deviation`` column will be all NaN, since it can not be recreated, and
	only ``n_peaks`` and ``n_harmonics`` will be available in the meta-information.

	Parameters:
		feat (dict): Dictionary of features.
		n_peaks (int): If not provided, it will be determined from ``feat``.
		n_harmonics (int): If not provided, it will be determined from ``feat``.
		flux_unit (:class:`astropy.units.Unit`): Unit to put on the amplitude, alpha and beta columns.

	Returns:
		:class:`astropy.table.Table`:

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	if n_peaks is None:
		n_peaks = 0
		while True:
			n_peaks += 1
			if 'freq{0:d}'.format(n_peaks) not in feat:
				n_peaks -= 1
				break

	if n_harmonics is None:
		n_harmonics = 0
		while True:
			n_harmonics += 1
			if 'freq1_harmonic{0:d}'.format(n_harmonics) not in feat:
				n_harmonics -= 1
				break

	rows = []
	for num, harmonic in itertools.product(range(1, n_peaks+1), range(n_harmonics+1)):
		if harmonic == 0:
			key = '{0:d}'.format(num)
		else:
			key = '{0:d}_harmonic{1:d}'.format(num, harmonic)

		amp = feat.get('amp' + key, np.NaN)
		freq = feat.get('freq' + key, np.NaN)
		phase = feat.get('phase' + key, np.NaN)
		rows.append([
			num,
			harmonic,
			freq,
			amp,
			phase,
			amp*np.cos(phase),
			amp*np.sin(phase),
			np.NaN # There is no way to recover this information
		])

	tab = Table(
		rows=rows,
		names=['num', 'harmonic', 'frequency', 'amplitude', 'phase', 'alpha', 'beta', 'deviation'],
		dtype=['int32', 'int32', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64'])

	# Add units to columns:
	tab['frequency'].unit = u.uHz
	tab['amplitude'].unit = flux_unit
	tab['phase'].unit = u.rad
	tab['alpha'].unit = flux_unit
	tab['beta'].unit = flux_unit

	# Add meta data to table on how the list was created:
	tab.meta['n_peaks'] = n_peaks
	tab.meta['n_harmonics'] = n_harmonics
	#tab.meta['harmonics_list'] = harmonics_list
	#tab.meta['hifac'] = hifac
	#tab.meta['ofac'] = ofac
	#tab.meta['snrlim'] = snrlim
	#tab.meta['snr_width'] = snr_width * u.uHz
	#tab.meta['faplim'] = faplim
	#tab.meta['devlim'] = devlim
	#tab.meta['conseclim'] = conseclim

	return tab

#--------------------------------------------------------------------------------------------------
def freqextr_table_to_dict(tab):
	"""
	Convert freqextr output table to features dictionary.

	This will return a dictionary with ``freq``, ``amp``, and ``phase`` keys.
	Please not that this operation does not conserve all information from the table.

	Parameters:
		tab (:class:`astropy.table.Table`): Table extracted by :func:`freqextr`.

	Returns:
		dict: Dictionary with frequencies, amplitudes and phases.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	features = {}
	for row in tab:
		if row['harmonic'] == 0:
			key = '{0:d}'.format(row['num'])
		else:
			key = '{0:d}_harmonic{1:d}'.format(row['num'], row['harmonic'])

		features['freq' + key] = row['frequency']
		features['amp' + key] = row['amplitude']
		features['phase' + key] = row['phase']

	return features

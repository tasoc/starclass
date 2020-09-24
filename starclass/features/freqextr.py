#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract frequencies from timeseries.

.. codeauthor:: Kristine Kousholt Mikkelsen <201505068@post.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import logging
from .powerspectrum import powerspectrum
from bottleneck import nanmedian, move_median
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d

#--------------------------------------------------------------------------------------------------
def _move_median_central_1d(x, width_points):
	y = move_median(x, width_points, min_count=1)
	y = np.roll(y, -width_points//2+1)
	for k in range(width_points//2+1):
		y[k] = nanmedian(x[:(k+2)])
		y[-(k+1)] = nanmedian(x[-(k+2):])
	return y

#--------------------------------------------------------------------------------------------------
def move_median_central(x, width_points, axis=0):
	return np.apply_along_axis(_move_median_central_1d, axis, x, width_points)

#--------------------------------------------------------------------------------------------------
# TODO: Replace with ps.ls.model?
def model(x, a, b, freq):
	omegax = 0.1728 * np.pi * freq * x # Strange factor is 2 * 86400 * 1e-6
	return a * np.sin(omegax) + b * np.cos(omegax)

#--------------------------------------------------------------------------------------------------
def freqextr(lightcurve, n_peaks=6, n_harmonics=10, hifac=1, ofac=4, snrlim=None, snr_width=None,
	faplim=1-0.9973, conseclim=10, harmonics_list=None):
	"""
	Extract frequencies from timeseries.

	Parameters:
		lightcurve (:class:`lightkurve.LightCurve`): Lightcurve to extract frequencies for.
		n_peaks (int, optional): Number of frequencies to extract.
		n_harmonics (int, optional): Number of harmonics to extract for each frequency.
		hifac (int, optional): Nyquist factor.
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
		conseclim (int, optional): Stop after this number of consecutive failed peaks.
			Default is 10.

	Returns:
		dict: Features

	Note:
		If the height of the peak of one of the harmonics are close to being insignificant,
		the harmonic may not be found as an harmonic, but will be found later as a peak in it self.

	.. codeauthor:: Kristine Kousholt Mikkelsen <201505068@post.au.dk>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# Default value for different parameters
	# TODO: Add these as inputs
	alphadev = 0.50 # 0.75
	Noptimize = 10 # Optimize all peaks = -1
	optim_max_diff = 10 # uHz
	estimate_noise = True

	# If no list of harmonics is given, do the simple one:
	if harmonics_list is None:
		harmonics_list = np.arange(2, n_harmonics+2)

	# Constants:
	power_median_to_mean = (1 - 1/9)**-3
	mean_noise = 1

	# Store original lightcurve and powerspectrum for later use:
	original_lightcurve = lightcurve.copy()
	original_ps = powerspectrum(original_lightcurve)
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

	for i in range(n_peaks):
		logger.debug("-"*72)

		# Calculate the powerspectrum and find the index of the largest power value
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
				mean_noise, bins, binindx = binned_statistic(frequency, power, bins=bins, statistic='median')

				redo = False
				for u, num in enumerate(np.bincount(binindx)):
					if num < 20:
						bins = np.delete(bins, u)
						redo = True
						break

				if not redo: break

			bins = bins[:-1] + 0.5*(bins[1:] - bins[:-1])

			indx = np.isfinite(mean_noise)
			mean_noise_func = interp1d(bins[indx], mean_noise[indx], kind='linear', fill_value='extrapolate', assume_sorted=True)
			mean_noise = power_median_to_mean * mean_noise_func(frequency)
			mean_noise = np.clip(mean_noise, 0, None)

			#plt.figure()
			#plt.plot(frequency, np.sqrt(power/mean_noise), 'k-', lw=0.5)
			#plt.plot(frequency, power, 'b')
			#plt.plot(frequency, mean_noise,'k-')
			#plt.title(i)
			#plt.show()

		# Finds the frequency of the largest peak:
		pmax_index = np.argmax(power / mean_noise)
		nu[i,0] = ps.optimize_peak(frequency[pmax_index])
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
			noise = np.sqrt(power_median_to_mean * np.median(power[(frequency > (nu[i,0] - snr_width)) & (frequency < (nu[i,0] + snr_width))]))
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
		atemp, btemp = original_ps.alpha_beta(nu[i,0])
		deviation[i,0] = (alpha[i,0]**2 + beta[i,0]**2) / (atemp**2 + btemp**2)

		# Stops if there are to many consecutive failed peaks
		if conseclim is not None:
			# Stop numpy from warning us that deviation contains NaN
			with np.errstate(invalid='ignore'):
				deviation_large = (deviation > 1/alphadev) | (deviation < alphadev)
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
				noise = np.sqrt(power_median_to_mean * np.median(power[(frequency > (nu[i,0] - snr_width)) & (frequency < (nu[i,0] + snr_width))]))
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

			# Check how the extracted peak compares with the original powerspectrum and stops if there are to many consecutive failed peaks
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
					if deviation[j] < 1/alphadev and deviation[j] > alphadev and np.isfinite(alpha[j]):
						#add the oscillation:
						lightcurve += model(lightcurve.time, alpha[j], beta[j], nu[j])
						ps = powerspectrum(lightcurve)

						# Find the frequency of maximum power and find alpha and beta again
						nu[j] = ps.optimize_peak(nu[j])
						alpha[j], beta[j] = ps.alpha_beta(nu[j])

						# Recalculate the deviation
						atemp, btemp = original_ps.alpha_beta(nu[j])
						deviation[j] = (alpha[j]**2 + beta[j]**2)/(atemp**2 + btemp**2)

						# Remove the oscillation again:
						lightcurve -= model(lightcurve.time, alpha[j], beta[j], nu[j])

	# Remove anything that in the end was marked with a large deviation:
	for i in range(n_peaks):
		if deviation[i,0] > 1/alphadev or deviation[i,0] < alphadev:
			# If main peak is rejected, then also reject all harmonics
			nu[i,:] = np.nan
			alpha[i,:] = np.nan
			beta[i,:] = np.nan
		else:
			for j in range(1, n_harmonics+1):
				if deviation[i,j] > 1/alphadev or deviation[i,j] < alphadev:
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
	amp[~np.isfinite(amp)] = np.nan

	# Gather into features dictionary:
	features = {}
	for i in range(n_peaks):
		features['freq' + str(i+1)] = nu[i,0]
		features['amp' + str(i+1)] = amp[i,0]
		features['phase' + str(i+1)] = phase[i,0]
		for j in range(1, n_harmonics+1):
			features['freq' + str(i+1) + '_harmonic' + str(j)] = nu[i,j]
			features['amp' + str(i+1) + '_harmonic' + str(j)] = amp[i,j]
			features['phase' + str(i+1) + '_harmonic' + str(j)] = phase[i,j]

	return features

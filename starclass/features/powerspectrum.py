#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import lightkurve
from astropy.stats import LombScargle
from bottleneck import nanmedian
from scipy.optimize import minimize_scalar
from scipy.integrate import simps

class powerspectrum(object):
	"""
	Attributes:
		nyquist (float): Nyquist frequency in Hz.
		df (float): Fundamental frequency spacing in Hz.
		standard (tuple): Frequency in microHz and power density spectrum sampled
			from 0 to ``nyquist`` with a spacing of ``df``.
		ls (``astropy.stats.LombScargle`` object):
	"""

	def __init__(self, lightcurve, fit_mean=False):
		"""
		Parameters:
			lightcurve (``lightkurve.LightCurve`` object): Lightcurve to estimate power spectrum for.
			fit_mean (boolean, optional):
		"""

		self.fit_mean = fit_mean

		indx = np.isfinite(lightcurve.flux)
		self.df = 1/(86400*(np.max(lightcurve.time[indx]) - np.min(lightcurve.time[indx]))) # Hz
		self.nyquist = 1/(2*86400*nanmedian(np.diff(lightcurve.time[indx]))) # Hz

		self.ls = LombScargle(lightcurve.time[indx]*86400, lightcurve.flux[indx], center_data=True, fit_mean=self.fit_mean) # , normalization='psd'

		# Calculate a better estimate of the fundamental frequency spacing:
		self.df = self.fundamental_spacing_integral()

		# Calculate standard power density spectrum:
		self.standard = self.powerspectrum(oversampling=1, nyquist_factor=1, scale='powerdensity')

	#------------------------------------------------------------------------------
	def fundamental_spacing_minimum(self):
		"""Estimate fundamental spacing using the first minimum spectral window function."""

		# Create "window" time series:
		freq_cen = 0.5*self.nyquist
		x = 0.5*np.sin(2*np.pi*freq_cen*self.ls.t) + 0.5*np.cos(2*np.pi*freq_cen*self.ls.t)

		# Calculate power spectrum for the given frequency range:
		ls = LombScargle(self.ls.t, x, center_data=True, fit_mean=self.fit_mean)

		# Minimize the window function around the first minimum:
		# Normalization is completely irrelevant
		window = lambda freq: ls.power(freq_cen+freq, normalization='psd', method='fast')
		res = minimize_scalar(window, [0.75*self.df, self.df, 1.25*self.df])
		df = res.x
		return df

	#------------------------------------------------------------------------------
	def fundamental_spacing_integral(self):
		"""Estimate fundamental spacing using the integral of the spectral window function."""
		# Integrate the windowfunction
		freq, window = self.windowfunction(width=100*self.df, oversampling=5)
		df = simps(window, freq)
		return df*1e-6

	#------------------------------------------------------------------------------
	def powerspectrum(self, freq=None, oversampling=1, nyquist_factor=1, scale='power'):
		"""
		Calculate power spectrum for time series.

		Parameters:
			freq (ndarray, optional): Frequencies to calculate power spectrum for. If set
				to None, the full frequency range from 0 to ``nyquist``*``nyquist_factor`` is calculated.
			oversampling (float, optional): Oversampling factor. Default=1.
			nyquist_factor (float, optional): Nyquist factor. Default=1.
			scale (string, optional): 'power', 'powerdensity' and 'amplitude'. Default='power'.

		Returns:
			tuple: Tuple of two ndarray with frequencies in microHz and corresponding
				power in units depending on the ``scale`` keyword.
		"""

		N = len(self.ls.t)

		# The frequency axis in Hertz:
		if freq is None:
			freq = np.arange(self.df/oversampling, nyquist_factor*self.nyquist, self.df/oversampling, dtype='float64')

		# Calculate power at frequencies using fast Lomb-Scargle periodiogram:
		power = self.ls.power(freq, normalization='psd', method='fast', assume_regular_frequency=True)

		# Different scales:
		freq *= 1e6 # Rescale frequencies to being in microHz
		if scale == 'power':
			power *= 4/N
		elif scale == 'powerdensity':
			power *= 4/(N*self.df*1e6)
		elif scale == 'amplitude':
			power = np.sqrt(power*4/N)

		return freq, power

	#------------------------------------------------------------------------------
	def windowfunction(self, width=None, oversampling=10):
		"""Spectral window function.

		Parameters:
			width (float, optional): The width in Hz on either side of zero to calculate spectral window.
			oversampling (float, optional): Oversampling factor. Default=10.
		"""

		if width is None:
			width = 100*self.df

		freq_cen = 0.5*self.nyquist
		Nfreq = int(oversampling*width/self.df)
		freq = freq_cen + (self.df/oversampling) * np.arange(-Nfreq, Nfreq, 1)

		x = 0.5*np.sin(2*np.pi*freq_cen*self.ls.t) + 0.5*np.cos(2*np.pi*freq_cen*self.ls.t)

		# Calculate power spectrum for the given frequency range:
		ls = LombScargle(self.ls.t, x, center_data=True, fit_mean=self.fit_mean)
		power = ls.power(freq, method='fast', normalization='psd', assume_regular_frequency=True)
		power /= power[int(len(power)/2)] # Normalize to have maximum of one

		freq -= freq_cen
		freq *= 1e6
		return freq, power

	#------------------------------------------------------------------------------
	def plot(self, ax=None, xlabel='Frequency (muHz)', ylabel=None, style='lightkurve'):

		if ylabel is None:
			ylabel = {
				'powerdensity': 'Power density (ppm^2/muHz)',
				'power': 'Power (ppm^2)',
				'amplitude': 'Amplitude (ppm)'
			}['powerdensity'] # TODO: Only one setting for now...

		if style is None or style == 'lightkurve':
			style = lightkurve.MPLSTYLE

		with plt.style.context(style):
			if ax is None:
				fig, ax = plt.subplots(1)

			ax.loglog(self.standard[0], self.standard[1], 'k-')
			ax.set_xlabel('Frequency (muHz)')
			ax.set_ylabel(ylabel)
			ax.set_xlim(self.standard[0][0], self.standard[0][-1])

	#------------------------------------------------------------------------------
	#def clean(self):
	#
	#	freq, power = self.powerspectrum(oversampling=4, scale='power')
	#
	#	freq_guess = freq[np.argmax(power)]




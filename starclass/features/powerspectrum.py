#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import numpy as np
from astropy.stats import LombScargle
from bottleneck import nanmedian
from scipy.optimize import minimize_scalar
from scipy.integrate import simps

class powerspectrum(object):

	def __init__(self, lightcurve, fit_mean=False):
		
		self.lightcurve = lightcurve
		self.fit_mean = fit_mean
		
		indx = np.isfinite(flux)
		self.df = 1/(86400*(np.max(self.lightcurve.time[indx]) - np.min(self.lightcurve.time[indx]))) # Hz
		self.nyquist = 1/(2*86400*nanmedian(np.diff(self.lightcurve.time[indx]))) # Hz
		
		self.ls = LombScargle(lightcurve.time[indx]*86400, lightcurve.flux[indx], center_data=True, fit_mean=self.fit_mean, normalization='psd')
		
	#------------------------------------------------------------------------------
	def fundamental_spacing_minimum(self):

		# Initial guess for equidistant timestamps:
		#indx = np.isfinite(flux)

		freq_cen = 0.5*self.nyquist
		x = 0.5*np.sin(2*np.pi*freq_cen*self.lightcurve.time*86400) + 0.5*np.cos(2*np.pi*freq_cen*self.lightcurve.time*86400)

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
		# Integrate the windowfunction
		freq, window = self.windowfunction(oversampling=5)
		df = simps(window, freq)
		return df
		
	#------------------------------------------------------------------------------
	def powerspectrum(self, freq=None, oversampling=1, nyquist_factor=1, scale='power'):
		"""
		Calculate power spectrum for time series.

		Parameters:
			freq (ndarray, optional): 
			oversampling (float, optional): Oversampling factor. Default=1.
			nyquist_factor (float, optional): Nyquist factor. Default=1.
			scale (string, optional): 'power', 'powerdensity' and 'amplitude'.

		Returns:
			ndarray: Frequencies in microHz.
			ndarray: Corresponding power in units depending on the ``scale`` keyword.
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
	def windowfunction(self, width=30.0, oversampling=10):

		freq_cen = 0.5*self.nyquist
		Nfreq = int(oversampling*width*1e-6/self.df)
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
	#def clean(self):
	#
	#	freq, power = self.powerspectrum(oversampling=4, scale='power')
	#	
	#	freq_guess = freq[np.argmax(power)]
		
		
		

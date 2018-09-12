#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code is the property of L. Bugnet (please see and cite Bugnet et al.,2018).

The user should use the FliPer method to calculate FliPer values
from 0.2 ,0.7, 7, 20 and 50 muHz.

.. codeauthor:: Lisa Bugnet <lisa.bugnet@cea.fr>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division
import numpy as np
import gatspy.periodic as gp

def _normalise(time, flux, f, p, bw):
	"""
	Normalise according to Parseval's theorem
	"""
	rhs = 1.0 / len(flux) * np.sum(flux**2.0)
	lhs = p.sum()
	ratio = rhs / lhs
	return p * ratio / bw / 1e6

def _compute_ps(time, flux):
	"""
	Compute power spectrum using gatspy fast lomb scargle
	"""

	dt = 86400 * np.median(np.diff(time))
	tottime = 86400 * (np.max(time) - np.min(time))

	# Nyquist frequency
	nyq = 1.0 / (2.0*dt)
	# Frequency bin width
	df = 1.0 / tottime
	# Number of frequencies to compute
	Nf = nyq / df
	# Compute psd
	f, p = gp.lomb_scargle_fast.lomb_scargle_fast(time, flux,
												  f0=df,
												  df=df, Nf=Nf,
												  use_fft=True)
	# Calibrate power
	p = _normalise(time, flux, f, p, df)
	return f*1e6, p


def _APODIZATION(star_tab_psd):
	"""
	Function that corrects the spectra from apodization
	"""
	freq = star_tab_psd[0]
	nq = np.max(freq)
	nu = np.sin(np.pi/2.*freq/nq) / (np.pi/2.*freq/nq)
	star_tab_psd[1] /= nu**2
	return star_tab_psd

def _region(star_tab_psd, inic, end):
	"""
	Function that calculates the average power in a given frequency range on PSD
	"""
	freq = np.float64(star_tab_psd[0]) # convert frequencies in muHz
	power = np.float64(star_tab_psd[1])
	return np.mean(power[(freq >= inic) & (freq <= end)])

def FliPer(lightcurve):
	"""
	Compute FliPer values from 0.7, 7, 20, & 50 muHz

	Parameters:
		lightcurve (`lightkurve.TessLightCurve` object): Lightcurve of which to calculate
		the FliPer metrics.

	Returns:
		dict: Features from FliPer method.
	"""

	# Calculate powerspectrum with custom treatment of nans for FliPer method:
	# TODO: Can we make this use a pre-calculated powerspectrum?
	nancut = (lightcurve.flux==0) | np.isnan(lightcurve.flux)
	flux_ppm = (lightcurve.normalize().flux.copy() - 1.0) * 1e6
	flux_ppm[nancut] = 0.
	star_tab_psd = _compute_ps(lightcurve.time, flux_ppm)

	#star_tab_psd = _APODIZATION(star_tab_psd)
	end = 277 # muHz

	# Function that computes photon noise from last 100 bins of the spectra
	noise = np.mean(star_tab_psd[1][-100:])

	return {
		'Fp07': _region(star_tab_psd, 0.7, end) - noise,
		'Fp7' : _region(star_tab_psd, 7, end)   - noise,
		'Fp20': _region(star_tab_psd, 20, end)  - noise,
		'Fp50': _region(star_tab_psd, 50, end)  - noise
	}

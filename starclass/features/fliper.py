#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This code is the property of L. Bugnet (please see and cite Bugnet et al.,2018).

The user should use the FliPer method to calculate FliPer values
from 0.2 ,0.7, 7, 20 and 50 muHz.

.. codeauthor:: Lisa Bugnet <lisa.bugnet@cea.fr>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import numpy as np

# def _APODIZATION(star_tab_psd):
# 	"""
# 	Function that corrects the spectra from apodization
# 	"""
# 	freq = star_tab_psd[0]
# 	nq = np.max(freq)
# 	nu = np.sinc(2.*freq/nq)
# 	star_tab_psd[1] /= nu**2
# 	return star_tab_psd

def _region(star_tab_psd, inic, end):
	"""
	Function that calculates the average power in a given frequency range on PSD
	"""
	freq = star_tab_psd[0] # frequencies in muHz
	power = star_tab_psd[1] # Power density in ppm2/muHz
	return np.mean(power[(freq >= inic) & (freq <= end)])

def FliPer(psd):
	"""
	Compute FliPer values from 0.7, 7, 20, & 50 muHz

	Parameters:
		psd (`powerspectrum` object): Power spectrum of which to calculate
			the FliPer metrics.

	Returns:
		dict: Features from FliPer method.
	"""

	# Calculate powerspectrum with custom treatment of nans for FliPer method:
	star_tab_psd = psd.standard

	#star_tab_psd = _APODIZATION(star_tab_psd)
	end = 277 # muHz

	# Function that computes photon noise from last 100 bins of the spectra
	noise = np.median(star_tab_psd[1][-100:])/((1-2./18.)**3)

	return {
		'Fp07': _region(star_tab_psd, 0.7, end) - noise,
		'Fp7' : _region(star_tab_psd, 7, end)   - noise,
		'Fp20': _region(star_tab_psd, 20, end)  - noise,
		'Fp50': _region(star_tab_psd, 50, end)  - noise,
		'FpWhite': noise,
		'Fphi': _region(star_tab_psd, 0, 28)    - noise,
		'Fplo': _region(star_tab_psd, 250, end) - noise
	}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for the SORTING-HAT classifier.

.. codeauthor:: Jeroen Audenaert <jeroen.audenaert@kuleuven.be>
"""

import numpy as np
from bottleneck import nansum
import pyentrp.entropy as ent
from . import npeet_entropy_estimators as npeet
from ..RFGCClassifier.RF_GC_featcalc import prepLCs # noqa: F401

#--------------------------------------------------------------------------------------------------
def compute_lpf1pa11(featdictrow):
	"""
	Harmonic power in freq1

	Inputs
	-----------------


	Returns
	-----------------
	lpf1pa11: float
		log10(((amp11**2+amp12**2+amp13**2+amp14**2)/amp11**2)-1)

	"""
	tab = featdictrow['frequencies']
	peak1 = tab[tab['num'] == 1]
	amp11 = peak1['amplitude'][peak1['harmonic'] == 1]
	amps = peak1['amplitude'][peak1['harmonic'] > 0]
	return np.log10(nansum(amps**2)/amp11 - 1)

#--------------------------------------------------------------------------------------------------
def compute_varrat(featdictrow):
	"""
	Returns the variance ratio and number of significant harmonics

	Inputs
	-----------------
	featdictrow:

	Returns
	-----------------
	varrat: float
		(varinit - sum(var_of_sines)) / varinit, where var_of_sines = Amp ** 2 / 2

	significant_harmonics: int
		number of harmonics of f1 that is not nan
	"""
	tab = featdictrow['frequencies']
	peak1 = tab[tab['num'] == 1]

	amps = np.array(peak1['amplitude'])
	significant_harmonics = int(np.sum((peak1['harmonic'] > 0) & ~np.isnan(peak1['amplitude'])))

	varrat = (featdictrow['variance'] - nansum(amps**2 / 2)) / featdictrow['variance']
	if np.isnan(varrat):
		varrat = 1

	return varrat, significant_harmonics

#--------------------------------------------------------------------------------------------------
def compute_multiscale_entropy(flux):
	"""
	Returns the multiscale entropy

	Inputs
	-----------------
	flux:
		flux data

	Returns
	-----------------
	mean:	float
		mean of the mse curve
	max_v:	float
		maximum of the mse curve
	stdev:	float
		standard deviation of the mse curve
	power:	float
		power present in the mse curve
	"""

	ms_ent = ent.multiscale_entropy(flux, 2, maxscale=10)
	mean = np.mean(ms_ent)
	stdev = np.std(ms_ent)
	max_v = np.max(ms_ent)
	power = 1/len(ms_ent) * np.sum(ms_ent ** 2)

	return mean, max_v, stdev, power

#--------------------------------------------------------------------------------------------------
def helper_extract_digits(lst):
	return list(map(lambda el:[el], lst))

#--------------------------------------------------------------------------------------------------
def compute_differential_entropy(flux):
	"""
	Returns the differential entropy

	Inputs
	-----------------
	flux:
		flux data

	Returns
	-----------------
		entr: float
	"""
	data = helper_extract_digits(flux)
	entr = npeet.entropy(data)

	return entr

#--------------------------------------------------------------------------------------------------
#def compute_max_lyapunov_exponent(flux):
	"""
	Returns the maximum Lyapunov exponent calculated through the algortihm by Eckmann et al. (1986)

	Inputs
	-----------------
	flux:
		flux data

	Returns
	-----------------
		lyap_exp: float
	"""
	#lyap_exp = nolds.lyap_r(flux, emb_dim=10, lag=None, min_tsep=None)
	#lyap_exp = max(nolds.lyap_e(flux, emb_dim=10, matrix_dim=4, min_nb=None, min_tsep=0, tau=1, debug_plot=False, debug_data=False, plot_file=None))

	#return lyap_exp

#--------------------------------------------------------------------------------------------------
def compute_flux_ratio(flux):
	"""
	Returns the ratio of fluxes that are either larger than or smaller than the mean flux (Kim & Bailer-Jones, 2016)

	Inputs
	-----------------
	flux:
		flux data

	Returns
	-----------------
		flux_ratio: float
	"""
	mean = np.mean(flux)

	# For lower (fainter) fluxes than average.
	index = np.where(flux < mean)
	lower_flux = flux[index]
	lower_sum = np.sum((lower_flux - mean) ** 2) / len(lower_flux)

	# For higher (brighter) fluxes than average.
	index = np.where(flux >= mean)
	higher_flux = flux[index]

	higher_sum = np.sum((mean - higher_flux) ** 2) / len(higher_flux)

	# Return flux ratio
	return np.log(np.sqrt(lower_sum / higher_sum))

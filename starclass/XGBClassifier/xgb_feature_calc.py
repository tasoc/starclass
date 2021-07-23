#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Refilwe Kgoadi <refilwe.kgoadi1@my.jcu.edu.au>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from bottleneck import anynan
import scipy.stats as ss
from ..RFGCClassifier import RF_GC_featcalc
from ..utilities import get_periods

#--------------------------------------------------------------------------------------------------
def feature_extract(features, featnames, total=None, linflatten=False, recalc=False):

	if isinstance(features, dict):
		features = [features]
	if total is None:
		total = len(features)

	featout = np.empty([total, len(featnames)], dtype='float32')
	for k, obj in enumerate(features):
		# Load features from the provided (cached) features if they exist:
		featout[k, :] = np.array([obj.get(key, np.NaN) for key in featnames], dtype='float32')

		if recalc or anynan(featout[k, :]):
			# TODO: Why is it needed to re-normalize the lightcurve here?
			lc = RF_GC_featcalc.prepLCs(obj['lightcurve'], linflatten)

			featout[k, 0] = ss.skew(lc.flux) # Skewness
			featout[k, 1] = ss.kurtosis(lc.flux) # Kurtosis
			featout[k, 2] = ss.shapiro(lc.flux)[0] # Shapiro-Wilk test statistic for normality
			featout[k, 3] = calculate_eta(lc)

			periods, n_usedfreqs, usedfreqs = get_periods(obj, 6, lc.time, ignore_harmonics=False)
			amp21, amp31 = RF_GC_featcalc.freq_ampratios(obj, n_usedfreqs, usedfreqs)
			pd21, pd31 = RF_GC_featcalc.freq_phasediffs(obj, n_usedfreqs, usedfreqs)

			featout[k, 4] = periods[0]

			if n_usedfreqs > 0:
				featout[k, 5] = obj['frequencies'][(obj['frequencies']['num'] == 1) & (obj['frequencies']['harmonic'] == 0)]['amplitude']
			else:
				featout[k, 5] = 0.

			featout[k, 6] = amp21
			featout[k, 7] = amp31
			featout[k, 8] = pd21
			featout[k, 9] = pd31

			# phase-fold lightcurve on dominant period
			folded_lc = lc.fold(period=periods[0])

			# Compute phi_rcs and rcs features
			featout[k, 10] = Rcs(lc)
			featout[k, 11] = Rcs(folded_lc)

		# If the amp1 features is NaN, replace it with zero:
		if np.isnan(featout[k, 5]):
			featout[k, 5] = 0

	return featout

#--------------------------------------------------------------------------------------------------
def calculate_eta(lc):
	"""
	Calculate Eta feature.

	Parameters:
		mag (array_like): An array of magnitudes.
		std (array_like): A standard deviation of magnitudes.

	Returns:
		float: The value of Eta index.
	"""

	weight = 1. / lc.flux_err
	weighted_sum = np.sum(weight)
	weighted_mean = np.sum(lc.flux * weight) / weighted_sum
	std = np.sqrt(np.sum((lc.flux - weighted_mean)**2 * weight) / weighted_sum)

	diff = lc.flux[1:] - lc.flux[:len(lc.flux) - 1]
	eta = np.sum(diff * diff) / (len(lc.flux) - 1.) / std / std

	return eta

#--------------------------------------------------------------------------------------------------
def Rcs(lc):
	"""
	Range of cumulative sum.

	Parameters:
		lc (:class:`lightkurve.LightCurve`): Lightcurve to calculate Rcs for.

	Returns:
		float: Range of cumulative sum.
	"""
	sigma = np.std(lc.flux)
	N = len(lc)
	m = np.mean(lc.flux)
	s = np.cumsum(lc.flux - m) / (N * sigma)
	R = np.max(s) - np.min(s)
	return R

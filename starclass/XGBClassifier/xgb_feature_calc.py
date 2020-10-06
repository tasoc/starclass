#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Refilwe Kgoadi <refilwe.kgoadi1@my.jcu.edu.au>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pandas as pd
import numpy as np
import scipy.stats as ss
import os.path
import types
#from tqdm import tqdm
from ..RFGCClassifier import RF_GC_featcalc

#--------------------------------------------------------------------------------------------------
def feature_extract(features, savefeat=None, linflatten=False, recalc=False):
	featout = pd.DataFrame()
	if not isinstance(features, types.GeneratorType):
		features = [features]

	for obj in features:

		precalc = False
		if savefeat is not None:
			featfile = os.path.join(savefeat, str(obj['priority'])+'.txt')
			if os.path.exists(featfile) and not recalc:
				objfeatures = pd.read_csv(featfile)
				precalc = True
				featout = featout.append(objfeatures)

		if not precalc:
			# TODO: Why is it needed to re-normalize the lightcurve here?
			lc = RF_GC_featcalc.prepLCs(obj['lightcurve'], linflatten)

			features_dict = {}
			features_dict['skewness'] = ss.skew(lc.flux) # Skewness
			features_dict['kurtosis'] = ss.kurtosis(lc.flux) # Kurtosis
			features_dict['shapiro_wilk'] = ss.shapiro(lc.flux)[0] # Shapiro-Wilk test statistic for normality
			features_dict['eta'] = calculate_eta(lc)

			forbiddenfreqs = [13.49/4.]
			periods, usedfreqs = checkfrequencies(obj, 6, 6, forbiddenfreqs, lc.time)
			amp21, amp31 = freq_ampratios(obj,usedfreqs)
			pd21, pd31 = freq_phasediffs(obj,usedfreqs)

			features_dict['PeriodLS'] = periods[0]
			if len(usedfreqs) > 0:
				features_dict['Freq_amp_0'] = obj['amp' + str(usedfreqs[0]+1)]
			else:
				features_dict['Freq_amp_0'] = 0.

			features_dict['Freq_ampratio_21'] = amp21
			features_dict['Freq_ampratio_31'] = amp31
			features_dict['Freq_phasediff_21'] = pd21
			features_dict['Freq_phasediff_31'] = pd31

			# phase-fold lightcurve on dominant period
			folded_lc = lc.fold(period=periods[0])

			# Compute phi_rcs and rcs features
			features_dict['Rcs'] = Rcs(lc)
			features_dict['psi_Rcs'] = Rcs(folded_lc)

			objfeatures = pd.DataFrame(features_dict, index=[0])
			if savefeat is not None:
				objfeatures.to_csv(featfile, index=False)
			featout = featout.append(objfeatures)

			#Features_all.to_csv(os.path.join(Features_file_path, 'feets_features.csv'), index=False)
			#Features_all['ID'] = ID
			#Features_all.set_index('ID', inplace=True)
		#featout = np.vstack((featout, objfeatures.values))
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

#--------------------------------------------------------------------------------------------------
def checkfrequencies(featdictrow, nfreqs, providednfreqs, forbiddenfreqs, time):
	"""
	Cuts frequency data down to desired number of frequencies, and removes harmonics
	of forbidden frequencies

	Parameters:
		featdictrow (dict):
		nfreqs (integer):

	Returns:
		ndarray[nfreqs]: Array of frequencies.
	"""
	freqs = []
	usedfreqs = []
	j = 0
	while len(freqs) < nfreqs:
		freqdict = featdictrow['freq' + str(j+1)]
		if np.isfinite(freqdict):
			freq = 1./(freqdict*1e-6)/86400. # convert to days

			#check to cut bad frequencies
			cut = False
			if (freq < 0) or (freq > np.max(time)-np.min(time)):
				cut = True
			for freqtocut in forbiddenfreqs:
				for k in range(4): # cuts 4 harmonics of frequency, within +-3% of given frequency
					if (1./freq > (1./((k+1)*freqtocut))*(1-0.01)) & (1./freq < (1./((k+1)*freqtocut))*(1+0.01)):
						cut = True
			if not cut:
				freqs.append(freq)
				usedfreqs.append(j)
		j += 1
		if j >= providednfreqs:
			break
	#fill in any unfilled frequencies with negative numbers
	gap = nfreqs - len(freqs)
	if gap > 0:
		for k in range(gap):
			freqs.append(np.max(time)-np.min(time))
	return np.array(freqs), np.array(usedfreqs)

#--------------------------------------------------------------------------------------------------
def freq_ampratios(featdictrow, usedfreqs):
	"""
	Amplitude ratios of frequencies

	Inputs
	-----------------


	Returns
	-----------------
	amp21, amp31: float, float
		ratio of 2nd to 1st and 3rd to 1st frequency amplitudes

	"""
	if len(usedfreqs) >= 2:
		amp21 = featdictrow['amp'+str(usedfreqs[1]+1)]/featdictrow['amp'+str(usedfreqs[0]+1)]
	else:
		amp21 = 0
	if len(usedfreqs) >= 3:
		amp31 = featdictrow['amp'+str(usedfreqs[2]+1)]/featdictrow['amp'+str(usedfreqs[0]+1)]
	else:
		amp31 = 0
	return amp21,amp31

#--------------------------------------------------------------------------------------------------
def freq_phasediffs(featdictrow, usedfreqs):
	"""
	Phase differences of frequencies

	Inputs
	-----------------

	Returns
	-----------------
	phi21, phi31: float, float
		phase difference of 2nd to 1st and 3rd to 1st frequencies

	"""
	if len(usedfreqs) >= 2:
		phi21 = featdictrow['phase'+str(usedfreqs[1]+1)] - 2*featdictrow['phase'+str(usedfreqs[0]+1)]
	else:
		phi21 = 0
	if len(usedfreqs) >= 3:
		phi31 = featdictrow['phase'+str(usedfreqs[2]+1)] - 3*featdictrow['phase'+str(usedfreqs[0]+1)]
	else:
		phi31 = 0
	return phi21,phi31

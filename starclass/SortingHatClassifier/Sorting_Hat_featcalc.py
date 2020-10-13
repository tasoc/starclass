#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for the SORTING-HAT classifier.

.. codeauthor:: Jeroen Audenaert <jeroen.audenaert@kuleuven.be>
"""

import numpy as np
import os
from bottleneck import nansum
import scipy.stats as stat
import astropy.units as u
import pyentrp.entropy as ent
from . import npeet_entropy_estimators as npeet
from ..utilities import get_periods

#--------------------------------------------------------------------------------------------------
def featcalc(features, providednfreqs=6, nfrequencies=3,
	linflatten=False, savefeat=None, recalc=False):
	"""
	Calculates features for set of lightcurves
	"""

	featout = np.zeros([1,nfrequencies+10])
	if isinstance(features, dict): # trick for single features
		features = [features]

	for obj in features:
		precalc = False
		if savefeat is not None:
			featfile = os.path.join(savefeat, str(obj['priority'])+'.txt')
			if os.path.exists(featfile) and not recalc:
				#logger.info(str(obj['priority'])+": Loading precalculated features...")
				objfeatures = np.loadtxt(featfile, delimiter=',')
				precalc = True

		if not precalc:
			objfeatures = np.zeros(nfrequencies+10)
			lc = prepLCs(obj['lightcurve'],linflatten)

			periods = get_periods(obj, nfrequencies, lc.time, False)
			objfeatures[:nfrequencies] = periods

			#EBper = EBperiod(lc.time, lc.flux, periods[0], linflatten=linflatten-1)
			#objfeatures[0] = EBper # overwrites top period

			objfeatures[nfrequencies:nfrequencies+2] = compute_varrat(obj)
			#objfeatures[nfrequencies+1:nfrequencies+2] = compute_lpf1pa11(obj)
			objfeatures[nfrequencies+2:nfrequencies+3] = stat.skew(lc.flux)
			objfeatures[nfrequencies+3:nfrequencies+4] = compute_flux_ratio(lc.flux)
			objfeatures[nfrequencies+4:nfrequencies+5] = compute_differential_entropy(lc.flux)
			objfeatures[nfrequencies+5:nfrequencies+6] = compute_differential_entropy(obj['powerspectrum'].standard[1])
			objfeatures[nfrequencies+6:nfrequencies+10] = compute_multiscale_entropy(lc.flux)
			#objfeatures[nfrequencies+10:nfrequencies+11] = compute_max_lyapunov_exponent(lc.flux)

			if savefeat is not None:
				np.savetxt(featfile,objfeatures, delimiter=',')
		featout = np.vstack((featout,objfeatures))
	return featout[1:,:]

#--------------------------------------------------------------------------------------------------
def prepLCs(lc, linflatten=False, detrending_coeff=1):
	"""
	Nancut lightcurve, converts from ppm to relative flux and centres around 1.
	Optionally removes ith polynomial trend (=detrending_coeff).
	Assumes LCs come in normalised ppm with median zero.
	"""
	lc = lc.remove_nans()
	lc = 1e-6*lc + 1
	lc.flux_unit = u.dimensionless_unscaled

	if linflatten:
		lc = lc - np.polyval(np.polyfit(lc.time, lc.flux, detrending_coeff), lc.time) + 1

	#mean = np.mean(lc.flux)
	#variance = np.sqrt(np.sum(lc.flux - mean)**2 / (len(lc.flux) - 1))

	return lc

#--------------------------------------------------------------------------------------------------
def EBperiod(time, flux, per, cut_outliers=0, linflatten=True):
	"""
	.. codeauthor:: David Armstrong <d.j.armstrong@warwick.ac.uk>

	Tests for phase variation at double the current prime period,
	to correct EB periods.

	Inputs
	-----------------
	time
	flux
	per: 			float
		Period to phasefold self.lc at.
	cut_outliers:	float
		outliers ignored if difference from median in bin divided by the MAD is
		greater than cut_outliers.

	Returns
	-----------------
	corrected period: float
		Either initial period or double
	"""
	if per < 0:
		return per
	if linflatten:
		flux_flat = flux - np.polyval(np.polyfit(time,flux,1),time) + 1
	else:
		flux_flat = flux

	phaselc2P = np.zeros([len(time),2])
	phaselc2P = phasefold(time,per*2)
	idx = np.argsort(phaselc2P)
	phaselc2P = phaselc2P[idx]
	flux = flux_flat[idx]
	binnedlc2P = binPhaseLC(phaselc2P, flux, 64, cut_outliers=5) # ADD OUTLIER CUTS?

	minima = np.argmin(binnedlc2P[:,1])
	posssecondary = np.mod(np.abs(binnedlc2P[:,0]-np.mod(binnedlc2P[minima,0]+0.5,1.)),1.)
	#within 0.05 either side of phase 0.5 from minima
	posssecondary = np.where((posssecondary < 0.05) | (posssecondary > 0.95))[0]

	pointsort = np.sort(flux)
	top10points = np.median(pointsort[-30:])
	bottom10points = np.median(pointsort[:30])

	periodlim = (np.max(time) - np.min(time))/2. # no effective limit, could be changed
	if np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1] > 0.0025 \
		and np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1] \
		> 0.03*(top10points-bottom10points) \
		and per*2 <= periodlim:
		return 2*per
	else:
		return per

#--------------------------------------------------------------------------------------------------
def phasefold(time,per,t0=0):
	return np.mod(time-t0,per)/per

#--------------------------------------------------------------------------------------------------
def binPhaseLC(phase, flux, nbins, cut_outliers=0):
	"""
	.. codeauthor:: David Armstrong <d.j.armstrong@warwick.ac.uk>

	Bins a lightcurve, typically phase-folded.

	Inputs
	-----------------
	phase: 			ndarray, N
		Phase data (could use a time array instead)
	flux:			ndarray, N
		Flux data
	nbins:			int
		Number of bins to use
	cut_outliers:	float
		If not zero, cuts outliers where (difference to median)/MAD > cut_outliers

	Returns
	-----------------
	binnedlc:		ndarray, (nbins, 2)
		Array of (bin phases, binned fluxes)
	"""
	bin_edges = np.arange(nbins)/float(nbins)
	bin_indices = np.digitize(phase, bin_edges) - 1
	binnedlc = np.zeros([nbins,2])
	#fixes phase of all bins - means ignoring locations of points in bin
	binnedlc[:,0] = 1./nbins * 0.5 + bin_edges
	for bin in range(nbins):
		if np.sum(bin_indices == bin) > 0:
			inbin = np.where(bin_indices == bin)[0]
			if cut_outliers and np.sum(bin_indices == bin) > 2:
				mad = np.median(np.abs(flux[inbin]-np.median(flux[inbin])))
				outliers = np.abs((flux[inbin] - np.median(flux[inbin])))/mad <= cut_outliers
				inbin = inbin[outliers]
			binnedlc[bin,1] = np.mean(flux[inbin])
		else:
			#bit awkward this, but only alternative is to interpolate?
			binnedlc[bin,1] = np.mean(flux)
	return binnedlc

#--------------------------------------------------------------------------------------------------
def prepFilePhasefold(time, flux, period, cardinality):
	"""
	.. codeauthor:: David Armstrong <d.j.armstrong@warwick.ac.uk>

	Prepares a lightcurve for using with the SOM.

	Inputs
	-----------------
	time
	flux
	period: 			float
		Period to phasefold self.lc at
	cardinality:		int
		Number of bins used in SOM

	Returns
	-----------------
	binnedlc:		ndarray, (cardinality, 2)
		Array of (bin phases, binned fluxes)
	range:			float
		Max - Min if binned lightcurve
	"""
	phase = phasefold(time,period)
	idx = np.argsort(phase)
	binnedlc = binPhaseLC(phase[idx],flux[idx],cardinality)
	#normalise to between 0 and 1
	minflux = np.min(binnedlc[:,1])
	maxflux = np.max(binnedlc[:,1])
	if maxflux != minflux:
		binnedlc[:,1] = (binnedlc[:,1]-minflux) / (maxflux-minflux)
	else:
		binnedlc[:,1] = np.ones(cardinality)
	#offset so minimum is at phase 0
	binnedlc[:,0] = np.mod(binnedlc[:,0]-binnedlc[np.argmin(binnedlc[:,1]),0],1)
	binnedlc = binnedlc[np.argsort(binnedlc[:,0]),:]
	return binnedlc[:,1],maxflux-minflux

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

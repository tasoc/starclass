#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of Frequency extraction.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""
import pytest
import numpy as np
import lightkurve as lk
import conftest # noqa: F401
from starclass.features.freqextr import freqextr
from starclass.features.powerspectrum import powerspectrum
from starclass.plots import plt

tol_freq = {'atol': 0.001, 'rtol': 0.001}
tol_amp = {'atol': 0.1, 'rtol': 0.1}
tol_phase = {'atol': 0.05, 'rtol': 0.05}

#--------------------------------------------------------------------------------------------------
def _summary(feat, ax):
	print("-"*72)
	k = 1
	while feat.get('freq%d' % k, None) is not None:
		f = feat['freq' + str(k)]
		a = feat['amp' + str(k)]
		p = feat['phase' + str(k)]
		if np.isfinite(f):
			ax.plot(f, a, 'ro')
			print("0   %7.3f   %6.3f   %6.3f" % (f, a, p))

		j = 1
		while feat.get('freq%d_harmonic%d' % (k,j), None) is not None:
			f = feat['freq' + str(k) + '_harmonic' + str(j)]
			a = feat['amp' + str(k) + '_harmonic' + str(j)]
			p = feat['phase' + str(k) + '_harmonic' + str(j)]
			if np.isfinite(f):
				ax.plot(f, a, 'go')
				print("%d   %7.3f   %6.3f   %6.3f" % (j, f, a, p))
			j += 1

		k += 1
	print("-"*72)

#--------------------------------------------------------------------------------------------------
def test_freqextr_simple():

	np.random.seed(42)
	time = np.arange(0, 27.0, 1800/86400)
	flux = 10*np.sin(2*np.pi*50e-6*time*86400) + 2*np.sin(2*np.pi*100e-6*time*86400) + np.random.normal(0, 2, size=len(time))
	lc = lk.TessLightCurve(
		time=time,
		flux=flux
	)

	feat = freqextr(lc, n_peaks=5, n_harmonics=0)

	ps = powerspectrum(lc)
	frequency, power = ps.powerspectrum(oversampling=50, scale='amplitude')

	fig, (ax1, ax2) = plt.subplots(2, 1)
	lc.plot(ax=ax1, normalize=False)
	ax2.plot(frequency, power, 'k-', lw=0.5)
	_summary(feat, ax2)

	np.testing.assert_allclose(feat['freq1'], 50, **tol_freq)
	np.testing.assert_allclose(feat['amp1'], 10, **tol_amp)
	np.testing.assert_allclose(feat['phase1'], 0, **tol_phase)

	np.testing.assert_allclose(feat['freq2'], 100, **tol_freq)
	np.testing.assert_allclose(feat['amp2'], 2, **tol_amp)
	np.testing.assert_allclose(feat['phase2'], 0, **tol_phase)

	assert np.isnan(feat['freq3'])
	assert np.isnan(feat['amp3'])
	assert np.isnan(feat['phase3'])

#--------------------------------------------------------------------------------------------------
def test_freqextr_onlynoise():

	np.random.seed(42)
	time = np.arange(0, 27.0, 1800/86400)
	flux = np.random.normal(0, 2, size=len(time))
	lc = lk.TessLightCurve(
		time=time,
		flux=flux
	)

	feat = freqextr(lc, n_peaks=5, n_harmonics=2)

	ps = powerspectrum(lc)
	frequency, power = ps.powerspectrum(oversampling=50, scale='amplitude')

	fig, (ax1, ax2) = plt.subplots(2, 1)
	lc.plot(ax=ax1, normalize=False)
	ax2.plot(frequency, power, 'k-', lw=0.5)
	_summary(feat, ax2)

	assert np.isnan(feat['freq1'])
	assert np.isnan(feat['amp1'])
	assert np.isnan(feat['phase1'])

#--------------------------------------------------------------------------------------------------
def test_freqextr():

	np.random.seed(42)
	time = np.arange(0, 27.0, 1800/86400)
	flux = 10*np.sin(2*np.pi*50e-6*time*86400) + 2*np.sin(2*np.pi*100e-6*time*86400) \
		+ 3*np.sin(2*np.pi*89e-6*time*86400 + 0.5) \
		+ 12*np.sin(2*np.pi*91.3e-6*time*86400 + 0.32) \
		+ 6*np.sin(2*np.pi*2*91.3e-6*time*86400 + 0.32) \
		+ 2.4*np.random.randn(len(time))

	lc = lk.TessLightCurve(
		time=time,
		flux=flux
	)

	feat = freqextr(lc, n_peaks=5, n_harmonics=2)

	ps = powerspectrum(lc)
	frequency, power = ps.powerspectrum(oversampling=50, scale='amplitude')

	fig, (ax1, ax2) = plt.subplots(2, 1)
	lc.plot(ax=ax1, normalize=False)
	ax2.plot(frequency, power, 'k-', lw=0.5)
	_summary(feat, ax2)

	np.testing.assert_allclose(feat['freq1'], 91.3, **tol_freq)
	np.testing.assert_allclose(feat['amp1'], 12, **tol_amp)
	np.testing.assert_allclose(feat['phase1'], 0.32, **tol_phase)

#--------------------------------------------------------------------------------------------------
def test_freqextr_kepler():
	#time, flux = np.loadtxt('kic1162345_all.dat', unpack=True, usecols=(0,1))
	#lc = lk.KeplerLightCurve(
	#	time=time,
	#	flux=flux,
	#	targetid=1162345,
	#	time_format='bkjd',
	#	time_scale='tdb'
	#)

	lcfs = lk.search_lightcurvefile(1162345, mission='Kepler', cadence='long').download_all()
	lc = lcfs.PDCSAP_FLUX.stitch()
	lc = lc.remove_nans().remove_outliers()
	lc = 1e6*(lc.normalize() - 1)

	feat = freqextr(lc, n_peaks=5, n_harmonics=2)

	ps = powerspectrum(lc)
	frequency, power = ps.powerspectrum(oversampling=50, scale='amplitude')

	fig, (ax1, ax2) = plt.subplots(2, 1)
	lc.plot(ax=ax1, normalize=False)
	ax2.plot(frequency, power, 'k-', lw=0.5)
	_summary(feat, ax2)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	plt.switch_backend('Qt5Agg')
	pytest.main([__file__])
	plt.show()

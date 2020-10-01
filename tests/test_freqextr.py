#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of Frequency extraction.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""
import pytest
import numpy as np
from bottleneck import allnan
import lightkurve as lk
from astropy.units import cds
import conftest # noqa: F401
from starclass.features.freqextr import freqextr
from starclass.features.powerspectrum import powerspectrum
from starclass.plots import plt, plots_interactive

tol_freq = {'atol': 0.001, 'rtol': 0.001}
tol_amp = {'atol': 0.1, 'rtol': 0.1}
tol_phase = {'atol': 0.05, 'rtol': 0.05}

#--------------------------------------------------------------------------------------------------
def _summary(lc, tab):

	tab.pprint_all()

	ps = powerspectrum(lc)
	frequency, power = ps.powerspectrum(oversampling=50, scale='amplitude')

	fig, (ax1, ax2) = plt.subplots(2, 1)
	lc.plot(ax=ax1, normalize=False)
	ax2.plot(frequency, power, 'k-', lw=0.5)

	for row in tab:
		f = row['frequency']
		if np.isfinite(f):
			a = row['amplitude']
			if row['harmonic'] == 0:
				ax2.plot(f, a, 'ro')
			else:
				ax2.plot(f, a, 'go')

	# Check the number of rows returned:
	assert len(tab) == tab.meta['n_peaks'] * (tab.meta['n_harmonics'] + 1), "Incorrect number of rows"
	assert sum(tab['harmonic'] == 0) == tab.meta['n_peaks'], "Incorrect number of peaks"
	assert sum(tab['harmonic'] > 0) == tab.meta['n_harmonics']*tab.meta['n_peaks'], "Incorrect number of harmonics"

	# Check that non-harmonic peaks are ordered by amplitude:
	indx = ~np.isnan(tab['amplitude']) & (tab['harmonic'] == 0)
	assert np.all(np.diff(tab['amplitude'][indx]) <= 0), "Not sorted by amplitude"

	# Check derived parameters match:
	np.testing.assert_allclose(tab['amplitude'], np.sqrt(tab['alpha']**2 + tab['beta']**2))
	np.testing.assert_allclose(tab['phase'], np.arctan2(tab['beta'], tab['alpha']))

#--------------------------------------------------------------------------------------------------
def test_freqextr_simple():

	np.random.seed(42)
	time = np.arange(0, 27.0, 1800/86400)
	flux = 10*np.sin(2*np.pi*50e-6*time*86400) \
		+ 2*np.sin(2*np.pi*100e-6*time*86400) \
		+ np.random.normal(0, 2, size=len(time))
	lc = lk.LightCurve(time=time, flux=flux)

	tab = freqextr(lc, n_peaks=5, n_harmonics=0)
	_summary(lc, tab)

	num = tab['num']
	h = tab['harmonic']
	assert tab.meta['n_peaks'] == 5
	assert tab.meta['n_harmonics'] == 0
	assert np.all(h == 0)

	#peak1 = tab.loc[(1, 0)]
	peak1 = tab[(num == 1)]
	assert len(peak1) == 1
	np.testing.assert_allclose(peak1['frequency'], 50, **tol_freq)
	np.testing.assert_allclose(peak1['amplitude'], 10, **tol_amp)
	np.testing.assert_allclose(peak1['phase'], 0, **tol_phase)

	#peak2 = tab.loc[[2, 0]]
	peak2 = tab[(num == 2)]
	assert len(peak2) == 1
	np.testing.assert_allclose(peak2['frequency'], 100, **tol_freq)
	np.testing.assert_allclose(peak2['amplitude'], 2, **tol_amp)
	np.testing.assert_allclose(peak2['phase'], 0, **tol_phase)

	for n in range(3, 6):
		#peakn = tab.loc[[n, 0]]
		peakn = tab[(num == n)]
		assert len(peakn) == 1
		assert np.isnan(peakn['frequency'])
		assert np.isnan(peakn['amplitude'])
		assert np.isnan(peakn['phase'])
		assert np.isnan(peakn['alpha'])
		assert np.isnan(peakn['beta'])
		assert np.isnan(peakn['deviation'])

#--------------------------------------------------------------------------------------------------
def test_freqextr_onlynoise():

	np.random.seed(42)
	time = np.arange(0, 27.0, 1800/86400)
	flux = np.random.normal(0, 2, size=len(time))
	lc = lk.TessLightCurve(time=time, flux=flux)

	tab = freqextr(lc, n_peaks=5, n_harmonics=2)
	_summary(lc, tab)

	assert tab.meta['n_peaks'] == 5
	assert tab.meta['n_harmonics'] == 2

	assert allnan(tab['frequency'])
	assert allnan(tab['amplitude'])
	assert allnan(tab['phase'])
	assert allnan(tab['alpha'])
	assert allnan(tab['beta'])
	assert allnan(tab['deviation'])

#--------------------------------------------------------------------------------------------------
def test_freqextr():

	np.random.seed(42)
	time = np.arange(0, 27.0, 1800/86400)
	flux = 10*np.sin(2*np.pi*50e-6*time*86400) + 2*np.sin(2*np.pi*100e-6*time*86400) \
		+ 3*np.sin(2*np.pi*89e-6*time*86400 + 0.5) \
		+ 12*np.sin(2*np.pi*91.3e-6*time*86400 + 0.32) \
		+ 6*np.sin(2*np.pi*2*91.3e-6*time*86400 + 0.32) \
		+ 2.4*np.random.randn(len(time))

	lc = lk.TessLightCurve(time=time, flux=flux)

	tab = freqextr(lc, n_peaks=5, n_harmonics=2)
	_summary(lc, tab)

	assert tab.meta['n_peaks'] == 5
	assert tab.meta['n_harmonics'] == 2

	peak1 = tab[0]
	assert peak1['num'] == 1
	assert peak1['harmonic'] == 0
	np.testing.assert_allclose(peak1['frequency'], 91.3, **tol_freq)
	np.testing.assert_allclose(peak1['amplitude'], 12, **tol_amp)
	np.testing.assert_allclose(peak1['phase'], 0.32, **tol_phase)

#--------------------------------------------------------------------------------------------------
def test_freqextr_kepler():

	lcfs = lk.search_lightcurvefile('KIC 1162345', mission='Kepler', cadence='long')
	# Pretty hacky way of making sure lightkurve only returned the target we want:
	lcfs.table = lcfs.table[lcfs.target_name == 'kplr001162345']
	lcfs = lcfs.download_all()
	lc = lcfs.PDCSAP_FLUX.stitch()
	lc = lc.remove_nans().remove_outliers()
	lc = 1e6*(lc - 1)
	lc.flux_unit = cds.ppm

	tab = freqextr(lc, n_peaks=10, n_harmonics=2, snrlim=4)
	_summary(lc, tab)

	assert tab.meta['n_peaks'] == 10
	assert tab.meta['n_harmonics'] == 2
	assert tab.meta['snrlim'] == 4

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	plots_interactive()
	pytest.main([__file__])
	plt.show(block=True)

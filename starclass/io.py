#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from astropy.units import cds
from astropy.io import fits
import lightkurve as lk

#--------------------------------------------------------------------------------------------------
def load_lightcurve(fname, starid=None):

	if fname.endswith(('.txt', '.noisy', '.sysnoise', '.clean')):
		data = np.loadtxt(fname)
		if data.shape[1] == 4:
			quality = np.asarray(data[:,3], dtype='int32')
		else:
			quality = np.zeros(data.shape[0], dtype='int32')

		lightcurve = lk.TessLightCurve(
			time=data[:,0],
			flux=data[:,1],
			flux_err=data[:,2],
			flux_unit=cds.ppm,
			quality=quality,
			time_format='jd',
			time_scale='tdb',
			targetid=starid,
			quality_bitmask=2+8+256, # lightkurve.utils.TessQualityFlags.DEFAULT_BITMASK,
			meta={}
		)

	elif fname.endswith(('.fits.gz', '.fits')):
		with fits.open(fname, mode='readonly', memmap=True) as hdu:
			telescope = hdu[0].header.get('TELESCOP')
			if telescope == 'TESS' and hdu[0].header.get('ORIGIN') == 'TASOC/Aarhus':
				lightcurve = lk.TessLightCurve(
					time=hdu['LIGHTCURVE'].data['TIME'],
					flux=hdu['LIGHTCURVE'].data['FLUX_CORR'],
					flux_err=hdu['LIGHTCURVE'].data['FLUX_CORR_ERR'],
					flux_unit=cds.ppm,
					centroid_col=hdu['LIGHTCURVE'].data['MOM_CENTR1'],
					centroid_row=hdu['LIGHTCURVE'].data['MOM_CENTR2'],
					quality=np.asarray(hdu['LIGHTCURVE'].data['QUALITY'], dtype='int32'),
					cadenceno=np.asarray(hdu['LIGHTCURVE'].data['CADENCENO'], dtype='int32'),
					time_format='btjd',
					time_scale='tdb',
					targetid=hdu[0].header.get('TICID'),
					label=hdu[0].header.get('OBJECT'),
					camera=hdu[0].header.get('CAMERA'),
					ccd=hdu[0].header.get('CCD'),
					sector=hdu[0].header.get('SECTOR'),
					ra=hdu[0].header.get('RA_OBJ'),
					dec=hdu[0].header.get('DEC_OBJ'),
					quality_bitmask=1+2+256, # CorrectorQualityFlags.DEFAULT_BITMASK
					meta={}
				)
			elif telescope == 'TESS':
				lightcurve = lk.TESSLightCurveFile(hdu).PDCSAP_FLUX
				lightcurve = 1e6 * (lightcurve.normalize() - 1)
				lightcurve.flux_unit = cds.ppm
			elif telescope == 'Kepler':
				lightcurve = lk.KeplerLightCurveFile(hdu).PDCSAP_FLUX
				lightcurve = 1e6 * (lightcurve.normalize() - 1)
				lightcurve.flux_unit = cds.ppm
			else:
				raise ValueError("Could not determine FITS lightcurve type")
	else:
		raise ValueError("Invalid file format")

	return lightcurve

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input/output functions.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pickle
import gzip
import json
import numpy as np
from bottleneck import nanmin
from astropy.units import cds
from astropy.io import fits
import lightkurve as lk

PICKLE_DEFAULT_PROTOCOL = 4 #: Default protocol to use for saving pickle files.

#--------------------------------------------------------------------------------------------------
def load_lightcurve(fname, starid=None, truncate_lightcurve=False):
	"""
	Load light curve from file.

	Parameters:
		fname (str): Path to file to be loaded.
		starid (int): Star identifier (TIC/KIC/EPIC number) to be added to lightcurve object.
			This is only used for file types where the number can not be determined from the
			file itself.
		truncate_lightcurve (bool): Truncate lightcurve to 27.4 days length, corresponding to
			the nominal length of a TESS observing sector. This is only applied to Kepler/K2
			data.

	Returns:
		:class:`lightkurve.LightCurve`: Lightcurve object.

	Raises:
		ValueError: On invalid file format.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

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
					targetid=hdu[0].header.get('TICID', starid),
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
				if truncate_lightcurve:
					indx = (lightcurve.time - nanmin(lightcurve.time) <= 27.4)
					lightcurve = lightcurve[indx]
				lightcurve = 1e6 * (lightcurve.normalize() - 1)
				lightcurve.flux_unit = cds.ppm
			else:
				raise ValueError("Could not determine FITS lightcurve type")
	else:
		raise ValueError("Invalid file format")

	return lightcurve

#--------------------------------------------------------------------------------------------------
def savePickle(fname, obj):
	"""
	Save an object to file using pickle.

	Parameters:
		fname (str): File name to save to. If the name ends in '.gz' the file
			will be automatically gzipped.
		obj (object): Any pickalble object to be saved to file.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	if fname.endswith('.gz'):
		o = gzip.open
	else:
		o = open

	with o(fname, 'wb') as fid:
		pickle.dump(obj, fid, protocol=PICKLE_DEFAULT_PROTOCOL)

#--------------------------------------------------------------------------------------------------
def loadPickle(fname):
	"""
	Load an object from file using pickle.

	Parameters:
		fname (str): File name to load from. If the name ends in '.gz' the file
			will be automatically unzipped.

	Returns:
		object: The unpickled object from the file.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	if fname.endswith('.gz'):
		o = gzip.open
	else:
		o = open

	with o(fname, 'rb') as fid:
		return pickle.load(fid)

#--------------------------------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.integer):
			return int(obj)
		return json.JSONEncoder.default(self, obj)

#--------------------------------------------------------------------------------------------------
def saveJSON(fname, obj):
	"""
	Save an object to JSON file.

	Parameters:
		fname (str): File name to save to. If the name ends in '.gz' the file
			will be automatically gzipped.
		obj (object): Any pickalble object to be saved to file.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	if fname.endswith('.gz'):
		o = gzip.open
	else:
		o = open

	with o(fname, 'wt', encoding='utf-8') as fid:
		json.dump(obj, fid, ensure_ascii=False, indent='\t', cls=NumpyEncoder)

#--------------------------------------------------------------------------------------------------
def loadJSON(fname):
	"""
	Load an object from a JSON file.

	Parameters:
		fname (str): File name to load to. If the name ends in '.gz' the file
			will be automatically unzipped.

	Returns:
		object: The object from the file.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	if fname.endswith('.gz'):
		o = gzip.open
	else:
		o = open

	with o(fname, 'r') as fid:
		return json.load(fid)

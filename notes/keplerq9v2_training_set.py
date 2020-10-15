#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create new Kepler Q9 version 2 training set.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import sys
import os.path
import shutil
import numpy as np
from bottleneck import nanmedian, nanvar
from tqdm import tqdm
import requests
from astropy.io import fits
from lightkurve import LightCurve
if sys.path[0] != os.path.abspath('..'):
	sys.path.insert(0, os.path.abspath('..'))
from starclass.utilities import rms_timescale
#from starclass.plots import plt

#--------------------------------------------------------------------------------------------------
def download_file(url, fpath):
	tqdm_settings = {
		'unit': 'B',
		'unit_scale': True,
		'unit_divisor': 1024,
		'position': 1,
		'leave': False
	}

	try:
		res = requests.get(url, stream=True)
		res.raise_for_status()
		total_size = int(res.headers.get('content-length', 0))
		block_size = 1024
		with tqdm(total=total_size, **tqdm_settings) as pbar:
			with open(fpath, 'wb') as fid:
				for data in res.iter_content(block_size):
					datasize = fid.write(data)
					pbar.update(datasize)

	except: # noqa: E722, pragma: no cover
		if os.path.isfile(fpath):
			os.remove(fpath)
		raise

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	# Directory where KeplerQ9 ASCII files are stored:
	# This can either contain all files in this directory,
	# or be divided into sub-directories.
	thisdir = r'G:\keplerq9\full_fits'

	# Where output will be saved:
	output_dir = r'G:\keplerq9\keplerq9v3'

	# Make sure directories exists:
	os.makedirs(output_dir, exist_ok=True)
	shutil.copy('keplerq9v2_targets.txt', os.path.join(output_dir, 'targets.txt'))

	# Load the list of KIC numbers and stellar classes:
	starlist = np.genfromtxt(os.path.join(output_dir, 'targets.txt'),
		delimiter=',', comments='#', dtype='str', encoding='utf-8')

	# Make sure that there are not duplicate entries:
	starlist_unique, cnt = np.unique(starlist, return_counts=True, axis=0)
	if len(starlist) != len(starlist_unique):
		print("Duplicate entries:")
		for line in starlist_unique[cnt > 1]:
			print('%s,%s' % tuple(line))

	# Make sure that there are not duplicate starids:
	us, cnt = np.unique(starlist[:, 0], return_counts=True)
	if len(starlist) != len(us):
		print("Duplicate starids in multiple classes:")
		for s in us[cnt > 1]:
			indx = (starlist[:, 0] == s)
			for line in starlist[indx, :]:
				print('%s,%s' % tuple(line))

		raise Exception("%d duplicate starids" % (len(starlist) - len(us)))

	# Open the diagnostics file for writing and loop through the
	# list of starids, load the data file, restructure the timeseries,
	# calculate diagnostics and save these to files:
	with open(os.path.join(output_dir, 'diagnostics.txt'), 'w') as diag:
		diag.write("# Kepler Q9 Training Set Targets (version 2)\n")
		diag.write("# Pre-calculated diagnostics\n")
		diag.write("# Column 1: Variance (ppm2)\n")
		diag.write("# Column 2: RMS per hour (ppm2/hour)\n")
		diag.write("# Column 3: Point-to-point scatter (ppm)\n")
		diag.write("#-------------------------------------------\n")

		for starid, sclass in tqdm(starlist):
			sclass = sclass.upper()
			if starid.startswith('constant_'):
				fname_save = starid + '.txt'
				starid = -10000 - int(starid[9:])
			elif starid.startswith('fakerrlyr_'):
				fname_save = starid + '.txt'
				starid = -20000 - int(starid[10:])
			else:
				starid = int(starid)
				fname_save = '{starid:09d}.txt'.format(starid=starid)

			# The path to save the final timeseries:
			fpath_save = os.path.join(output_dir, sclass, fname_save)

			if os.path.isfile(fpath_save):
				# Load file that should already exist:
				time, flux, flux_err = np.loadtxt(fpath_save, usecols=(0,1,2), unpack=True, comments='#')
			else:
				fname = 'kplr{starid:09d}-2011177032512_llc.fits'.format(starid=starid)
				subdir = os.path.join(thisdir, '{0:09d}'.format(starid)[:5])
				fpath = os.path.join(subdir, fname)

				# Check if the file is available as gzipped FITS:
				if os.path.isfile(fpath + '.gz'):
					fpath += '.gz'

				# Print if file does not exist, instead of failing one
				# at a time. Making debugging a little easier:
				#tqdm.write(fpath)
				if not os.path.isfile(fpath):
					url = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:Kepler/url/missions/kepler/lightcurves/{subdir:s}/{starid:09d}/{fname:s}'.format(
						subdir='{0:09d}'.format(starid)[:4],
						starid=starid,
						fname=fname)
					os.makedirs(subdir, exist_ok=True)
					download_file(url, fpath)

				# Load Kepler Q9 FITS file (PDC corrected):
				with fits.open(fpath, mode='readonly', memmap=True) as hdu:
					time = np.asarray(hdu[1].data['TIME'])
					flux = np.asarray(hdu[1].data['PDCSAP_FLUX'])
					flux_err = np.asarray(hdu[1].data['PDCSAP_FLUX_ERR'])
					quality = np.asarray(hdu[1].data['SAP_QUALITY'], dtype='int32')

				# Remove anything without a timestamp:
				indx = np.isfinite(time)
				time, flux, flux_err, quality = time[indx], flux[indx], flux_err[indx], quality[indx]

				# Remove anything with bad quality:
				indx = ~np.isfinite(flux) | ~np.isfinite(flux_err) | (quality & 1130799 != 0)
				flux[indx] = np.NaN
				flux_err[indx] = np.NaN

				# Just removing the first data point as it is always NaN anyway
				time = time[1:]
				flux = flux[1:]
				flux_err = flux_err[1:]

				# Subtract the first timestamp from all timestamps:
				time -= time[0]

				# Only keep the first 27.4 days of data:
				indx = (time <= 27.4)
				time, flux, flux_err = time[indx], flux[indx], flux_err[indx]

				# Convert to ppm:
				m = nanmedian(flux)
				flux = 1e6*(flux/m - 1)
				flux_err = 1e6*flux_err/m

				#fig, ax = plt.subplots()
				#ax.plot(data[:,0], data[:,1])
				#fig.savefig(os.path.splitext(fpath_save)[0] + '.png', bbox_inches='tight')
				#plt.close(fig)

				# Save file:
				os.makedirs(os.path.dirname(fpath_save), exist_ok=True)
				np.savetxt(fpath_save, np.column_stack((time, flux, flux_err)),
					delimiter='  ', fmt=('%.8f', '%.16e', '%.16e'))

			# Calculate diagnostics:
			lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
			variance = nanvar(flux, ddof=1)
			rms_hour = rms_timescale(lc, timescale=3600/86400)
			ptp = nanmedian(np.abs(np.diff(flux)))

			# Add target to TODO-list:
			diag.write("{variance:.16e},{rms_hour:.16e},{ptp:.16e}\n".format(
				variance=variance,
				rms_hour=rms_hour,
				ptp=ptp
			))

		diag.write("#-------------------------------------------\n")

	print("DONE")

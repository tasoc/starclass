#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create new Kepler Q9 version 3 training set.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import sys
import os.path
import shutil
import numpy as np
from bottleneck import nanvar
from tqdm import tqdm
import requests
import subprocess
if sys.path[0] != os.path.abspath('..'):
	sys.path.insert(0, os.path.abspath('..'))
import starclass.utilities as util
from starclass import io
from starclass.plots import plt, plots_interactive

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
def create_constant_star(fpath, timelen=90.0, sampling=30.0, random_state=None):

	# Get the standard deviation for this star from the pre-computed list:
	data = np.genfromtxt('keplerq9v3_targets_constant.txt',
		delimiter=',', comments='#', dtype=None, encoding='utf-8')
	const_stars_ids = data['f0']
	const_stars_sigma = data['f1']

	identifier = os.path.basename(fpath).replace('.txt', '')
	sigma = const_stars_sigma[const_stars_ids == identifier]
	if len(sigma) != 1:
		raise Exception("SIGMA not found")
	sigma = float(sigma)

	# Create random lightcurve:
	time = np.arange(0, timelen, sampling/1440, dtype='float64')
	N = len(time)
	flux = random_state.normal(size=N, loc=0, scale=sigma)

	# Save the lightcurve to file:
	os.makedirs(os.path.dirname(fpath), exist_ok=True)
	with open(fpath, 'w') as fid:
		fid.write("# Kepler Q9 Training Set Targets artificial constant star\n")
		fid.write("# Identifier: %s\n" % identifier)
		fid.write("# Column 1: Time (days)\n")
		fid.write("# Column 2: Flux (ppm)\n")
		fid.write("# Column 3: Flux error (ppm)\n")
		fid.write("#-------------------------------------------\n")
		for k in range(N):
			fid.write("{0:.16e}  {1:.16e}  {2:.16e}\n".format(
				time[k],
				flux[k],
				sigma
			))
		fid.write("#-------------------------------------------\n")

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	plots_interactive()

	# Seed the random state used below to ensure reproducability:
	rng = np.random.RandomState(42)

	# Directory where KeplerQ9 files are stored locally:
	thisdir = r'G:\keplerq9\full_fits'

	# Where output will be saved:
	output_dir = r'G:\keplerq9\keplerq9v3-long'

	# Make sure directories exists:
	os.makedirs(output_dir, exist_ok=True)
	shutil.copy('keplerq9v3_targets.txt', os.path.join(output_dir, 'targets.txt'))

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
		diag.write("# Kepler Q9 Training Set Targets (version 3, long)\n")
		diag.write("# Pre-calculated diagnostics\n")
		diag.write("# Column 1: Variance (ppm2)\n")
		diag.write("# Column 2: RMS per hour (ppm2/hour)\n")
		diag.write("# Column 3: Point-to-point scatter (ppm)\n")
		diag.write("#-------------------------------------------\n")

		for starid, sclass in tqdm(starlist):
			sclass = sclass.upper()
			if starid.startswith('constant_'):
				fpath_save = os.path.join(output_dir, sclass, starid + '.txt')
				starid = -10000 - int(starid[9:])
				if not os.path.isfile(fpath_save):
					create_constant_star(fpath_save, random_state=rng)

			elif starid.startswith('fakerrlyr_'):
				fpath_save = os.path.join(output_dir, sclass, starid + '.txt')
				starid = -20000 - int(starid[10:])
			else:
				starid = int(starid)
				# The path to save the final timeseries:
				fname = 'kplr{starid:09d}-2011177032512_llc.fits'.format(starid=starid)
				fpath_save = os.path.join(output_dir, sclass, fname)

			if starid < 0:
				lc = io.load_lightcurve(fpath_save)
			elif os.path.isfile(fpath_save + '.gz'):
				# Load file that should already exist:
				lc = io.load_lightcurve(fpath_save + '.gz')
			else:
				subdir = os.path.join(thisdir, '{0:09d}'.format(starid)[:5])
				fpath = os.path.join(subdir, fname)

				# Check if the file is available as gzipped FITS:
				if os.path.isfile(fpath + '.gz'):
					fpath += '.gz'
					fpath_save += '.gz'

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

				# Save file:
				os.makedirs(os.path.dirname(fpath_save), exist_ok=True)
				shutil.copy(fpath, fpath_save)
				if not fpath_save.endswith('.gz'):
					subprocess.check_output(['gzip', fpath_save])

				# Load Kepler Q9 FITS file (PDC corrected):
				lc = io.load_lightcurve(fpath_save)

			#print(fpath)
			#lc.show_properties()
			#lc.plot()
			#plt.show()

			# Calculate diagnostics:
			variance = nanvar(lc.flux, ddof=1)
			rms_hour = util.rms_timescale(lc, timescale=3600/86400)
			ptp = util.ptp(lc)

			# Add target to TODO-list:
			diag.write("{variance:.16e},{rms_hour:.16e},{ptp:.16e}\n".format(
				variance=variance,
				rms_hour=rms_hour,
				ptp=ptp
			))

		diag.write("#-------------------------------------------\n")

	print("DONE")

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
from astropy.table import Table, Column
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
		raise RuntimeError("SIGMA not found")
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
def create_fake_rrlyr(fpath, timelen=90.0, sampling=30.0, random_state=None):

	# Directory where original simulations by Laszlo Molnar are stored:
	rootdir = r'I:\keplerq9\fake_cepheids_kepler-tess\fake_cepheids_kepler'

	# FIXME: How do these map to each other???
	identifier = os.path.basename(fpath).replace('.txt', '')
	identifier_laszlo = '202064435'

	timelen_name = str(timelen)
	sampling_name = str(sampling)
	if sampling == 30:
		sampling_name = '29.425'

	# Load the original data provided by Laszlo:
	fname = f'{identifier_laszlo:s}_{timelen_name:s}_{sampling_name:s}.txt'
	fpath_orig = os.path.join(rootdir, fname)
	data = np.genfromtxt(fpath_orig, delimiter=' ', comments='#', dtype='float64', encoding='utf-8')

	# Create lightcurve in correct units:
	time = data[:,0]
	m = np.nanmedian(data[:,1])
	flux = 1e6*(data[:,1]/m - 1)
	flux_err = flux

	# Save the lightcurve to file:
	os.makedirs(os.path.dirname(fpath), exist_ok=True)
	with open(fpath, 'w') as fid:
		fid.write("# Kepler Q9 Training Set Targets artificial RRLyr/Cepheid star\n")
		fid.write(f"# Identifier: {identifier:s}\n")
		fid.write("# Column 1: Time (days)\n")
		fid.write("# Column 2: Flux (ppm)\n")
		fid.write("# Column 3: Flux error (ppm)\n")
		fid.write("#-------------------------------------------\n")
		for k in range(len(time)):
			fid.write(f"{time[k]:.16e}  {flux[k]:.16e}  {flux_err[k]:.16e}\n")
		fid.write("#-------------------------------------------\n")

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	plots_interactive()

	# Standard "TESS-like" Kepler data:
	#output_dir = r'G:\keplerq9\keplerq9v3'  # Where output will be saved
	#timelen = 27.4
	#sampling = 30

	# Long (90 day) Kepler dataset primarily to be used with K2:
	#output_dir = r'I:\keplerq9\keplerq9v3-long'  # Where output will be saved
	output_dir = r'../starclass/training_sets/data/keplerq9v3-long'  # Where output will be saved
	timelen = 90
	sampling = 29.425

	# Seed the random state used below to ensure reproducability:
	rng = np.random.RandomState(42)

	# Directory where KeplerQ9 files are stored locally:
	thisdir = r'I:\keplerq9\full_fits'

	# Make sure directories exists:
	os.makedirs(output_dir, exist_ok=True)
	shutil.copy('keplerq9v3_targets.txt', os.path.join(output_dir, 'targets.ecsv'))

	# Load the list of KIC numbers and stellar classes:
	starlist = Table.read(os.path.join(output_dir, 'targets.ecsv'),
		format='ascii.csv',
		delimiter=',',
		comment='#',
		names=['starname','starclass'])

	starlist['starname'].description = 'Star identifier (mostly KIC number)'
	starlist['starclass'].description = 'Known stellar class'

	# Fix the header description:
	del starlist.meta['comments']
	starlist.meta['description'] = 'Kepler Q9 Training Set (version 3), 90-day edition'
	print(starlist)

	# Make sure that there are not duplicate entries:
	starlist_unique, cnt = np.unique(starlist, return_counts=True, axis=0)
	if len(starlist) != len(starlist_unique):
		print("Duplicate entries:")
		for line in starlist_unique[cnt > 1]:
			print('%s,%s' % tuple(line))

	# Make sure that there are not duplicate starids:
	us, cnt = np.unique(starlist['starname'], return_counts=True)
	if len(starlist) != len(us):
		print("Duplicate starids in multiple classes:")
		for s in us[cnt > 1]:
			indx = (starlist[:, 0] == s)
			for line in starlist[indx, :]:
				print('%s,%s' % tuple(line))

		raise RuntimeError("%d duplicate starids" % (len(starlist) - len(us)))

	# Just to catch the obvious problems:
	if sampling not in (30, 29.425):
		raise NotImplementedError("Only 'long cadence' kepler sampling is implemented")
	if timelen not in (27.4, 90):
		raise NotImplementedError("Time length is not supported")

	# Loop through the list of stars, load the data file, restructure the timeseries,
	# calculate diagnostics and save these to files:
	lightcurves = []
	col_variance = np.full(len(starlist), np.NaN, dtype='float64')
	col_rms_hour = np.full(len(starlist), np.NaN, dtype='float64')
	col_ptp = np.full(len(starlist), np.NaN, dtype='float64')
	for k, row in enumerate(tqdm(starlist)):

		starname = row['starname']
		sclass = row['starclass'].upper()

		if starname.startswith('constant_'):
			fpath_save = os.path.join(output_dir, sclass, starname + '.txt')
			starid = -10000 - int(starname[9:])
			if not os.path.isfile(fpath_save):
				create_constant_star(fpath_save, timelen=timelen, sampling=sampling, random_state=rng)

		elif starname.startswith('fakerrlyr_'):
			fpath_save = os.path.join(output_dir, sclass, starname + '.txt')
			starid = -20000 - int(starname[10:])
			if not os.path.isfile(fpath_save):
				create_fake_rrlyr(fpath_save, timelen=timelen, sampling=sampling, random_state=rng)

		else:
			starid = int(starname)
			# The path to save the final timeseries:
			fname = f'kplr{starid:09d}-2011177032512_llc.fits'
			fpath_save = os.path.join(output_dir, sclass, fname)

		if starid < 0:
			lc = io.load_lightcurve(fpath_save)
		elif os.path.isfile(fpath_save + '.gz'):
			# Load file that should already exist:
			lc = io.load_lightcurve(fpath_save + '.gz')
		else:
			subdir = os.path.join(thisdir, f'{starid:09d}'[:5])
			fpath = os.path.join(subdir, fname)

			# Check if the file is available as gzipped FITS:
			if os.path.isfile(fpath + '.gz'):
				fpath += '.gz'
				fpath_save += '.gz'

			# Print if file does not exist, instead of failing one
			# at a time. Making debugging a little easier:
			#tqdm.write(fpath)
			if not os.path.isfile(fpath):
				urlsubdir = f'{starid:09d}'[:4]
				url = f'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:Kepler/url/missions/kepler/lightcurves/{urlsubdir:s}/{starid:09d}/{fname:s}'
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

		# Relative path to lightcurve:
		lightcurves.append(os.path.relpath(fpath_save, output_dir).replace('\\', '/').replace('.fits', '.fits.gz'))

		# Calculate diagnostics:
		#col_variance[k] = nanvar(lc.flux, ddof=1)
		#col_rms_hour[k] = util.rms_timescale(lc, timescale=3600/86400)
		#col_ptp[k] = util.ptp(lc)

	col_lightcurve = Column(data=lightcurves, name='lightcurve', dtype='str', description='Relative path to lightcurve')
	col_variance = Column(data=col_variance, name='variance', dtype='float64', unit='ppm^2', description='Variance (ppm2)')
	col_rms_hour = Column(data=col_rms_hour, name='rms_hour', dtype='float64', unit='ppm^2 / hour', description='RMS per hour (ppm2/hour)')
	col_ptp = Column(data=col_ptp, name='ptp', dtype='float64', unit='ppm', description='Point-to-point scatter (ppm)')
	starlist.add_columns([col_lightcurve, col_variance, col_rms_hour, col_ptp])

	print(starlist)

	starlist.write(os.path.join(output_dir, 'targets.ecsv'),
		delimiter=',',
		overwrite=True,
		format='ascii.ecsv')

	print("DONE")

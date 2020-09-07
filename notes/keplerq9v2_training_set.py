#!/usr/bin/env python
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
from lightkurve import LightCurve
sys.path.insert(0, os.path.abspath('..'))
from starclass.utilities import rms_timescale
#from starclass.plots import plt

if __name__ == '__main__':
	# Directory where KeplerQ9 ASCII files are stored:
	# This can either contain all files in this directory,
	# or be divided into sub-directories.
	thisdir = r'E:\keplerq9\full'

	# Where output will be saved:
	output_dir = r'E:\keplerq9\keplerq9v2'

	# Make sure directories exists:
	os.makedirs(output_dir, exist_ok=True)
	shutil.copy('keplerq9v2_targets.txt', os.path.join(output_dir, 'targets.txt'))

	# Load the list of KIC numbers and stellar classes:
	starlist = np.genfromtxt(os.path.join(output_dir, 'targets.txt'),
		delimiter=',', comments='#', dtype=None, encoding='utf-8')

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
			if starid.startswith('constant_') or starid.startswith('fakerrlyr_'):
				fname_save = starid + '.txt'
				starid = -1
			else:
				starid = int(starid)
				fname_save = '{starid:09d}.txt'.format(starid=starid)

			# The path to save the final timeseries:
			fpath_save = os.path.join(output_dir, sclass, fname_save)
			#if os.path.exists(fpath_save):
			#	continue

			if starid == -1:
				# Load file that should already exist:
				data = np.loadtxt(fpath_save, usecols=(0,1,2))
			else:
				# Find the Kepler Q9 ASCII file for this target:
				fname = 'kplr{starid:09d}-2011177032512_llc.dat'.format(
					starid=starid
				)

				subdir = '{0:09d}'.format(starid)[:5]
				fpath = os.path.join(thisdir, subdir, fname)

				# If it doesn't exist in subdir (to not have too many files in one directory)
				# check of it is in the root dir instead:
				if not os.path.exists(fpath) and os.path.exists(os.path.join(thisdir, fname)):
					fpath = os.path.join(thisdir, fname)

				# Print if file does not exist, instead of failing one
				# at a time. Making debugging a little easier:
				if not os.path.isfile(fpath):
					# Check if the target was actually observed in Q9:
					r = requests.get('https://kasoc.phys.au.dk/catalog/sectors.php',
						params={'starid': starid})
					r.raise_for_status()
					if 'Q9' not in r.json():
						tqdm.write("Not observed in Q9: %d,%s" % (starid, sclass))
					else:
						tqdm.write("Data file does not exist: %d,%s" % (starid, sclass))

					continue

				# Load Kepler Q9 ASCII file (PDC corrected):
				data = np.loadtxt(fpath, usecols=(0,3,4), comments='#')

				# Remove anything with quality > 0:
				indx = ~np.isfinite(data[:,1]) | ~np.isfinite(data[:,2])
				data[indx, 1:3] = np.NaN
				data = data[1:, :] # Just removing the first data point as it is always NaN anyway

				# Subtract the first timestamp from all timestamps:
				data[:, 0] -= data[0, 0]

				# Only keep the first 27.4 days of data:
				data = data[data[:, 0] <= 27.4, :]

				# Convert to ppm:
				m = nanmedian(data[:,1])
				data[:,1] = 1e6*(data[:,1]/m - 1)
				data[:,2] = 1e6*data[:,2]/m

				#fig, ax = plt.subplots()
				#ax.plot(data[:,0], data[:,1])
				#fig.savefig(os.path.splitext(fpath_save)[0] + '.png', bbox_inches='tight')
				#plt.close(fig)

				# Save file:
				os.makedirs(os.path.dirname(fpath_save), exist_ok=True)
				np.savetxt(fpath_save, data, delimiter='  ', fmt=('%.8f', '%.16e', '%.16e'))

			# Calculate diagnostics:
			lc = LightCurve(time=data[:,0], flux=data[:,1], flux_err=data[:,2])
			variance = nanvar(data[:,1], ddof=1)
			rms_hour = rms_timescale(lc, timescale=3600/86400)
			ptp = nanmedian(np.abs(np.diff(data[:,1])))

			# Add target to TODO-list:
			diag.write("{variance:.16e},{rms_hour:.16e},{ptp:.16e}\n".format(
				variance=variance,
				rms_hour=rms_hour,
				ptp=ptp
			))

		diag.write("#-------------------------------------------\n")

	print("DONE")

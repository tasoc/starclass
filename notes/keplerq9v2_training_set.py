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
import tarfile
from bottleneck import nanmedian, nanvar
from tqdm import tqdm
from lightkurve import LightCurve
sys.path.insert(0, os.path.abspath('..'))
from starclass.utilities import rms_timescale
#from starclass.plots import plt

if __name__ == '__main__':
	# Directory where KeplerQ9 ASCII files are stored:
	# Thic can either contain all files in this directory,
	# or be divided into sub-directories.
	thisdir = r'E:\keplerq9\full'

	# Where output will be saved:
	output_dir = r'E:\keplerq9\keplerq9v2'

	# Make sure directories exists:
	os.makedirs(output_dir, exist_ok=True)
	shutil.copy('keplerq9v2_targets.txt', os.path.join(output_dir, 'targets.txt'))

	starlist = np.genfromtxt(os.path.join(output_dir, 'targets.txt'), delimiter=',', dtype=None, encoding='utf-8')

	starlist_unique, cnt = np.unique(starlist, return_counts=True, axis=0)
	if len(starlist) != len(starlist_unique):
		print("Duplicate entries:")
		for line in starlist_unique[cnt > 1]:
			print('%s,%s' % tuple(line))

	# Make sure that there are not duplicate starids:
	us, cnt = np.unique(starlist[:, 0], return_counts=True)
	if len(starlist) != len(us):
		print("Duplicate entries:")
		for s in us[cnt > 1]:
			indx = (starlist[:, 0] == s)
			for line in starlist[indx, :]:
				print('%s,%s' % tuple(line))

		raise Exception("%d duplicate starids" % (len(starlist) - len(us)))

	with open(os.path.join(output_dir, 'diagnostics.txt'), 'w') as diag:
		for starid, sclass in tqdm(starlist):
			sclass = sclass.upper()
			if starid.startswith('constant_') or starid.startswith('fakerrlyr_'):
				fname_save = starid + '.txt'
				starid = -1
			else:
				starid = int(starid)
				fname_save = '{starid:09d}.txt'.format(starid=starid)

			# The path to the
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

				# Move file into subdir, just to not have too many files in one directory:
				if not os.path.exists(fpath) and os.path.exists(os.path.join(thisdir, fname)):
					os.makedirs(os.path.join(thisdir, subdir), exist_ok=True)
					shutil.move(os.path.join(thisdir, fname), fpath)

				# Load Kepler Q9 ASCII file (PDC corrected):
				data = np.loadtxt(fpath, usecols=(0,3,4), comments='#')

				# Remove anything with quality > 0:
				indx = np.isfinite(data[:,1]) & np.isfinite(data[:,2])
				data = data[indx, :]

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
			lc = LightCurve(time=data[:,0], flux=data[:,1])
			variance = nanvar(data[:,1], ddof=1)
			rms_hour = rms_timescale(lc, timescale=3600/86400)
			ptp = nanmedian(np.abs(np.diff(data[:,1])))

			# Add target to TODO-list:
			diag.write("{variance:.16e},{rms_hour:.16e},{ptp:.16e}\n".format(
				variance=variance,
				rms_hour=rms_hour,
				ptp=ptp
			))

	print("DONE")

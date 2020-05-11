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
import matplotlib.pyplot as plt
# Load from photometry, since that avoids having to create a LightCurve object:
sys.path.insert(0, os.path.abspath('../../photometry'))
from photometry.utilities import rms_timescale

if __name__ == '__main__':
	#plt.switch_backend('Qt5Agg')

	# Directory where KeplerQ9 ASCII files are stored:
	thisdir = r'E:\keplerq9\full'

	# Make sure directories exists:
	os.makedirs('keplerq9v2', exist_ok=True)
	shutil.copy('keplerq9v2_targets.txt', 'keplerq9v2/targets.txt')

	starlist = np.genfromtxt('keplerq9v2/targets.txt', delimiter=',', dtype=None, encoding='utf-8')

	with open('keplerq9v2/diagnostics.txt', 'w') as diag:
		with tarfile.open(os.path.join(thisdir, 'Q09_public_ascii_20160905.tgz'), 'r') as tgz:
			for starid, sclass in tqdm(starlist):
				starid = int(starid)
				sclass = sclass.upper()

				# Find the Kepler Q9 ASCII file for this target:
				fname = 'kplr{starid:09d}-2011177032512_llc.dat'.format(
					starid=starid
				)
				fpath = os.path.join(thisdir, fname)
				print(fpath)

				if not os.path.exists(fpath):
					print("Extracting")
					tgz.extract(fname, fpath)

				# Load Kepler Q9 ASCII file (PDC corrected):
				data = np.loadtxt(fpath, usecols=(0,3,4))

				# Subtract the first timestamp from all timestamps:
				data[:, 0] -= data[0, 0]

				# Only keep the first 27.4 days of data:
				data = data[data[:, 0] <= 27.4, :]

				# Remove anything with quality > 0:
				indx = ~np.isfinite(data[:,1]) | ~np.isfinite(data[:,2])
				data[indx, 1:3] = np.NaN

				# Convert to ppm:
				m = nanmedian(data[:,1])
				data[:,1] = 1e6*(data[:,1]/m - 1)
				data[:,2] = 1e6*data[:,2]/m

				#plt.figure()
				#plt.plot(data[:,0], data[:,1])
				#plt.show()
				#sys.exit()

				# Save file:
				fpath = os.path.join('keplerq9v2', sclass, '{starid:09d}.txt'.format(starid=starid))
				os.makedirs(os.path.dirname(fpath), exist_ok=True)
				np.savetxt(fpath, data, delimiter='  ', fmt=('%.8f', '%.16e', '%.16e'))

				# Calculate diagnostics:
				variance = nanvar(data[:,1], ddof=1)
				rms_hour = rms_timescale(data[:,0], data[:,1], timescale=3600/86400)
				ptp = nanmedian(np.abs(np.diff(data[:,1])))

				# Add target to TODO-list:
				diag.write("{variance:e},{rms_hour:e},{ptp:e}\n".format(
					variance=variance,
					rms_hour=rms_hour,
					ptp=ptp
				))

	print("DONE")

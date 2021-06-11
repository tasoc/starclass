
import numpy as np
import os.path
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

files = glob.glob(r'G:\keplerq9\keplerq9v3-long\CONSTANT\constant_*.txt')
files = sorted(files, key=lambda x: int(os.path.basename(x).replace('constant_', '').replace('.txt', '')))

with open('keplerq9v3_targets_constant.txt', 'w') as fid:

	fid.write("# Kepler Q9 Training Set Targets (version 3)\n")
	fid.write("# Column 1: Constant star identifier\n")
	fid.write("# Column 2: Standard deviation (sigma)\n")
	fid.write("#-------------------------------------------\n")

	for fname in tqdm(files):

		data = np.loadtxt(fname, dtype='float64')

		#print(data)
		#sigma = np.std(data[:,1])
		flux_err = np.median(data[:,2])

		#print(sigma, flux_err)

		#plt.figure()
		#plt.scatter(data[:,0], data[:,1])
		#plt.show()

		fid.write("{0:s},{1:.16e}\n".format(
			os.path.basename(fname).replace('.txt', ''),
			flux_err
		))

	fid.write("#-------------------------------------------\n")

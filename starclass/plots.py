#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plotting utilities for stellar classification.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import matplotlib
matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
def plotConfMatrix(confmatrix, ticklabels, ax=None, cmap='Blues'):
	"""
	Plot a confusion matrix. Axes size and labels are hardwired.

	Parameters:
		cfmatrix (ndarray, [nobj x n_classes]): Confusion matrix.
		ticklabels (array, [n_classes]): labels for plot axes.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	confmatrix = np.asarray(confmatrix, dtype='float64')
	N = confmatrix.shape[0]
	norms = np.sum(confmatrix, axis=1)
	for i in range(N):
		if norms[i] > 0:
			confmatrix[i, :] /= norms[i]

	if ax is None:
		ax = plt.gca()

	ax.imshow(confmatrix, interpolation='nearest', origin='lower', cmap=cmap)

	for x in range(N):
		for y in range(N):
			if confmatrix[y,x] > 0.7:
				ax.text(x, y, "%d"%np.round(confmatrix[y,x]*100),
						 va='center', ha='center', color='w')
			elif confmatrix[y,x] < 0.01 and confmatrix[y,x] > 0:
				ax.text(x, y, "<1", va='center', ha='center')
			elif confmatrix[y,x] > 0:
				ax.text(x, y, "%d"%np.round(confmatrix[y,x]*100),
						 va='center', ha='center')

	for x in np.arange(confmatrix.shape[0]):
		ax.plot([x+0.5,x+0.5],[-0.5,N-0.5], ':', color='0.5', lw=0.5)
		ax.plot([-0.5,N-0.5],[x+0.5,x+0.5], ':', color='0.5', lw=0.5)

	ax.set_xlim(-0.5, N-0.5)
	ax.set_ylim(-0.5, N-0.5)
	ax.set_xlabel('Predicted Class')
	ax.set_ylabel('True Class')

	#class labels
	plt.xticks(np.arange(N), ticklabels, rotation='vertical')
	plt.yticks(np.arange(N), ticklabels)

if __name__ == '__main__':
	mat = np.identity(7)
	mat[2,3] = 0.5
	labels = ['test']*7

	plt.figure()
	plotConfMatrix(mat, labels)
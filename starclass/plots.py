#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plotting utilities for stellar classification.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Change to a non-GUI backend since this
# should be able to run on a cluster:
plt.switch_backend('Agg')

#--------------------------------------------------------------------------------------------------
def plots_interactive(backend=('Qt5Agg', 'MacOSX', 'Qt4Agg', 'Qt5Cairo', 'TkAgg')):
	"""
	Change plotting to using an interactive backend.

	Parameters:
		backend (str or list): Backend to change to. If not provided, will try different
			interactive backends and use the first one that works.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)
	logger.debug("Valid interactive backends: %s", matplotlib.rcsetup.interactive_bk)

	if isinstance(backend, str):
		backend = [backend]

	for bckend in backend:
		if bckend not in matplotlib.rcsetup.interactive_bk:
			logger.warning("Interactive backend '%s' is not found", bckend)
			continue

		# Try to change the backend, and catch errors
		# it it didn't work:
		try:
			plt.switch_backend(bckend)
		except (ModuleNotFoundError, ImportError):
			pass
		else:
			break

#--------------------------------------------------------------------------------------------------
def plots_noninteractive():
	"""
	Change plotting to using a non-interactive backend, which can e.g. be used on a cluster.
	Will set backend to 'Agg'.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	plt.switch_backend('Agg')

#--------------------------------------------------------------------------------------------------
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

	text_settings = {'va': 'center', 'ha': 'center', 'fontsize': 14}
	for x in range(N):
		for y in range(N):
			if confmatrix[y,x] > 0.7:
				ax.text(x, y, "%d" % np.round(confmatrix[y,x]*100), color='w', **text_settings)
			elif confmatrix[y,x] < 0.01 and confmatrix[y,x] > 0:
				ax.text(x, y, "<1", **text_settings)
			elif confmatrix[y,x] > 0:
				ax.text(x, y, "%d" % np.round(confmatrix[y,x]*100), **text_settings)

	for x in np.arange(confmatrix.shape[0]):
		ax.plot([x+0.5,x+0.5], [-0.5,N-0.5], ':', color='0.5', lw=0.5)
		ax.plot([-0.5,N-0.5], [x+0.5,x+0.5], ':', color='0.5', lw=0.5)

	ax.set_xlim(-0.5, N-0.5)
	ax.set_ylim(-0.5, N-0.5)
	ax.set_xlabel('Predicted Class', fontsize=18)
	ax.set_ylabel('True Class', fontsize=18)

	#class labels
	plt.xticks(np.arange(N), ticklabels, rotation='vertical')
	plt.yticks(np.arange(N), ticklabels)
	ax.tick_params(axis='both', which='major', labelsize=18)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plotting utilities for stellar classification.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import os.path
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator
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
def plotConfMatrix(confmatrix, ticklabels, ax=None, cmap='Blues', style=None):
	"""
	Plot a confusion matrix.

	Parameters:
		cfmatrix (ndarray, [nobj x n_classes]): Confusion matrix.
		ticklabels (array, [n_classes]): labels for plot axes.
		ax (:py:class:`matplotlib.pyplot.Axes`):
		cmap (str, optional):
		style (str, optional):

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
	if style is None:
		style = os.path.abspath(os.path.join(os.path.dirname(__file__), 'starclass.mplstyle'))

	with plt.style.context(style):
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

#--------------------------------------------------------------------------------------------------
def plot_roc_curve(diagnostics, ax=None, style=None):
	"""
	Plot Receiver Operating Characteristic (ROC) curve.

	Parameters:
		diagnostics (dict): Diagnostics coming from :py:func:`utilities.roc_curve` or saved to file
			during :py:func:`BaseClassifier.test`.
		ax (:py:class:`matplotlib.pyplot.Axes`):
		style (str, optional):

	See also:
		:py:func:`utilities.roc_curve`

	.. codeauthor:: Jeroen Audenaert <jeroen.audenaert@kuleuven.be>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	if ax is None:
		ax = plt.gca()
	if style is None:
		style = os.path.abspath(os.path.join(os.path.dirname(__file__), 'starclass.mplstyle'))

	# Pull things out from the diagnostics dict:
	fpr = diagnostics['false_positive_rate']
	tpr = diagnostics['true_positive_rate']
	roc_auc = diagnostics['roc_auc']
	idx = diagnostics['roc_threshold_index']
	classes = diagnostics['classes']

	with plt.style.context(style):

		# Reference line for a pure random classifier:
		ax.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')

		# Plot individual classes:
		lw = 1
		for ckey, cname in classes.items():
			ax.plot(fpr[ckey], tpr[ckey],
				label=f'{cname:s} (area = {roc_auc[ckey]:.4f})',
				lw=lw)
			ax.scatter(fpr[ckey][idx[ckey]], tpr[ckey][idx[ckey]], marker='o')

		ax.plot(fpr['micro'], tpr['micro'], lw=lw, label=f"micro avg (area = {roc_auc['micro']:.4f})")

		ax.set_xlim(-0.05, 1.05)
		ax.set_ylim(-0.05, 1.05)
		ax.set_xlabel('False Positive Rate')
		ax.set_ylabel('True Positive Rate')
		ax.set_title('ROC Curve - ' + diagnostics['classifier'] + ' - ' + diagnostics['tset'] + ' - ' + diagnostics['level'])
		ax.legend(loc="lower right")

		ax.xaxis.set_major_locator(MultipleLocator(0.1))
		ax.xaxis.set_minor_locator(MultipleLocator(0.05))
		ax.yaxis.set_major_locator(MultipleLocator(0.1))
		ax.yaxis.set_minor_locator(MultipleLocator(0.05))

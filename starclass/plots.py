#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities for stellar classification.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import warnings
import os.path
import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

with warnings.catch_warnings():
	warnings.filterwarnings('ignore', module='shap', message="IPython could not be loaded!")
	from shap import summary_plot

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
def plot_confusion_matrix(diagnostics=None, cfmatrix=None, ticklabels=None,
	ax=None, cmap='Blues', style=None):
	"""
	Plot a confusion matrix.

	If both ``diagnostics`` and ``cfmatrix`` or ``ticklabels`` are provided,
	the last two will take precedence.

	Parameters:
		diagnostics (dict, optional): Diagnostics to load confusion matrix from.
			Is created during testing and can be loaded from the diagnostics JSON files.
		cfmatrix (ndarray, [n_classes x n_classes]): Confusion matrix.
		ticklabels (list, [n_classes]): labels for plot axes.
		ax (:py:class:`matplotlib.pyplot.Axes`):
		cmap (str, optional):
		style (str, optional):

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)
	if diagnostics is None and cfmatrix is None:
		raise ValueError("One of DIAGNOSTICS or CFMATRIX must be provided.")

	# Pull things out from the diagnostics dict:
	if diagnostics:
		if cfmatrix is None:
			cfmatrix = diagnostics['confusion_matrix']
		if ticklabels is None:
			ticklabels = [s['value'] for s in diagnostics['classes']]

	cfmatrix = np.asarray(cfmatrix, dtype='float64')
	N = cfmatrix.shape[0]
	norms = np.sum(cfmatrix, axis=1)
	for i in range(N):
		if norms[i] > 0:
			cfmatrix[i, :] /= norms[i]

	# Warn if we don't have any labels to show, and create some dummy labels:
	if diagnostics is None and ticklabels is None:
		logger.warning("No class names were provided for confusion matrix. Assigning dummy-labels for plotting.")
		ticklabels = ['#{k:d}' for k in range(N)]

	if style is None:
		style = os.path.abspath(os.path.join(os.path.dirname(__file__), 'starclass.mplstyle'))

	with plt.style.context(style):
		if ax is None:
			fig, ax = plt.subplots()
		else:
			fig = ax.figure

		ax.imshow(cfmatrix, interpolation='nearest', origin='lower', cmap=cmap)

		text_settings = {'va': 'center', 'ha': 'center', 'fontsize': 14}
		for x in range(N):
			for y in range(N):
				if cfmatrix[y,x] > 0.7:
					ax.text(x, y, "%d" % np.round(cfmatrix[y,x]*100), color='w', **text_settings)
				elif cfmatrix[y,x] < 0.01 and cfmatrix[y,x] > 0:
					ax.text(x, y, "<1", **text_settings)
				elif cfmatrix[y,x] > 0:
					ax.text(x, y, "%d" % np.round(cfmatrix[y,x]*100), **text_settings)

		for x in np.arange(cfmatrix.shape[0]):
			ax.plot([x+0.5,x+0.5], [-0.5,N-0.5], ':', color='0.5', lw=0.5)
			ax.plot([-0.5,N-0.5], [x+0.5,x+0.5], ':', color='0.5', lw=0.5)

		ax.set_xlim(-0.5, N-0.5)
		ax.set_ylim(-0.5, N-0.5)
		ax.set_xlabel('Predicted Class', fontsize=18)
		ax.set_ylabel('True Class', fontsize=18)
		if diagnostics is not None:
			ax.set_title(diagnostics.get('classifier', '') + ' - ' + diagnostics.get('tset', '') + ' - ' + diagnostics.get('level', ''))

		# Class labels:
		plt.xticks(np.arange(N), ticklabels, rotation='vertical')
		plt.yticks(np.arange(N), ticklabels)
		ax.tick_params(axis='both', which='major', labelsize=18)

	return fig

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

	# Pull things out from the diagnostics dict:
	fpr = diagnostics['false_positive_rate']
	tpr = diagnostics['true_positive_rate']
	roc_auc = diagnostics['roc_auc']
	idx = diagnostics['roc_threshold_index']
	classes = diagnostics['classes']

	if style is None:
		style = os.path.abspath(os.path.join(os.path.dirname(__file__), 'starclass.mplstyle'))

	with plt.style.context(style):
		if ax is None:
			fig, ax = plt.subplots()
		else:
			fig = ax.figure

		# Reference line for a pure random classifier:
		ax.plot([0, 1], [0, 1], color='k', lw=0.5, linestyle='--')

		# Plot individual classes:
		lw = 1
		for c in classes:
			cname = c['name']
			cvalue = c['value']
			ax.plot(fpr[cname], tpr[cname],
				label=f'{cvalue:s} (area = {roc_auc[cname]:.4f})',
				lw=lw)
			ax.scatter(fpr[cname][idx[cname]], tpr[cname][idx[cname]], marker='o')

		ax.plot(fpr['micro'], tpr['micro'], lw=lw, label=f"micro avg (area = {roc_auc['micro']:.4f})")

		ax.set_xlim(-0.05, 1.05)
		ax.set_ylim(-0.05, 1.05)
		ax.set_xlabel('False Positive Rate')
		ax.set_ylabel('True Positive Rate')
		ax.set_title('ROC Curve - ' + diagnostics.get('classifier', '') + ' - ' + diagnostics.get('tset', '') + ' - ' + diagnostics.get('level', ''))
		ax.legend(loc="lower right")

		ax.xaxis.set_major_locator(MultipleLocator(0.1))
		ax.xaxis.set_minor_locator(MultipleLocator(0.05))
		ax.yaxis.set_major_locator(MultipleLocator(0.1))
		ax.yaxis.set_minor_locator(MultipleLocator(0.05))

	return fig

#--------------------------------------------------------------------------------------------------
def plot_feature_importance(shap_values, features, features_names, class_names, ax=None, style=None):

	if style is None:
		style = os.path.abspath(os.path.join(os.path.dirname(__file__), 'starclass.mplstyle'))

	with plt.style.context(style):

		summary_plot(shap_values,
			features=features,
			feature_names=features_names,
			class_names=class_names,
			max_display=len(features_names),
			plot_type='bar',
			show=False)

		# SHAP creates it's own figures, but doesn't return then,
		# so ask matplotlib what the latest figure is:
		fig = plt.gcf()
		ax = fig.axes[0]

		ax.spines['right'].set_visible(True)
		ax.spines['top'].set_visible(True)
		ax.spines['left'].set_visible(True)

	return fig

#--------------------------------------------------------------------------------------------------
def plot_feature_scatter_density(shap_values, features, features_names, class_name, ax=None, style=None):

	if style is None:
		style = os.path.abspath(os.path.join(os.path.dirname(__file__), 'starclass.mplstyle'))

	with plt.style.context(style):
		summary_plot(shap_values,
			features=features,
			feature_names=features_names,
			class_names=class_name,
			plot_type='dot',
			show=False)

		# SHAP creates it's own figures, but doesn't return then,
		# so ask matplotlib what the latest figure is:
		fig = plt.gcf()
		ax = fig.axes[0]

		ax.set_title(class_name)
		ax.spines['right'].set_visible(True)
		ax.spines['top'].set_visible(True)
		ax.spines['left'].set_visible(True)

	return fig

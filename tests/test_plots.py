#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of plotting utilities for stellar classification.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import numpy as np
import conftest # noqa: F401
from starclass import StellarClassesLevel1, io
from starclass.plots import plt, plot_confusion_matrix, plots_interactive, plot_roc_curve

#--------------------------------------------------------------------------------------------------
def test_plot_confusion_matrix():

	mat = np.identity(len(StellarClassesLevel1))
	mat[2,3] = 0.5
	mat[1,1] = 0.8
	mat[1,3] = 0.001
	labels = ['test']*mat.shape[0]

	# Putting into a diagnostics array:
	diagnostics = {
		'confusion_matrix': mat,
		'classes': [{'name': s.name, 'value': s.value} for s in StellarClassesLevel1]
	}

	# Providing matric and labels directly:
	plot_confusion_matrix(cfmatrix=mat, ticklabels=labels)

	# Produce identical plot with diagnostics info:
	plot_confusion_matrix(diagnostics)

	# Override the labels:
	plot_confusion_matrix(diagnostics, ticklabels=labels)

	# Load diagnostics from JSON file:
	diagnostics = io.loadJSON(os.path.join(os.path.dirname(__file__),
		'input', 'diagnostics', 'diagnostics_keplerq9v3_L1_xgb.json'))

	# Override the labels:
	plot_confusion_matrix(diagnostics)

#--------------------------------------------------------------------------------------------------
def test_plot_roc_curve():

	# Putting into a diagnostics array:
	diagnostics = io.loadJSON(os.path.join(os.path.dirname(__file__),
		'input', 'diagnostics', 'diagnostics_keplerq9v3_L1_xgb.json'))

	plot_roc_curve(diagnostics)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	plots_interactive()
	pytest.main([__file__])
	plt.show()

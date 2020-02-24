#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of plotting utilities for stellar classification.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from starclass.plots import plt, plotConfMatrix

def test_plots_confmatrix():

	mat = np.identity(7)
	mat[2,3] = 0.5
	mat[1,1] = 0.8
	mat[1,3] = 0.001
	labels = ['test']*7

	plt.figure()
	plotConfMatrix(mat, labels)

if __name__ == '__main__':
	plt.switch_backend('Qt5Agg')
	pytest.main([__file__])
	plt.show()

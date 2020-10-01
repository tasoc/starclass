#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source: Non-parametric Entropy Estimation Toolbox (NPEET) - https://github.com/gregversteeg/NPEET
Written by Greg Ver Steeg
Go to http://www.isi.edu/~gregv/npeet.html for documentation
"""

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree

#--------------------------------------------------------------------------------------------------
def entropy(x, k=3, base=2):
	"""
	The classic K-L k-nearest neighbor continuous entropy estimator
	x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
	if x is a one-dimensional scalar and we have four samples
	"""
	if k <= len(x) - 1:
		raise ValueError("Set k smaller than num. samples - 1")
	x = np.asarray(x)
	n_elements, n_features = x.shape
	x = add_noise(x)
	tree = build_tree(x)
	nn = query_neighbors(tree, x, k)
	const = digamma(n_elements) - digamma(k) + n_features * np.log(2)
	return (const + n_features * np.log(nn).mean()) / np.log(base)

#--------------------------------------------------------------------------------------------------
def centropy(x, y, k=3, base=2):
	""" The classic K-L k-nearest neighbor continuous entropy estimator for the
		entropy of X conditioned on Y.
	"""
	xy = np.c_[x, y]
	entropy_union_xy = entropy(xy, k=k, base=base)
	entropy_y = entropy(y, k=k, base=base)
	return entropy_union_xy - entropy_y

#--------------------------------------------------------------------------------------------------
def add_noise(x, intens=1e-10):
	# small noise to break degeneracy, see doc.
	return x + intens * np.random.random_sample(x.shape)

#--------------------------------------------------------------------------------------------------
def query_neighbors(tree, x, k):
	return tree.query(x, k=k + 1)[0][:, k]

#--------------------------------------------------------------------------------------------------
def count_neighbors(tree, x, r):
	return tree.query_radius(x, r, count_only=True)

#--------------------------------------------------------------------------------------------------
def avgdigamma(points, dvec):
	# This part finds number of neighbors in some radius in the marginal space
	# returns expectation value of <psi(nx)>
	tree = build_tree(points)
	dvec = dvec - 1e-15
	num_points = count_neighbors(tree, points, dvec)
	return np.mean(digamma(num_points))

#--------------------------------------------------------------------------------------------------
def build_tree(points):
	if points.shape[1] >= 20:
		return BallTree(points, metric='chebyshev')
	return KDTree(points, metric='chebyshev')

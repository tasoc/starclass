#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training set convenience functions.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
from . import training_sets as tsets
from . import RFGCClassifier, SLOSHClassifier, XGBClassifier, SortingHatClassifier, MetaClassifier

#--------------------------------------------------------------------------------------------------
def get_classifier(classifier_key):
	"""
	Get class for given classifier key.

	Parameters:
		classifier_key (str): Classifier keyword. Choices can be found in :func:`classifier_list`.

	Returns:
		:class:`BaseClassifier`: Class for the classifier.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	ClassificationClass = {
		'rfgc': RFGCClassifier,
		'slosh': SLOSHClassifier,
		#'foptics': FOPTICSClassifier,
		'xgb': XGBClassifier,
		'sortinghat': SortingHatClassifier,
		'meta': MetaClassifier
	}.get(classifier_key, None)

	if ClassificationClass is None:
		raise ValueError("Invalid classifier key specified")

	return ClassificationClass

#--------------------------------------------------------------------------------------------------
def get_trainingset(tset_key='keplerq9v3'):
	"""
	Get training set class for given training set key.

	Parameters:
		tset_key (str): Training set keyword. Choices can be found in :func:`trainingset_list`.

	Returns:
		:class:`TrainingSet`: Class for the training set.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	TsetClass = {
		'keplerq9v3': tsets.keplerq9v3,
		'keplerq9v3-instr': tsets.keplerq9v3_instr,
		'keplerq9v2': tsets.keplerq9v2,
		'keplerq9': tsets.keplerq9,
		'tdasim': tsets.tdasim,
		'tdasim-raw': tsets.tdasim_raw,
		'tdasim-clean': tsets.tdasim_clean,
		'testing': tsets.testing_tset
	}.get(tset_key, None)

	if TsetClass is None:
		raise ValueError("Invalid training set key specified")

	return TsetClass

#--------------------------------------------------------------------------------------------------
def trainingset_available(tset_key):
	"""
	Check if a training set is available, meaning that it has been downloaded and set up.

	Parameters:
		tset_key (str): Training set keyword. Choices can be found in :func:`trainingset_list`.

	Returns:
		bool: True if the trainingset is available.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	# Use the other function to ensure that tset_key is correct:
	tset = get_trainingset(tset_key)
	# Check if the todo.sqlite file has been created:
	return os.path.isfile(os.path.join(tset.find_input_folder(), tset._todo_name + '.sqlite'))

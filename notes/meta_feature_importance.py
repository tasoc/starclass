#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MetaClassifier feature importances.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import os.path
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import shap
if sys.path[0] != os.path.abspath('..'):
	sys.path.insert(0, os.path.abspath('..'))
import starclass
from starclass.plots import plt

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	with starclass.MetaClassifier() as meta:

		if not meta.classifier.trained:
			raise Exception("Not trained")

		feature_names = ['{0:s}_{1:s}'.format(classifier, stcl.name) for classifier, stcl in meta.features_used]
		class_names = np.unique(['{0:s}'.format(stcl.name) for classifier, stcl in meta.features_used])

		tsetclass = starclass.get_trainingset('keplerq9v2')
		tset = tsetclass()
		fitlabels = tset.labels()

		# Create table of features, just like it is done in the classifier:
		features = meta.build_features_table(tset.features(), total=len(tset))

		X_train, X_test, y_train, y_test = train_test_split(features, fitlabels, test_size=0.1, random_state=42)

		explainer = shap.TreeExplainer(meta.classifier)
		shap_values = explainer.shap_values(X_test)

		fig = shap.summary_plot(shap_values, X_test,
			feature_names=feature_names,
			class_names=class_names)
		fig.savefig('meta_feature_importance.png', bbox_inches='tight')

		for i, clsname in enumerate(class_names):
			fig = plt.figure(i)
			plt.title(clsname)
			shap.summary_plot(shap_values[i], X_test,
				feature_names=feature_names,
				class_names=clsname,
				plot_type="bar")
			fig.savefig('meta_feature_importance_' + clsname + '.png', bbox_inches='tight')

	plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The basic stellar classifier class for the TASOC pipeline.
All other specific stellar classification algorithms will inherit from BaseClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import os.path
import logging
import warnings
import traceback
import enum
from tqdm import tqdm
from sklearn import metrics
from bottleneck import nanvar
from timeit import default_timer

with warnings.catch_warnings():
	warnings.filterwarnings('ignore', module='shap', message="IPython could not be loaded!")
	import shap

from .features.freqextr import freqextr, freqextr_table_from_dict, freqextr_table_to_dict
from .features.fliper import FliPer
from .features.powerspectrum import powerspectrum
from .utilities import rms_timescale, ptp
from .plots import plt
from .StellarClasses import StellarClassesLevel1
from . import utilities, io, plots

__docformat__ = 'restructuredtext'

#--------------------------------------------------------------------------------------------------
@enum.unique
class STATUS(enum.Enum):
	"""
	Status indicator of the processing.
	"""
	UNKNOWN = 0 #: The status is unknown. The actual calculation has not started yet.
	STARTED = 6 #: The calculation has started, but not yet finished.
	OK = 1      #: Everything has gone well.
	ERROR = 2   #: Encountered a catastrophic error that I could not recover from.
	WARNING = 3 #: Something is a bit fishy. Maybe we should try again with a different algorithm?
	ABORT = 4   #: The calculation was aborted.
	SKIPPED = 5 #: The target was skipped because the algorithm found that to be the best solution.

#--------------------------------------------------------------------------------------------------
class BaseClassifier(object):
	"""
	The basic stellar classifier class for the TASOC pipeline.
	All other specific stellar classification algorithms will inherit from BaseClassifier.

	Attributes:
		plot (bool): Indicates wheter plotting is enabled.
		data_dir (str): Path to directory where classifiers store auxiliary data.
			Different directories will be used for each classification level.
		features_cache (str): Path to directory where calculated features will be
			saved/loaded as needed.
		classifier_key (str): Keyword/name of the current classifier.
		StellarClasses (:class:`enum.Enum`): Enum of all possible labels the classifier
			should be able to classify stars into. This will depend on the ``level``
			which the classifier is run on.
		features_names (list): List of names of features used by the classifier.
		truncate_lightcurves (bool): Indicating if Kepler/K2 lightcurves will be trunctated
			to 27.4 days when loaded. Default is to truncate lightcurves if running with short
			training sets (27.4 days) and not truncate if running with long (90 day) training-sets.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, tset=None, features_cache=None, plot=False, data_dir=None,
		truncate_lightcurves=None):
		"""
		Initialize the classifier object.

		Parameters:
			tset (:class:`TrainingSet`): From which training-set should the classifier be loaded?
			level (str, optional): Classification-level to load. Choices are ``'L1'`` and ``'L2'``.
				Default is ``'L1'``.
			features_cache (str, optional): Path to director where calculated features will be
				saved/loaded as needed.
			plot (bool, optional): Create plots as part of the output. Default is ``False``.
			data_dir (str):
			truncate_lightcurves (bool): Force truncation of lightcurves to 27.4 days.
				If ``None``, the default will be decided based on the training-set
				provided in ``tset``.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Start logger:
		self.logger = logging.getLogger(__name__)

		# Store the input:
		self.tset = tset
		self.plot = plot
		self.features_cache = features_cache
		self._random_seed = 2187
		self.truncate_lightcurves = truncate_lightcurves
		self.features_names = None
		self._classifier_model = None

		# Inherit settings from the Training Set, just as a conveience:
		if tset is None:
			self.logger.warning("BaseClassifier initialized without TrainingSet")
			self.StellarClasses = StellarClassesLevel1
			self.linfit = False
		else:
			self.StellarClasses = tset.StellarClasses
			self.linfit = tset.linfit

		# Decide if we should enable truncation of lightcurves upon loading:
		if self.truncate_lightcurves is None:
			if tset is None:
				self.truncate_lightcurves = False
			else:
				self.truncate_lightcurves = (not tset.key.startswith('keplerq9v3-long'))
		self.logger.debug("Truncate lightcurves = %s", self.truncate_lightcurves)

		# Set the data directory, where results (trained models) will be saved:
		if data_dir is None:
			data_dir = os.environ.get('STARCLASS_DATADIR',
				os.path.join(os.path.dirname(__file__), 'data'))

		self.data_dir = os.path.abspath(data_dir)
		if tset is not None:
			self.data_dir = os.path.join(data_dir, tset.level, tset.key)
			if tset.fold > 0:
				self.data_dir = os.path.join(self.data_dir, f'meta_fold{tset.fold:02d}')

		self.logger.debug("Data Directory: %s", self.data_dir)
		os.makedirs(self.data_dir, exist_ok=True)

		if self.features_cache is not None and not os.path.exists(self.features_cache):
			raise ValueError("features_cache directory does not exists")

		self.classifier_key = {
			'BaseClassifier': 'base',
			'RFGCClassifier': 'rfgc',
			'SLOSHClassifier': 'slosh',
			'XGBClassifier': 'xgb',
			'SortingHatClassifier': 'sortinghat',
			'MetaClassifier': 'meta'
		}[self.__class__.__name__]

		# If the training-set has fake_metaclassifier enabled, we change the
		# classifier key to the MetaClassifier, so we can load things in load_star
		# as if we are the MetaClassifier.
		if self.classifier_key == 'base' and tset is not None and tset.fake_metaclassifier:
			self.classifier_key = 'meta'

		# Just for catching all those places random numbers are used without explicitly requesting
		# a random_state:
		np.random.seed(self._random_seed)

	#----------------------------------------------------------------------------------------------
	def __enter__(self):
		return self

	#----------------------------------------------------------------------------------------------
	def __exit__(self, *args):
		self.close()

	#----------------------------------------------------------------------------------------------
	def close(self):
		"""Close the classifier."""
		pass

	#----------------------------------------------------------------------------------------------
	@property
	def random_seed(self):
		"""Random seed used in derived classifiers."""
		return self._random_seed

	#----------------------------------------------------------------------------------------------
	@property
	def random_state(self):
		"""Random state (:class:`numpy.random.RandomState`) corresponding to ``random_seed``."""
		return np.random.RandomState(self._random_seed)

	#----------------------------------------------------------------------------------------------
	@property
	def classifier_model(self):
		return self._classifier_model

	#----------------------------------------------------------------------------------------------
	def classify(self, task):
		"""
		Classify a star from the lightcurve and other features.

		Will run the :py:func:`do_classify` method and
		check some of the output and calculate various
		performance metrics.

		Parameters:
			features (dict): Dictionary of features, including the lightcurve itself.

		Returns:
			dict: Dictionary of classifications

		See Also:
			:py:func:`do_classify`, :py:func:`load_star`

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		result = task.copy()
		result.update({
			'tset': self.tset.key,
			'classifier': self.classifier_key
		})
		try:
			# Load the common features from the task information
			# and run the prediction/classification on the features:
			tic_predict = default_timer()
			features_common = self.load_star(task)
			res, features = self.do_classify(features_common)
			toc_predict = default_timer()

			# Basic checks of results:
			for key, value in res.items():
				if key not in self.StellarClasses:
					raise ValueError(f"Classifier returned unknown stellar class: '{key}'")
				if value < 0 or value > 1:
					raise ValueError("Classifier should return probability between 0 and 1.")

			# Remove complex or redundant features from common features:
			for rm in ('lightcurve', 'powerspectrum', 'frequencies', 'priority', 'starid', 'tmag', 'other_classifiers'):
				if rm in features_common:
					del features_common[rm]

			if self.features_names:
				# If needed, convert features to dictionary:
				if not isinstance(features, dict):
					if isinstance(features, np.ndarray):
						features = features.flatten()

					features = dict(zip(self.features_names, [float(feat) for feat in features]))

				# Remove features which are already in the common:
				features = {k: features[k] for k in set(features) - set(features_common)}
			else:
				features = None

			# Pad results with metadata:
			result.update({
				'starclass_results': res,
				'features_common': features_common,
				'features': features,
				'status': STATUS.OK,
				'elaptime': toc_predict - tic_predict
			})
		except (KeyboardInterrupt, SystemExit): # pragma: no cover
			result.update({
				'status': STATUS.ABORT
			})
		except: # noqa: E722, pragma: no cover
			# Something went wrong
			error_msg = traceback.format_exc().strip()
			result.update({
				'status': STATUS.ERROR,
				'details': {'errors': [error_msg]},
			})
			self.logger.exception("Classify failed: Priority '%s', Classifier '%s'.",
				task.get('priority'), self.classifier_key)

		return result

	#----------------------------------------------------------------------------------------------
	def do_classify(self, features):
		"""
		Classify a star from the lightcurve and other features.

		This method should be overwritten by child classes.

		Parameters:
			features (dict): Dictionary of features of star, including the lightcurve itself.

		Returns:
			dict: Dictionary where the keys should be from ``StellarClasses`` and the
			corresponding values indicate the probability of the star belonging to
			that class.

		Raises:
			NotImplementedError: If classifier has not implemented this subroutine.
		"""
		raise NotImplementedError()

	#----------------------------------------------------------------------------------------------
	def train(self, tset):
		"""
		Train classifier on training set.

		This method should be overwritten by child classes.

		Parameters:
			tset (:class:`TrainingSet`): Training-set to train classifier on.

		Raises:
			NotImplementedError: If classifier has not implemented this subroutine.
		"""
		raise NotImplementedError()

	#----------------------------------------------------------------------------------------------
	def test(self, tset, save=None, feature_importance=False):
		"""
		Test classifier using training-set, which has been created with a test-fraction.

		Parameters:
			tset (:class:`TrainingSet`): Training-set to run testing on.
			save (callable, optional): Function to call for saving test-predictions.
		"""

		# Start logger:
		tqdm_settings = {'disable': None if self.logger.isEnabledFor(logging.INFO) else True}

		# If the training-set is created with zero testfraction,
		# simply don't do anything:
		if tset.testfraction <= 0:
			self.logger.info("Test-fraction is zero, so no testing is performed.")
			return

		# All available labels in the current lavel (values):
		all_classes = [lbl.value for lbl in self.StellarClasses]

		# Classify test set (has to be one by one unless we change classifiers)
		N = len(tset.test_idx)
		Nclasses = len(all_classes)
		probs = np.full((N, Nclasses), np.NaN, dtype='float32')
		features = np.full((N, len(self.features_names)), np.NaN, dtype='float32')
		for k, task in enumerate(tqdm(tset.features_test(), total=N, **tqdm_settings)):

			# Classify this star from the test-set:
			result = self.classify(task)

			# All probabilities for each class:
			probs[k, :] = [result['starclass_results'].get(key, np.NaN) for key in self.StellarClasses]

			# Gather up features used for testing:
			if result['features'] is None:
				feat = result['features_common']
			else:
				feat = {**result['features_common'], **result['features']}
			features[k, :] = [feat[key] for key in self.features_names]

			# Save results for this classifier/trainingset in database:
			if save is not None:
				self.logger.debug(result)
				save(result)

		# Convert labels to ndarray:
		# FIXME: Only comparing to the first label
		y_pred = np.array(all_classes)[np.nanargmax(probs, axis=1)]
		labels_test = self.parse_labels(tset.labels_test())

		# Create dictionary which will gather all the diagnostics from the testing:
		diagnostics = {
			'tset': tset.key,
			'classifier': self.classifier_key,
			'level': tset.level,
			'classes': [{'name': s.name, 'value': s.value} for s in self.StellarClasses]
		}

		# Compare to known labels:
		# For some reason we have to call the function twice to generate both the dict and string
		# version of the report. Luckily this is not a demanding function to call.
		report = metrics.classification_report(
			labels_test,
			y_pred,
			labels=all_classes,
			target_names=[lbl.name for lbl in self.StellarClasses],
			output_dict=True,
			zero_division=0)
		diagnostics.update(report)
		if self.logger.isEnabledFor(logging.INFO):
			self.logger.info("Classification report:\n%s", metrics.classification_report(
				labels_test,
				y_pred,
				labels=all_classes,
				target_names=[lbl.name for lbl in self.StellarClasses],
				digits=4,
				output_dict=False,
				zero_division=0))

		# Confusion Matrix:
		self.logger.info('Calculating confusion matrix...')
		diagnostics['confusion_matrix'] = metrics.confusion_matrix(labels_test, y_pred, labels=all_classes)

		# Create plot of confusion matrix:
		fig = plots.plot_confusion_matrix(diagnostics=diagnostics)
		fig.savefig(os.path.join(self.data_dir, 'confusion_matrix_' + tset.key + '_' + tset.level + '_' + self.classifier_key + '.png'), bbox_inches='tight')
		plt.close(fig)

		# Prepare input for ROC/AUC
		self.logger.info('Calculating ROC curve...')
		diag_roc = utilities.roc_curve(labels_test, probs, self.StellarClasses)
		diagnostics.update(diag_roc)

		# Create plot of ROC curves:
		fig = plots.plot_roc_curve(diagnostics)
		fig.savefig(os.path.join(self.data_dir, 'roc_curve_' + tset.key + '_' + tset.level + '_' + self.classifier_key + '.png'), bbox_inches='tight')
		plt.close(fig)

		# Save test diagnostics:
		diagnostics_file = os.path.join(self.data_dir, 'diagnostics_' + tset.key + '_' + tset.level + '_' + self.classifier_key + '.json')
		io.saveJSON(diagnostics_file, diagnostics)

		# Call function-hook where individual classifiers can execute custom diagnostics:
		self.test_complete(tset=tset, features=features, probs=probs, diagnostics=diagnostics)

		# If we are asked to do so, calculate and plot the feature importances:
		if feature_importance:
			self.logger.info('Calculating feature importances...')
			if self.classifier_model is not None:
				with warnings.catch_warnings():
					# Ignore:
					#  - DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`.
					#  - DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`.
					warnings.filterwarnings('ignore', category=DeprecationWarning)
					explainer = shap.TreeExplainer(self.classifier_model)
					shap_values = explainer.shap_values(features)

				fig = plots.plot_feature_importance(shap_values, features, self.features_names, all_classes)
				fig.savefig(os.path.join(self.data_dir, 'feature_importance_' + tset.key + '_' + tset.level + '_' + self.classifier_key + '.png'), bbox_inches='tight')
				plt.close(fig)

				for k, cl in enumerate(self.StellarClasses):
					fig = plots.plot_feature_scatter_density(shap_values[k], features, self.features_names, cl.value)
					fig.savefig(os.path.join(self.data_dir, 'scatter_density_' + tset.key + '_' + tset.level + '_' + self.classifier_key + '_' + cl.name + '.png'), bbox_inches='tight')
					plt.close(fig)

			# Call function-hook where individual classifiers can execute custom diagnostics:
			self.feature_importance_complete(tset=tset, features=features, probs=probs, diagnostics=diagnostics)

	#----------------------------------------------------------------------------------------------
	def test_complete(self, tset=None, features=None, probs=None, diagnostics=None):
		"""
		Function which will be called when training is finishing.

		Parameters:
			tset:
			features:
			probs:
			diagnostics:

		See Also:
			:py:func:`BaseClassifier.train`

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		pass

	#----------------------------------------------------------------------------------------------
	def feature_importance_complete(self, tset=None, features=None, probs=None, diagnostics=None):
		"""
		Function which will be called when feature importance is finishing.

		Parameters:
			tset:
			features:
			probs:
			diagnostics:

		See Also:
			:py:func:`BaseClassifier.train`

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		pass

	#----------------------------------------------------------------------------------------------
	def load_star(self, task):
		"""
		Receive a task from the TaskManager, loads the lightcurve and returns derived features.

		Parameters:
			task (dict): Task dictionary as returned by :func:`TaskManager.get_task`.

		Returns:
			dict: Dictionary with features.

		See Also:
			:py:func:`TaskManager.get_task`

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Define variables used below:
		features = {}

		# The Meta-classifier is only using features from the other classifiers,
		# so there is no reason to load lightcurves and calculate/load any other classifiers:
		if self.classifier_key != 'meta':
			# Load features from cache file, or calculate them
			# and put them into cache file for other classifiers
			# to use later on:
			save_to_cache = False
			if self.features_cache:
				features_file = os.path.join(self.features_cache, 'features-' + str(task['priority']) + '.pickle')
				if os.path.exists(features_file):
					features = io.loadPickle(features_file)
				else:
					save_to_cache = True

			# Transfer cache of common features from task (MOAT):
			features.update(task.get('features_common', {}))

			# Transfer cache of features specific to this classifier from task (MOAT):
			features.update(task.get('features', {}))

			# Add the fields from the task to the list of features:
			for key in ('tmag', 'variance', 'rms_hour', 'ptp'):
				if key in task.keys():
					features[key] = task[key]
				else:
					self.logger.warning("Key '%s' not found in task.", key)
					features[key] = np.NaN

			# Load lightcurve file and create a TessLightCurve object:
			if 'lightcurve' in features:
				lightcurve = features['lightcurve']
			else:
				lightcurve = io.load_lightcurve(task['lightcurve'],
					starid=task['starid'],
					truncate_lightcurve=self.truncate_lightcurves)

				# Add the lightcurve as a seperate feature:
				features['lightcurve'] = lightcurve

			# Prepare lightcurve for power spectrum calculation:
			# NOTE: Lightcurves are now in relative flux (ppm) with zero mean!
			lc = lightcurve.remove_nans()

			if self.linfit:
				# Do a robust fitting with a first-order polynomial,
				# where we are catching cases where the fitting goes bad.
				indx = np.isfinite(lc.time) & np.isfinite(lc.flux) & np.isfinite(lc.flux_err)
				mintime = np.nanmin(lc.time[indx])
				with warnings.catch_warnings():
					warnings.filterwarnings('error', category=np.RankWarning)
					try:
						p = np.polyfit(lc.time[indx] - mintime, lc.flux[indx], 1, w=1/lc.flux_err[indx])
						lc -= np.polyval(p, lc.time - mintime)
					except np.RankWarning: # pragma: no cover
						self.logger.warning("Could not detrend light curve")
						p = np.array([0, 0])

				# Store the coefficients of the above detrending as a seperate feature:
				features['detrend_coeff'] = p

			# Calculate power spectrum:
			psd = features.get('powerspectrum')
			if psd is None:
				psd = powerspectrum(lc)

				# Save the entire power spectrum object in the features:
				features['powerspectrum'] = psd

			# Individual frequencies:
			if 'frequencies' not in features:
				if 'freq1' in features:
					# There is no frequency table, but individual keys,
					# so reconstruct the frequencies table from the features dict:
					features['frequencies'] = freqextr_table_from_dict(features, n_peaks=6, n_harmonics=5,
						flux_unit=lc.flux_unit)
				else:
					# Extract primary frequencies from lightcurve and add to features:
					features['frequencies'] = freqextr(lc, n_peaks=6, n_harmonics=5,
						Noptimize=5, devlim=None, initps=psd)

					# Add these for backward compatibility:
					features.update(freqextr_table_to_dict(features['frequencies']))

			# Calculate FliPer features:
			# TODO: Should these be done before or after linfit?
			#       Hopefully after, since otherwise we have to calculate another powerspectrum
			if 'Fp07' not in features:
				features.update(FliPer(psd))

			# If these features were not provided with the task, i.e. they
			# have not been pre-computed, we should compute them now:
			# Note we are using the un-corrected lightcurve here
			# since this will otherwise change from the values originally calculated from
			# the corrections pipeline, where the lightcurve was not detrended first.
			if features['variance'] is None or not np.isfinite(features['variance']):
				features['variance'] = nanvar(lightcurve.flux, ddof=1)
			if features['rms_hour'] is None or not np.isfinite(features['rms_hour']):
				features['rms_hour'] = rms_timescale(lightcurve)
			if features['ptp'] is None or not np.isfinite(features['ptp']):
				features['ptp'] = ptp(lightcurve)

			# Save features in cache file for later use:
			if save_to_cache:
				io.savePickle(features_file, features)

		else:
			# Add the results from other classifiers to the features:
			features['other_classifiers'] = task['other_classifiers']

		# Add the fields from the task to the list of features:
		features['priority'] = task['priority']
		features['starid'] = task['starid']

		self.logger.debug(features)
		return features

	#----------------------------------------------------------------------------------------------
	def parse_labels(self, labels):
		"""
		Convert iterator of labels into full numpy array, with only one label per star.

		TODO: How do we handle multiple labels better?
		"""
		fitlabels = []
		for lbl in labels:
			# Is it multi-labelled? In which case, what takes priority?
			# Priority order loosely based on signal clarity
			if len(lbl) > 1:
				if self.StellarClasses.ECLIPSE in lbl:
					fitlabels.append(self.StellarClasses.ECLIPSE.value)
				elif self.StellarClasses.RRLYR_CEPHEID in lbl:
					fitlabels.append(self.StellarClasses.RRLYR_CEPHEID.value)
				elif self.StellarClasses.CONTACT_ROT in lbl:
					fitlabels.append(self.StellarClasses.CONTACT_ROT.value)
				elif self.StellarClasses.DSCT_BCEP in lbl:
					fitlabels.append(self.StellarClasses.DSCT_BCEP.value)
				elif self.StellarClasses.GDOR_SPB in lbl:
					fitlabels.append(self.StellarClasses.GDOR_SPB.value)
				elif self.StellarClasses.SOLARLIKE in lbl:
					fitlabels.append(self.StellarClasses.SOLARLIKE.value)
				else:
					fitlabels.append(lbl[0].value)
			else:
				fitlabels.append(lbl[0].value)
		return np.array(fitlabels)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The basic stellar classifier class for the TASOC pipeline.
All other specific stellar classification algorithms will inherit from BaseClassifier.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import os.path
import logging
from astropy.units import cds
from lightkurve import TessLightCurve
from astropy.io import fits
from tqdm import tqdm
import enum
import warnings
from sklearn.metrics import accuracy_score, confusion_matrix
from .features.freqextr import freqextr
from .features.fliper import FliPer
from .features.powerspectrum import powerspectrum
from .utilities import savePickle, loadPickle
from .plots import plotConfMatrix, plt
from .StellarClasses import StellarClassesLevel1

__docformat__ = 'restructuredtext'

#--------------------------------------------------------------------------------------------------
@enum.unique
class STATUS(enum.Enum):
	"""
	Status indicator of the status of the photometry.
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
		classifier_key (str): Keyword/name of the current classifier.
		StellarClasses (:class:`enum.Enum`): Enum of all possible labels the classifier
			should be able to classify stars into. This will depend on the ``level``
			which the classifier is run on.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, tset=None, features_cache=None, plot=False, data_dir=None):
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

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Start logger:
		logger = logging.getLogger(__name__)

		# Store the input:
		self.tset = tset
		self.plot = plot
		self.features_cache = features_cache
		self._random_seed = 2187

		# Inherit settings from the Training Set, just as a conveience:
		if tset is None:
			logger.warning("BaseClassifier initialized without TrainingSet")
			self.StellarClasses = StellarClassesLevel1
			self.linfit = False
		else:
			self.StellarClasses = tset.StellarClasses
			self.linfit = tset.linfit

		# Set the data directory, where results (trained models) will be saved:
		if tset is not None:
			if data_dir is None:
				data_dir = tset.key
			if self.linfit:
				data_dir += '-linfit'
			self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', tset.level, data_dir))
		else:
			self.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

		logger.debug("Data Directory: %s", self.data_dir)
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
		return self._random_seed

	#----------------------------------------------------------------------------------------------
	@property
	def random_state(self):
		return np.random.RandomState(self._random_seed)

	#----------------------------------------------------------------------------------------------
	def classify(self, features):
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
			:py:func:`do_classify`

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		res = self.do_classify(features)
		# Check results
		for key, value in res.items():
			if key not in self.StellarClasses:
				raise ValueError("Classifier returned unknown stellar class: '%s'" % key)
			if value < 0 or value > 1:
				raise ValueError("Classifier should return probability between 0 and 1.")

		return res

	#----------------------------------------------------------------------------------------------
	def do_classify(self, features):
		"""
		Classify a star from the lightcurve and other features.

		This method should be overwritten by child classes.

		Parameters:
			features (dict): Dictionary of features of star, including the lightcurve itself.

		Returns:
			dict: Dictionary where the keys should be from :class:`StellarClasses` and the
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
	def test(self, tset, save=None):
		"""
		Test classifier using training-set, which has been created with a test-fraction.

		Parameters:
			tset (:class:`TrainingSet`): Training-set to run testing on.
			save (callable, optional): Function to call for saving test-predictions.
		"""

		# Start logger:
		logger = logging.getLogger(__name__)

		# If the training-set is created with zero testfraction,
		# simply don't do anything:
		if tset.testfraction <= 0:
			logger.info("Test-fraction is zero, so no testing is performed.")
			return

		# All available labels in the current lavel (values):
		all_classes = [lbl.value for lbl in self.StellarClasses]

		# Classify test set (has to be one by one unless we change classifiers)
		y_pred = []
		for features in tqdm(tset.features_test(), total=len(tset.test_idx)):
			# Create result-dict that is understood by the TaskManager:
			res = {
				'priority': features['priority'],
				'classifier': self.classifier_key,
				'status': STATUS.OK
			}

			# Classify this star from the test-set:
			res['starclass_results'] = self.classify(features)

			# FIXME: Only keeping the first label
			prediction = max(res['starclass_results'], key=lambda key: res['starclass_results'][key]).value
			y_pred.append(prediction)

			# Save results for this classifier/trainingset in database:
			if save is not None:
				logger.debug(res)
				save(res)

		# Convert labels to ndarray:
		# FIXME: Only comparing to the first label
		y_pred = np.array(y_pred)
		labels_test = self.parse_labels(tset.labels_test())

		# Compare to known labels:
		acc = accuracy_score(labels_test, y_pred)
		logger.info('Accuracy: %.2f%%', acc*100)

		# Confusion Matrix:
		cf = confusion_matrix(labels_test, y_pred, labels=all_classes)

		# Create plot of confusion matrix:
		fig = plt.figure(figsize=(12,12))
		plotConfMatrix(cf, all_classes)
		plt.title(self.classifier_key + ' - ' + tset.key + ' - ' + tset.level)
		fig.savefig(os.path.join(self.data_dir, 'confusion_matrix_' + tset.key + '_' + tset.level + '_' + self.classifier_key + '.png'), bbox_inches='tight')
		plt.close(fig)

	#----------------------------------------------------------------------------------------------
	def load_star(self, task, fname):
		"""
		Receive a task from the TaskManager, loads the lightcurve and returns derived features.

		Parameters:
			task (dict): Task dictionary as returned by :func:`TaskManager.get_task`.
			fname (str): Path to lightcurve file associated with task.

		Returns:
			dict: Dictionary with features.

		See Also:
			:py:func:`TaskManager.get_task`

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		# Define variables used below:
		features = {}
		save_to_cache = False

		# The Meta-classifier is only using features from the other classifiers,
		# so there is no reason to load lightcurves and calculate/load any other classifiers:
		if self.classifier_key != 'meta':
			# Load features from cache file, or calculate them
			# and put them into cache file for other classifiers
			# to use later on:
			if self.features_cache:
				features_file = os.path.join(self.features_cache, 'features-' + str(task['priority']) + '.pickle')
				if os.path.exists(features_file):
					features = loadPickle(features_file)

			# Load lightcurve file and create a TessLightCurve object:
			if 'lightcurve' in features:
				lightcurve = features['lightcurve']

			elif fname.endswith('.noisy') or fname.endswith('.sysnoise') or fname.endswith('.txt') or fname.endswith('.clean'):
				data = np.loadtxt(fname)
				if data.shape[1] == 4:
					quality = np.asarray(data[:,3], dtype='int32')
				else:
					quality = np.zeros(data.shape[0], dtype='int32')

				lightcurve = TessLightCurve(
					time=data[:,0],
					flux=data[:,1],
					flux_err=data[:,2],
					flux_unit=cds.ppm,
					quality=quality,
					time_format='jd',
					time_scale='tdb',
					targetid=task['starid'],
					quality_bitmask=2+8+256, # lightkurve.utils.TessQualityFlags.DEFAULT_BITMASK,
					meta={}
				)

			elif fname.endswith('.fits') or fname.endswith('.fits.gz'):
				with fits.open(fname, mode='readonly', memmap=True) as hdu:
					lightcurve = TessLightCurve(
						time=hdu['LIGHTCURVE'].data['TIME'],
						flux=hdu['LIGHTCURVE'].data['FLUX_CORR'],
						flux_err=hdu['LIGHTCURVE'].data['FLUX_CORR_ERR'],
						flux_unit=cds.ppm,
						centroid_col=hdu['LIGHTCURVE'].data['MOM_CENTR1'],
						centroid_row=hdu['LIGHTCURVE'].data['MOM_CENTR2'],
						quality=np.asarray(hdu['LIGHTCURVE'].data['QUALITY'], dtype='int32'),
						cadenceno=np.asarray(hdu['LIGHTCURVE'].data['CADENCENO'], dtype='int32'),
						time_format='btjd',
						time_scale='tdb',
						targetid=hdu[0].header.get('TICID'),
						label=hdu[0].header.get('OBJECT'),
						camera=hdu[0].header.get('CAMERA'),
						ccd=hdu[0].header.get('CCD'),
						sector=hdu[0].header.get('SECTOR'),
						ra=hdu[0].header.get('RA_OBJ'),
						dec=hdu[0].header.get('DEC_OBJ'),
						quality_bitmask=1+2+256, # CorrectorQualityFlags.DEFAULT_BITMASK
						meta={}
					)

			else:
				raise ValueError("Invalid file format")

			# No features found in cache, so calculate them:
			if not features:
				save_to_cache = True
				features = self.calc_features(lightcurve)

		# Save features in cache file for later use:
		if save_to_cache and self.features_cache:
			savePickle(features_file, features)

		# Add the fields from the task to the list of features:
		features['priority'] = task['priority']
		features['starid'] = task['starid']
		for key in ('tmag', 'variance', 'rms_hour', 'ptp', 'other_classifiers'):
			if key in task.keys():
				features[key] = task[key]
			else:
				logger.warning("Key '%s' not found in task.", key)
				features[key] = np.NaN

		logger.debug(features)
		return features

	#----------------------------------------------------------------------------------------------
	def calc_features(self, lightcurve):
		"""
		Calculate common derived features from the lightcurve.

		Parameters:
			lightcurve (:class:`TessLightCurve`): Lightcurve object to claculate features from.

		Returns:
			dict: Dictionary of features.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# We start out with an empty list of features:
		features = {}

		# Add the lightcurve as a seperate feature:
		features['lightcurve'] = lightcurve

		# Prepare lightcurve for power spectrum calculation:
		# NOTE: Lightcurves are now in relative flux (ppm) with zero mean!
		lc = lightcurve.remove_nans()
		#lc = lc.remove_outliers(5.0, stdfunc=mad_std) # Sigma clipping

		if self.linfit:
			# Do a robust fitting with a first-order polynomial,
			# where we are catching cases where the fitting goes bad.
			indx = np.isfinite(lc.time) & np.isfinite(lc.flux) & np.isfinite(lc.flux_err)
			with warnings.catch_warnings():
				warnings.filterwarnings('error', category=np.RankWarning)
				try:
					p = np.polyfit(lc.time[indx], lc.flux[indx], 1, w=1/lc.flux_err[indx])
					lc = lc - np.polyval(p, lc.time)
				except np.RankWarning:
					p = [0, 0]

			# Store the coefficients of the above detrending as a seperate feature:
			features['detrend_coeff'] = p

		# Calculate power spectrum:
		psd = powerspectrum(lc)

		# Save the entire power spectrum object in the features:
		features['powerspectrum'] = psd

		# Extract primary frequencies from lightcurve and add to features:
		features['frequencies'] = freqextr(lc, n_peaks=6, n_harmonics=5,
			Noptimize=5, devlim=None, initps=psd)

		# Add these for backward compatibility:
		for row in features['frequencies']:
			if row['harmonic'] == 0:
				key = '{0:d}'.format(row['num'])
			else:
				key = '{0:d}_harmonic{1:d}'.format(row['num'], row['harmonic'])

			features['freq' + key] = row['frequency']
			features['amp' + key] = row['amplitude']
			features['phase' + key] = row['phase']

		# Calculate FliPer features:
		features.update(FliPer(psd))

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

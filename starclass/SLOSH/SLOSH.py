#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The SLOSH method for detecting solar-like oscillations (2D deep learning methods).

.. codeauthor:: Marc Hon <mtyh555@uowmail.edu.au>
.. codeauthor:: James Kuszlewicz <kuszlewicz@mps.mpg.de>
"""

import numpy as np
import os.path
import logging
from tqdm import tqdm
import h5py
import tempfile
import tensorflow
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from . import SLOSH_prepro as preprocessing
from .. import BaseClassifier

#--------------------------------------------------------------------------------------------------
class SLOSHClassifier(BaseClassifier):
	"""
	Solar-like Oscillation Shape Hunter (SLOSH) Classifier.

	.. codeauthor:: Marc Hon <mtyh555@uowmail.edu.au>
	.. codeauthor:: James Kuszlewicz <kuszlewicz@mps.mpg.de>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, clfile='SLOSH_Classifier_Model.hdf5', mc_iterations=10, *args, **kwargs):
		"""
		Initialization for the class.

		:param saved_models: LIST of saved classifier filenames. Supports multi-classifier predictions.
		"""

		# Initialize parent:
		super().__init__(*args, **kwargs)

		logger = logging.getLogger(__name__)

		self.classifier_list = []
		#self.classifier = None
		self.mc_iterations = mc_iterations
		self.num_labels = len(self.StellarClasses)
		self.features_names = [] # SLOSH have no features as such

		# Set the global random seeds:
		np.random.seed(self.random_seed)
		tensorflow.random.set_seed(self.random_seed)

		# Find model file
		if clfile is not None:
			self.model_file = os.path.join(self.data_dir, clfile)
		else:
			self.model_file = None

		if self.model_file is not None and os.path.exists(self.model_file):
			logger.info("Loading pre-trained model...")
			# load pre-trained classifier
			self.predictable = True
			self.classifier_list.append(tensorflow.keras.models.load_model(self.model_file))
		else:
			logger.info('No saved models provided. Predict functions are disabled.')
			self.predictable = False

	#----------------------------------------------------------------------------------------------
	def do_classify(self, features):
		"""
		Prediction for a star, producing output determining if it is a solar-like oscillator.

		Parameters:
			features (dict): Dictionary of features.
				Of particular interest should be the `lightcurve` (``lightkurve.TessLightCurve`` object) and
				`powerspectum` which contains the lightcurve and power density spectrum respectively.

		Returns:
			dict: Dictionary of stellar classifications.
		"""
		logger = logging.getLogger(__name__)
		if not self.predictable:
			raise ValueError('No saved models provided. Predict functions are disabled.')

		# Pre-calculated power density spectrum:
		psd = features['powerspectrum'].standard

		logger.debug('Generating Image...')
		img_array = preprocessing.generate_single_image(psd[0], psd[1])
		img_array = img_array.reshape(1, 128, 128, 1)

		logger.debug('Making Predictions...')
		pred_array = np.zeros((self.mc_iterations, self.num_labels))
		for i in range(self.mc_iterations):
			pred_array[i, :] = self.classifier_list[0](img_array, training=False)
		pred = np.mean(pred_array, axis=0)

		# Convert the integer labels used by SLOSH to StellarClasses again
		# and put it all together in the result dict:
		result = {}
		for k, stcl in enumerate(self.StellarClasses):
			result[stcl] = pred[k]

		return result, []

	#----------------------------------------------------------------------------------------------
	def train(self, tset):
		"""
		Trains a fresh classifier using a default NN architecture and parameters as of the Hon et al. (2018) paper.

		Parameters:
			train_folder: The folder where training images are kept. These must be separated into
				subfolders by the image categories. For example:
				Train_Folder/1/ - Positive Detections; Train_Folder/0/ - Non-Detections
			features (iterator of dicts): Iterator of features-dictionaries similar to those
				in :meth:`do_classify`.
			labels (iterator of lists): For each feature, provides a list of the assigned known
				:class:`StellarClasses` identifications.

		Returns:
			model: A trained classifier model.
		"""

		logger = logging.getLogger(__name__)

		if self.predictable:
			return

		# Settings for progress bar used below:
		tqdm_settings = {
			'disable': not logger.isEnabledFor(logging.INFO)
		}
		dset_settings = {
			'compression': 'lzf',
			'shuffle': True,
			'fletcher32': True,
			'chunks': (128, 128),
			'dtype': 'float32'
		}

		# Convert classification labels to integers:
		intlookup = {key.value: value for value, key in enumerate(self.StellarClasses)}
		intlabels = [intlookup[lbl] for lbl in self.parse_labels(tset.labels())]

		logger.info('Generating Train Images...')
		if self.features_cache:
			train_folder = os.path.join(self.features_cache, 'SLOSH_Train_Images')
			os.makedirs(train_folder, exist_ok=True)
		else:
			tmpdir = tempfile.TemporaryDirectory()
			train_folder = tmpdir.name

		# Go through the training-set and ensure that all images are created:
		# Images are stored in a HDF5 file as individual datasets.
		hdf5_file = os.path.join(train_folder, 'SLOSH_Train_Images.hdf5')
		datasets = []
		with h5py.File(hdf5_file, 'a') as hdf:
			images = hdf.require_group('images')
			for feat in tqdm(tset.features(), total=len(tset), **tqdm_settings):
				dset_name = str(feat['priority'])
				datasets.append(dset_name)
				if dset_name not in images:
					# Power density spectrum from pre-calculated features:
					psd = feat['powerspectrum'].standard
					# Generate and save image to file:
					img = preprocessing.generate_single_image(psd[0], psd[1])
					images.create_dataset(dset_name, data=img, **dset_settings)
					hdf.flush()

		# Find the level of verbosity to add to tensorflow calls:
		if logger.isEnabledFor(logging.DEBUG):
			verbose = 2
		elif logger.isEnabledFor(logging.INFO):
			verbose = 1
		else:
			verbose = 0

		# Open the HDF5 file containing the features cache in read-only mode
		# so it can be passed to the generators:
		with h5py.File(hdf5_file, 'r') as hdf:

			train_generator = preprocessing.npy_generator(datasets, intlabels,
				hdf5_file=hdf, subset='train', random_seed=self.random_seed)
			valid_generator = preprocessing.npy_generator(datasets, intlabels,
				hdf5_file=hdf, subset='valid', random_seed=self.random_seed)

			reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, verbose=verbose)
			early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
			checkpoint = ModelCheckpoint(self.model_file, monitor='val_loss', verbose=verbose, save_best_only=True)
			#class_accuracy = TestCallback(valid_generator, classes=self.StellarClasses)

			model = preprocessing.default_classifier_model(num_classes=len(self.StellarClasses))

			logger.info('Training Classifier...')
			epochs = 50
			model.fit(train_generator, epochs=epochs, steps_per_epoch=len(train_generator),
				validation_data=valid_generator, validation_steps=len(valid_generator),
				callbacks=[reduce_lr, early_stop, checkpoint], verbose=verbose)

		# Save the model to file:
		self.save_model(model, self.model_file)

	#----------------------------------------------------------------------------------------------
	def save_model(self, model, model_file):
		'''
		Saves out trained model
		: param model: trained model
		: param model_file: Output file name for model
		:return: None
		'''
		# Save out model
		model.save(model_file)
		self.classifier_list = [model]
		# Set predictable to true so can predict
		self.predictable = True

	#----------------------------------------------------------------------------------------------
	def save(self, outfile):
		'''
		Saves all loaded classifier models.
		:param outfile: Base output file name
		:return: None
		'''
		if not self.predictable:
			raise ValueError('No saved models in memory.')
		else:
			for i in range(len(self.classifier_list)):
				self.classifier_list[i].save(outfile + '-%s.h5' % i)

	#----------------------------------------------------------------------------------------------
	def load(self, infile):
		'''
		Loads a classifier model and adds it to the list of classifiers.
		:param infile: Path to trained model
		:return: None
		'''
		self.classifier_list.append(tensorflow.keras.models.load_model(infile))
		self.predictable = True

	#----------------------------------------------------------------------------------------------
	def clear_model_list(self):
		'''
		Helper function to clear classifiers in the classifier list.
		:return: None
		'''
		del self.classifier_list[:]
		self.predictable = False

#--------------------------------------------------------------------------------------------------
class TestCallback(tensorflow.keras.callbacks.Callback):

	def __init__(self, val_data, classes):
		self.validation_data = val_data
		self.batch_size = 32
		self.num_classes = len(classes)
		self.class_names = [cl.name for cl in classes]

	def on_train_begin(self, logs={}):
		print(self.validation_data)
		#self.val_vals = []

	def on_epoch_end(self, epoch, logs={}):
		#batches = len(self.validation_data)
		#total = batches * self.batch_size

		val_pred = np.zeros((1, self.num_classes))
		val_true = np.zeros((1, self.num_classes))

		for xVal, yVal in self.validation_data:
			val_pred = np.vstack([val_pred, self.model.predict(xVal)])
			val_true = np.vstack([val_true, yVal])

		val_pred = np.argmax(val_pred[1:,:], axis=1)
		val_true = np.argmax(val_true[1:,:], axis=1)

		print(np.shape(val_pred), np.shape(val_true))
		#print(np.argmax(val_pred, axis=1))
		#print(np.argmax(val_true, axis=1))
		print(classification_report(val_true, val_pred,
			target_names=self.class_names))

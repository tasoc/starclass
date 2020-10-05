#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The SLOSH method for detecting solar-like oscillations (2D deep learning methods).

.. codeauthor:: Marc Hon <mtyh555@uowmail.edu.au>
.. codeauthor:: James Kuszlewicz <kuszlewicz@mps.mpg.de>
"""

import numpy as np
import os
import logging
import tensorflow
from sklearn.metrics import classification_report

from . import SLOSH_prepro as preprocessing
from .. import BaseClassifier, StellarClasses

#--------------------------------------------------------------------------------------------------
class SLOSHClassifier(BaseClassifier):
	"""
	Solar-like Oscillation Shape Hunter (SLOSH) Classifier

	.. codeauthor:: Marc Hon <mtyh555@uowmail.edu.au>
	.. codeauthor:: James Kuszlewicz <kuszlewicz@mps.mpg.de>
	"""

	def __init__(self, clfile='SLOSH_Classifier_Model.h5', mc_iterations=10, *args, **kwargs):
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
			tensorflow.keras.backend.set_learning_phase(1)
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
		assert self.predictable, 'No saved models provided. Predict functions are disabled.'

		# Pre-calculated power density spectrum:
		psd = features['powerspectrum'].standard

		logger.debug('Generating Image...')
		img_array = preprocessing.generate_single_image(psd[0], psd[1])
		logger.debug('Making Predictions...')
		pred_array = np.zeros((self.mc_iterations, 8))

		for i in range(self.mc_iterations):
			pred_array[i,:] = self.classifier_list[0].predict(img_array.reshape(1, 128, 128, 1))
		average_over_mc_iterations = np.mean(pred_array, axis=0)
		pred = average_over_mc_iterations

		# Must be a better way to do this!
		result = {}
		result[StellarClasses.RRLYR_CEPHEID] = pred[0]
		result[StellarClasses.APERIODIC] = pred[1]
		result[StellarClasses.CONSTANT] = pred[2]
		result[StellarClasses.CONTACT_ROT] = pred[3]
		result[StellarClasses.DSCT_BCEP] = pred[4]
		result[StellarClasses.GDOR_SPB] = pred[5]
		result[StellarClasses.SOLARLIKE] = pred[6]
		result[StellarClasses.ECLIPSE] = pred[7]

		return result

	#----------------------------------------------------------------------------------------------
	def train(self, tset):
		'''
		Trains a fresh classifier using a default NN architecture and parameters as of the Hon et al. (2018) paper.

		Parameters:
			train_folder: The folder where training images are kept. These must be separated into subfolders by the
				image categories. For example: Train_Folder/1/ - Positive Detections; Train_Folder/0/ - Non-Detections
			features (iterator of dicts): Iterator of features-dictionaries similar to those in ``do_classify``.
			labels (iterator of lists): For each feature, provides a list of the assigned known ``StellarClasses`` identifications.

		Returns:
			model: A trained classifier model.
		'''

		logger = logging.getLogger(__name__)

		if self.predictable:
			return

		train_folder = os.path.join(self.features_cache, 'SLOSH_Train_Images')
		if not os.path.exists(train_folder):
			os.makedirs(train_folder)
			logger.info('Generating Train Images...')

			for feat, lbl in zip(tset.features(), tset.labels()):
				# Power density spectrum from pre-calculated features:
				psd = feat['powerspectrum'].standard
				# Convert classifications to integer labels:
				if StellarClasses.RRLYR_CEPHEID in lbl:
					label = 0
				elif StellarClasses.APERIODIC in lbl:
					label = 1
				elif StellarClasses.CONSTANT in lbl:
					label = 2
				elif StellarClasses.CONTACT_ROT in lbl:
					label = 3
				elif StellarClasses.DSCT_BCEP in lbl:
					label = 4
				elif StellarClasses.GDOR_SPB in lbl:
					label = 5
				elif StellarClasses.SOLARLIKE in lbl:
					label = 6
				elif StellarClasses.ECLIPSE in lbl:
					label = 7
				else:
					logger.error("Label doesn't exist")
				preprocessing.generate_train_images(psd[0], psd[1],
					feat['priority'], output_path=train_folder, label=label)
		else:
			logger.info('Train Images exist...')

		reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
		early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
		checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(self.model_file,
			monitor='val_loss', verbose=1, save_best_only=True)

		#model = None
		model = preprocessing.default_classifier_model()

		train_generator = preprocessing.npy_generator(root=train_folder, batch_size=32,
			dim=(128,128), extension='.npz', subset='train')
		valid_generator = preprocessing.npy_generator(root=train_folder, batch_size=32,
			dim=(128,128), extension='.npz', subset='valid')

		#class_accuracy = TestCallback(valid_generator)
		logger.info('Training Classifier...')
		epochs = 50
		model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=len(train_generator),
			validation_data=valid_generator, validation_steps=len(valid_generator),
			callbacks=[reduce_lr, early_stop, checkpoint], verbose=2)
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
		tensorflow.keras.backend.set_learning_phase(1)
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

	def __init__(self, val_data):
		self.validation_data = val_data
		self.batch_size = 32

	def on_train_begin(self, logs={}):
		print(self.validation_data)
		#self.val_vals = []

	def on_epoch_end(self, epoch, logs={}):
		#batches = len(self.validation_data)
		#total = batches * self.batch_size

		#val_pred = np.zeros((total, 8))
		#val_true = np.zeros((total, 8))
		val_pred = np.zeros((1,8))
		val_true = np.zeros((1,8))

		i = 0
		for batch in self.validation_data:
			xVal, yVal = batch
			val_pred = np.vstack([val_pred, self.model.predict(xVal)])
			val_true = np.vstack([val_true, yVal])
			i += 1
		val_pred = np.argmax(val_pred[1:,:], axis=1)
		val_true = np.argmax(val_true[1:,:], axis=1)
		print(np.shape(val_pred), np.shape(val_true))
		#print(np.argmax(val_pred, axis=1))
		#print(np.argmax(val_true, axis=1))
		target_names = ['RRLyr/Cepheid', 'Aperiodic', 'Constant',
			'ContactBinary/Rotation', 'deltaSct/BetaCeph',
			'GammaDor/SPB', 'Solarlike', 'Eclipse']
		print(classification_report(val_true, val_pred,
			target_names=target_names))

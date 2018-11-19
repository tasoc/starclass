#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The SLOSH method for detecting solar-like oscillations (2D deep learning methods).

.. codeauthor::  Marc Hon <mtyh555@uowmail.edu.au>
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import os, re, math, logging
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from . import SLOSH_prepro as preprocessing
from .. import BaseClassifier, StellarClasses


class SLOSH_Classify(BaseClassifier):
    '''
    Solar-like Oscillation Shape Hunter (SLOSH) Classifier
    '''

    def __init__(self, saved_models=None, mc_iterations = 10,*args, **kwargs):
        '''
        Initialization for the class.
        :param saved_models: LIST of saved classifier filenames. Supports multi-classifier predictions.
        '''

        # Initialise parent
        super(self.__class__, self).__init__(*args, **kwargs)
        self.classifier_list = []
        self.mc_iterations = mc_iterations
        logger = logging.getLogger(__name__)


        if saved_models is not None:
            if not isinstance(saved_models, list):
                raise ValueError('Saved model input is not in the form of a list!')

            self.predictable = True
            K.set_learning_phase(1)
            for model in saved_models:
                if os.path.exists(os.path.join(self.data_dir,model)):
                    self.classifier_list.append(load_model(os.path.join(self.data_dir,model)))
        else:
            logger.warning('No saved models provided. Predict functions are disabled.')
            self.predictable = False

    def do_classify(self, features):
        '''
        Prediction for a star, producing output determining if it is a solar-like oscillator
        :param features (dict): Dictionary of features.
				Of particular interest should be the `lightcurve` (``lightkurve.TessLightCurve`` object) and
				`powerspectum` which contains the lightcurve and power density spectrum respectively.
		Returns:
			dict: Dictionary of stellar classifications..
        '''
        logger = logging.getLogger(__name__)
        assert self.predictable == True, 'No saved models provided. Predict functions are disabled.'
        preprocessing.generate_single_image(features['frequency'], features['power_density'], features['star_id'], output_path = os.getcwd(), label=None, numax=None)

        logger.info('Generating Image...')
        img_array = preprocessing.img_to_array(os.path.join(os.getcwd(), '/%s.png' %features['star_id']), normalize=True)

        logger.info('Making Predictions...')
        pred_array = np.zeros((mc_iterations, len(self.classifier_list)))
        for i in range(mc_iterations):
            for j in range(len(self.classifier_list)):
                prediction = self.classifier_list[j].predict(img_array.reshape(1, 128, 128, 1))
                try:  # some models have 2 output neurons instead of 1
                    pred_array[i, j] = prediction[:, 1]
                except:
                    pred_array[i, j] = prediction[:]
        average_over_models = np.mean(pred_array, axis=1)
        std_over_models = np.std(pred_array, axis=1)
        average_over_mc_iterations = np.mean(average_over_models)
        std_over_mc_iterations = 0

        for i in range(len(std_over_models)):
            std_over_mc_iterations += std_over_models[i] ** 2
        std_over_mc_iterations = np.sqrt(std_over_mc_iterations)

        pred = average_over_mc_iterations
        if pred >= 0.5:
            label = 1
        else:
            label = 0
        pred_sigma = std_over_mc_iterations

        result = {StellarClasses.SOLARLIKE: pred}

        return result

    def train(self, features, labels):
        '''
        Trains a fresh classifier using a default NN architecture and parameters as of the Hon et al. (2018) paper.
        :param train_folder: The folder where training images are kept. These must be separated into subfolders by the
        image categories. For example: Train_Folder/1/ - Positive Detections; Train_Folder/0/ - Non-Detections
        :param features (iterator of dicts): Iterator of features-dictionaries similar to those in ``do_classify``.
        :param labels (iterator of lists): For each feature, provides a list of the assigned known ``StellarClasses`` identifications.
        :return: model: A trained classifier model
        '''

        logger = logging.getLogger(__name__)
        train_folder = os.getcwd() + '/Train_Images/'
        if not os.path.exists(train_folder):
            os.mkdir(train_folder)

        logger.info('Generating Train Images...')
        for i in range(len(features)):
            preprocessing.generate_single_image(features[i]['frequency'], features[i]['power_density'],
                                                features[i]['star_id'],
                                                output_path=train_folder, label=features[i]['label'], numax=None)

        from keras.callbacks import ReduceLROnPlateau
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
        model = preprocessing.default_classifier_model()

        nb_files = 0
        for dirpath, dirnames, filenames in os.walk(os.getcwd() + '/Train_Images/'):
            for i in range(len(filenames)):
                nb_files += 1

        datagen = ImageDataGenerator(rescale=1. / 255., height_shift_range=0.15)
        train_generator = datagen.flow_from_directory(train_folder, target_size=(128, 128), color_mode='grayscale',
                                                      class_mode='categorical', batch_size=32)
        logger.info('Training Classifier...')
        model.fit_generator(train_generator, epochs=200, steps_per_epoch=math.ceil(nb_files / 32),
                            callbacks=[reduce_lr], verbose=2)
        model.save('SLOSH_Classifier_Model.h5')


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
                self.classifier_list[i].save(outfile+'-%s.h5'%i)

    def load(self, infile):
        '''
        Loads a classifier model and adds it to the list of classifiers.
        :param infile: Path to trained model
        :return: None
        '''
        K.set_learning_phase(1)
        self.classifier_list.append(load_model(infile))
        self.predictable = True

    def clear_model_list(self):
        '''
        Helper function to clear classifiers in the classifier list.
        :return: None
        '''
        del self.classifier_list[:]
        self.predictable = False
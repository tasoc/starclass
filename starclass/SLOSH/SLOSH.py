#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The SLOSH method for detecting solar-like oscillations (2D deep learning methods).

.. codeauthor::  Marc Hon <mtyh555@uowmail.edu.au>
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os, re, warnings
from keras import backend as K
from keras.models import load_model, save_model
from PIL import Image as pil_image
from keras.preprocessing.image import ImageDataGenerator
from . import SLOSH_prepro as preprocessing
from .. import BaseClassifier, StellarClasses


class SLOSH_Classify(BaseClassifier):
    '''
    Solar-like Oscillation Shape Hunter (SLOSH) Classifier
    '''

    def __init__(self, saved_models=None, *args, **kwargs):
        '''
        Initialization for the class.
        :param saved_models: LIST of saved classifier filenames. Supports multi-classifier predictions.
        '''

        # Initialise parent
        super(self.__class__, self).__init__(*args, **kwargs)

        if saved_models is not None:
            if not isinstance(saved_models, list):
                raise ValueError('Saved model input is not in the form of a list!')

            self.predictable = True
            self.classifier_list = []
            for model in saved_models:
                if os.path.exists(os.path.join(self.data_dir,model)):
                    self.classifier_list.append(load_model(os.path.join(self.data_dir,model)))
        else:
            warnings.warn('No saved models provided. Predict functions are disabled.')
            self.predictable = False

    def predict(self, batch, target_path, mc_iterations=10):
        '''
        Prediction for a star, producing output determining if it is a solar-like oscillator
        :param batch: String, 'single' - Prediction on a single image; 'folder' - Multiple predictions where
        the images should be placed in a folder.
        :param target_path: String, for batch = 'single' this should be an image path; for batch = 'folder' this should
         be a folder path.
        :param mc_iterations: Number of repetitions for Monte Carlo Dropout.
        :return: A 4 x N array of predictions with columns [Star_ID, Label, Probability, Prob. Sigma] where N is the
        number of predicted targets.
        '''
        assert self.predictable == True, 'No saved models provided. Predict functions are disabled.'
        assert batch in ('single', 'folder'), "batch parameter should be either 'single' or 'folder"
        K.set_learning_phase(1)

        if batch == 'single':
            if not os.path.exists(target_path):
                raise ValueError('Target path does not exist!')
            img = pil_image.open(os.path.join(self.data_dir, target_path)).convert('L')
            img = img.resize((128, 128), pil_image.NEAREST)
            img_array = np.array(img,dtype=K.floatx())
            img_array *= 1 / 255.0

            pred_array = np.zeros((mc_iterations, len(self.classifier_list)))
            for i in range(mc_iterations):
                for j in range(len(self.classifier_list)):
                    pred_array[i,j] = self.classifier_list[j].predict(img_array.reshape(1,128,128,1))[:,1]
            average_over_models = np.mean(pred_array, axis=1)
            std_over_models = np.std(pred_array,axis=1)
            average_over_mc_iterations = np.mean(average_over_models)
            std_over_mc_iterations = np.zeros(len(average_over_mc_iterations))

            for i in range(len(std_over_mc_iterations)):
                std_over_mc_iterations += std_over_models[i] ** 2
            std_over_mc_iterations = np.sqrt(std_over_mc_iterations)

            pred = average_over_mc_iterations
            if pred >= 0.5:
                label = 1
            else:
                label = 0
            pred_sigma = std_over_mc_iterations
            file_id = int(re.search(r'\d+', target_path).group())
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The SLOSH method for detecting solar-like oscillations (2D deep learning methods).

.. codeauthor::  Marc Hon <mtyh555@uowmail.edu.au>
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import os, re, warnings, math
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from . import SLOSH_prepro as preprocessing
from .. import BaseClassifier


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

    def predict(self, batch, target_path, im_array=None, mc_iterations=10):
        '''
        Prediction for a star, producing output determining if it is a solar-like oscillator
        :param batch: String, 'single' - Prediction on a single image from path, or an image array
        ; 'folder' - Multiple predictions where the images should be placed in a folder.
        :param target_path: String, for batch = 'single' this should be an image path; for batch = 'folder' this should
         be a folder path.
        :param im_array: For 'single' batch only. Instead of loading a single image from path, you can predict directly
        from an array of grayscale 2D pixel values
        :param mc_iterations: Number of repetitions for Monte Carlo Dropout.
        :return: file_id - Array of Star ID; Label - Classification (1 is positive, 0 is negative); Pred- Classification
         probabilities; pred_sigma - Standard deviation of probabilities. Each is array of length N, where N is number
         of input stars.
        '''
        assert self.predictable == True, 'No saved models provided. Predict functions are disabled.'
        assert batch in ('single', 'folder'), "batch parameter should be either 'single' or 'folder"
        K.set_learning_phase(1)

        if batch == 'single':
            if im_array is not None:
                assert im_array.shape == (128,128), 'Improper image array shape'
                img_array = np.array(im_array, dtype=K.floatx())/255.
            else:
                img_array = preprocessing.img_to_array(os.path.join(self.data_dir, target_path), normalize=True)

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

        elif batch == 'folder':
            if not os.path.isdir(os.path.join(self.data_dir, target_path)):
                raise ValueError('Target path does not exist!')
            nb_files = 0
            for cdirpath, cdirnames, pfilenames in os.walk(os.path.join(self.data_dir, target_path)):
                for p in range(len(pfilenames)):
                    nb_files += 1
            im_gen = ImageDataGenerator(rescale=1. / 255)
            im_gen_flow = im_gen.flow_from_directory(os.path.join(self.data_dir, target_path), target_size=(128, 128)
                                                     , color_mode='grayscale', class_mode=None,
                                                     batch_size=32, shuffle=False)
            file_id = []
            for star_id in im_gen_flow.filenames:
                file_id.append(int(re.search(r'\d+', star_id[2:]).group()))

            pred_array = np.zeros(shape=(nb_files, mc_iterations, len(self.classifier_list)))
            for i in range(mc_iterations):
                for j in range(len(self.classifier_list)):
                    pred_array[:,i,j] = self.classifier_list[j].predict_generator(im_gen_flow, steps=nb_files / 32,
                                                                                  pickle_safe=True, verbose=1)[:,1]
            average_over_models = np.mean(pred_array, axis=1)
            std_over_models = np.std(pred_array,axis=1)
            average_over_mc_iterations = np.mean(average_over_models, axis=1)
            std_over_mc_iterations = np.zeros(len(average_over_mc_iterations))

            for i in range(std_over_mc_iterations.shape[1]):
                std_over_mc_iterations += std_over_models[i] ** 2
            std_over_mc_iterations = np.sqrt(std_over_mc_iterations)

            pred = average_over_mc_iterations
            if pred >= 0.5:
                label = 1
            else:
                label = 0
            pred_sigma = std_over_mc_iterations
            file_id = np.array(file_id)
        else:
            raise ValueError("batch parameter should be either 'single' or 'folder'")

        pred = average_over_models

        return file_id,label,pred,pred_sigma

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
        self.classifier_list.append(load_model(infile))
        self.predictable = True

    def clear_model_list(self):
        '''
        Helper function to clear classifiers in the classifier list.
        :return: None
        '''
        del self.classifier_list[:]

    def create_single_image(self, freq, power,star_id, out_path, label=None, numax=None):
        '''
         Creates a 2D grayscale image from the PSD of a single star and saves it to a folder.
         :param freq: Array of frequency values for the PSD
         :param power: Array of power values for the PSD
         :param star_id: The identifier of the star for file naming purposes
         :param output_path: Image output path
         :param label: Classification label for star for classifier training,
          typically 0 for nondet and 1 for positive detection
         :param numax: Numax value for star for regressor training
         :return: None
         '''
        preprocessing.generate_single_image(freq,power,star_id,out_path,label,numax)

    def create_batch_images(self, input_folder_path, output_folder_path, star_list=None, label_list=None,
                            numax_list=None):
        '''
        The batch version of create_single_image. Instead of passing single freq and array arrays, a folder with psd
        files is given. This saves all images to a folder, instead of returning an image array.
        :param input_folder_path: The folder containing all the PSD
        :param output_folder_path: The folder to contain images of the PSD
        :param star_list: For generating images for a training set, a list to cross-match with known parameters
        :param label_list: Ground truth detection values for classifier training set creation
        :param numax_list: Ground truth numax values for regressor training set creation
        :return: None
        '''
        preprocessing.generate_images(input_folder_path, output_folder_path, star_list, label_list, numax_list)


    def train_classifier(self, train_folder, validation_split=None):
        '''
        Trains a fresh classifier using a default NN architecture and parameters as of the Hon et al. (2018) paper.
        :param train_folder: The folder where training images are kept. These must be separated into subfolders by the
        image categories. For example: Train_Folder/1/ - Positive Detections; Train_Folder/0/ - Non-Detections
        :param validation_split: Fraction of training set to use as validation, from 0 to 1 if not None
        :return: model: A trained classifier model
        '''
        from keras.callbacks import ReduceLROnPlateau
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
        model = preprocessing.default_classifier_model()

        nb_files = 0
        for dirpath, dirnames, filenames in os.walk(train_folder):
            for i in range(len(filenames)):
                nb_files += 1

        if validation_split is not None:
            datagen = ImageDataGenerator(rescale=1. / 255., height_shift_range=0.15, validation_split=validation_split)
            train_generator = datagen.flow_from_directory(train_folder, target_size=(128, 128),color_mode='grayscale',
                                        class_mode='categorical', batch_size=32, subset='training')
            val_generator = datagen.flow_from_directory(train_folder, target_size=(128, 128), color_mode='grayscale',
                                                       class_mode='categorical', batch_size=32, subset='validation')
            model.fit_generator(train_generator, epochs=200, steps_per_epoch=math.ceil((1-validation_split)*nb_files / 32),
                                validation_data=val_generator, validation_steps=math.ceil(validation_split*nb_files / 32),
                                callbacks=[reduce_lr])

        else:
            datagen = ImageDataGenerator(rescale=1. / 255., height_shift_range=0.15)
            train_generator = datagen.flow_from_directory(train_folder, target_size=(128, 128),color_mode='grayscale',
                                        class_mode='categorical', batch_size=32)
            model.fit_generator(train_generator, epochs=200, steps_per_epoch=math.ceil((1-validation_split)*nb_files / 32),
                                callbacks=[reduce_lr])

        return model


class SLOSH_Regressor(BaseClassifier):
    '''
        Solar-like Oscillation Shape Hunter (SLOSH) Regressor
        '''

    def __init__(self, saved_model=None, aleatoric=False, *args, **kwargs):
        '''
        Initialization for the class. Currently the use of only one regressor is supported, with multiple to be
        supported in future!
        :param saved_models: LIST of saved classifier filenames. Supports multi-classifier predictions.
        :param aleatoric: Boolean flag. If true, uses prototype models for heteroscedatic noise estimates
        '''

        # Initialise parent
        super(self.__class__, self).__init__(*args, **kwargs)
        self.aleatoric = aleatoric
        if saved_model is not None:
            if os.path.exists(os.path.join(self.data_dir, saved_model)):
                if self.aleatoric:
                    self.regressor_model = load_model(os.path.join(self.data_dir, saved_model),
                                                      custom_objects=
                                                      {'weighted_mean_squared_error':preprocessing.weighted_mean_squared_error,
                                                       'aleatoric_loss': preprocessing.aleatoric_loss})
                else:
                    self.regressor_model = load_model(os.path.join(self.data_dir, saved_model),
                                                      custom_objects=
                                                      {'weighted_mean_squared_error':preprocessing.weighted_mean_squared_error})
                self.predictable = True
        else:
            warnings.warn('No saved models provided. Predict functions are disabled.')
            self.predictable = False

    def predict(self, batch, target_path, im_array=None, mc_iterations=10):
        '''
        Prediction for a star, producing numax estimate outputs
        :param batch: String, 'single' - Prediction on a single image from path, or an image array
        ; 'folder' - Multiple predictions where the images should be placed in a folder.
        :param target_path: String, for batch = 'single' this should be an image path; for batch = 'folder' this should
         be a folder path.
        :param im_array: For 'single' batch only. Instead of loading a single image from path, you can predict directly
        from an array of grayscale 2D pixel values
        :param mc_iterations: Number of repetitions for Monte Carlo Dropout.
        :return: file_id - Array of Star ID; numax- Prediction of frequency at maximum power ; numax_sigma - Standard
         deviation of probabilities. Each is array of length N, where N is number of input stars.
        '''
        assert self.predictable == True, 'No saved models provided. Predict functions are disabled.'
        assert batch in ('single', 'folder'), "batch parameter should be either 'single' or 'folder"
        K.set_learning_phase(1)
        conversion_a = 3 # constants for conversion from pixel coordinate to frequency in uHz
        conversion_b = (1. / 128.) * np.log(283. / 3.)

        if batch == 'single':
            if im_array is not None:
                assert im_array.shape == (128, 128), 'Improper image array shape'
                img_array = np.array(im_array, dtype=K.floatx()) / 255.
            else:
                img_array = preprocessing.img_to_array(os.path.join(self.data_dir, target_path), normalize=True)

            if self.aleatoric:
                pred_array = np.zeros(mc_iterations)
                var_array = np.zeros(mc_iterations)
                for i in range(mc_iterations):
                    _, pixel_var = self.regressor_model.predict(img_array.reshape(1, 128, 128, 1))
                    pred_array[i] = pixel_var[0][0]
                    var_array[i] = np.log10(pixel_var[0][1])
                pred = np.mean(pred_array)
                var = np.var(pred_array) + np.mean(var_array) # heteroscedatic noise
            else:
                pred_array = np.zeros(mc_iterations)
                for i in range(mc_iterations):
                    pred_array[i] = self.regressor_model.predict(img_array.reshape(1, 128, 128, 1))
                pred = np.mean(pred_array)
                var = np.var(pred_array)
                var += 1.33 # homoscedatic noise, assuming a prior length scale of l=5

            numax = conversion_a * (np.exp((pred) * conversion_b))
            numax_sigma = conversion_b * numax * np.sqrt(var)
            file_id = int(re.search(r'\d+', target_path).group())

        elif batch == 'folder':
            if not os.path.isdir(os.path.join(self.data_dir, target_path)):
                raise ValueError('Target path does not exist!')
            nb_files = 0
            for cdirpath, cdirnames, pfilenames in os.walk(os.path.join(self.data_dir, target_path)):
                for p in range(len(pfilenames)):
                    nb_files += 1
            im_gen = ImageDataGenerator(rescale=1. / 255)
            im_gen_flow = im_gen.flow_from_directory(os.path.join(self.data_dir, target_path), target_size=(128, 128)
                                                     , color_mode='grayscale', class_mode=None,
                                                     batch_size=32, shuffle=False)
            file_id = []
            for star_id in im_gen_flow.filenames:
                file_id.append(int(re.search(r'\d+', star_id[2:]).group()))

            if self.aleatoric:
                pred_array = np.zeros(shape=(nb_files, mc_iterations))
                var_array = np.zeros(shape=(nb_files, mc_iterations))
                for i in range(mc_iterations):
                    _, pixel_var = self.regressor_model.predict_generator(im_gen_flow, steps=nb_files / 32,
                                                                                        pickle_safe=True, verbose=1)
                    pred_array[:,i] = pixel_var[:,0]
                    var_array[:,i] = np.log10(pixel_var[:,1])
                pred = np.mean(pred_array)
                var = np.var(pred_array) + np.mean(var_array) # heteroscedatic noise
            else:
                pred_array = np.zeros(shape=(nb_files, mc_iterations))
                for i in range(mc_iterations):
                    pred_array[:, i] = self.regressor_model.predict_generator(im_gen_flow, steps=nb_files / 32,
                                                                                        pickle_safe=True, verbose=1)
                pred = np.mean(pred_array, axis=1)
                var = np.var(pred_array, axis=1)
                var += 1.33 # homoscedatic noise, assuming a prior length scale of l=5
            numax = conversion_a * (np.exp((pred) * conversion_b))
            numax_sigma = conversion_b * numax * np.sqrt(var)

            file_id = np.array(file_id)
        else:
            raise ValueError("batch parameter should be either 'single' or 'folder'")


        return file_id, numax, numax_sigma

    def save(self, outfile):
        '''
        Saves the loaded regressor model.
        :param outfile: Base output file name
        :return: None
        '''
        if not self.predictable:
            raise ValueError('No saved models in memory.')
        else:
            self.regressor_model.save(outfile + '-%s.h5')

    def load(self, infile):
        '''
        Loads a regressor model.
        :param infile: Path to trained model
        :return: None
        '''
        if self.aleatoric:
            self.regressor_model = load_model(infile, custom_objects={'weighted_mean_squared_error'
                                                                      :preprocessing.weighted_mean_squared_error,
                                                       'aleatoric_loss': preprocessing.aleatoric_loss})
        else:
            self.regressor_model = load_model(infile, custom_objects=
                                                      {'weighted_mean_squared_error':
                                                           preprocessing.weighted_mean_squared_error})
        self.predictable = True


    def create_single_image(self, freq, power, star_id, out_path, label=None, numax=None):
        '''
         Creates a 2D grayscale image from the PSD of a single star and saves it to a folder.
         :param freq: Array of frequency values for the PSD
         :param power: Array of power values for the PSD
         :param star_id: The identifier of the star for file naming purposes
         :param output_path: Image output path
         :param label: Classification label for star for classifier training,
          typically 0 for nondet and 1 for positive detection
         :param numax: Numax value for star for regressor training
         :return: None
         '''
        preprocessing.generate_single_image(freq, power, star_id, out_path, label, numax)

    def create_batch_images(self, input_folder_path, output_folder_path, star_list=None, label_list=None,
                            numax_list=None):
        '''
        The batch version of create_single_image. Instead of passing single freq and array arrays, a folder with psd
        files is given. This saves all images to a folder, instead of returning an image array.
        :param input_folder_path: The folder containing all the PSD
        :param output_folder_path: The folder to contain images of the PSD
        :param star_list: For generating images for a training set, a list to cross-match with known parameters
        :param label_list: Ground truth detection values for classifier training set creation
        :param numax_list: Ground truth numax values for regressor training set creation
        :return: None
        '''
        preprocessing.generate_images(input_folder_path, output_folder_path, star_list, label_list, numax_list)

    def train_regressor(self, train_folder, validation_split=None):
        '''
        Trains a fresh regressor using a default NN architecture and parameters as of the Hon et al. (2018) paper.
        :param train_folder: The folder where training images are kept. These must be given an extra subfolder depth.
        E.g: folders should be in /TrainFolder/1/, with the specified depth up to /TrainFolder/ in the train_folder arg
        :param validation_split: Fraction of training set to use as validation, from 0 to 1 if not None
        :return: model: A trained regressor model
        '''
        from keras.callbacks import ReduceLROnPlateau
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)

        if self.aleatoric:
            model = preprocessing.default_regressor_model_aleatoric()
        else:
            model = preprocessing.default_classifier_model()

        nb_files = 0
        for dirpath, dirnames, filenames in os.walk(train_folder):
            for i in range(len(filenames)):
                nb_files += 1

        if validation_split is not None:
            datagen = ImageDataGenerator(rescale=1. / 255., height_shift_range=0.15, validation_split=validation_split)
            train_generator = datagen.flow_from_directory(train_folder, target_size=(128, 128), color_mode='grayscale',
                                                          class_mode=None, batch_size=32, subset='training')
            val_generator = datagen.flow_from_directory(train_folder, target_size=(128, 128), color_mode='grayscale',
                                                        class_mode=None, batch_size=32, subset='validation')
            if self.aleatoric:
                train_numax_generator = preprocessing.numax_generator_aleatoric(train_generator)
                validation_numax_generator = preprocessing.numax_generator_aleatoric(val_generator)
            else:
                train_numax_generator = preprocessing.numax_generator(train_generator)
                validation_numax_generator = preprocessing.numax_generator(val_generator)

            model.fit_generator(train_numax_generator, epochs=200,
                                steps_per_epoch=math.ceil((1 - validation_split) * nb_files / 32),
                                validation_data=validation_numax_generator,
                                validation_steps=math.ceil(validation_split * nb_files / 32),
                                callbacks=[reduce_lr])

        else:
            datagen = ImageDataGenerator(rescale=1. / 255., height_shift_range=0.15)
            train_generator = datagen.flow_from_directory(train_folder, target_size=(128, 128), color_mode='grayscale',
                                                          class_mode=None, batch_size=32)
            if self.aleatoric:
                train_numax_generator = preprocessing.numax_generator_aleatoric(train_generator)
            else:
                train_numax_generator = preprocessing.numax_generator(train_generator)
            model.fit_generator(train_numax_generator, epochs=200,
                                steps_per_epoch=math.ceil((1 - validation_split) * nb_files / 32),
                                callbacks=[reduce_lr])

        return model
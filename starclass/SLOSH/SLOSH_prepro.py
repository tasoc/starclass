#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for SLOSH (2D deep learning methods).

.. codeauthor::  Marc Hon <mtyh555@uowmail.edu.au>
"""

import matplotlib as mpl
import keras.backend as K
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image as pil_image
from keras.layers import Input, Dropout, MaxPooling2D, Flatten, Conv2D, LeakyReLU, concatenate
from keras.models import Model
from keras.layers.core import Dense
from keras.regularizers import l2
from keras.optimizers import Adam

mpl.rcParams['agg.path.chunksize'] = 10000

def print_images(freq, power, star_id, output_folder_path, designation=None, numax=None):
    '''
    Plots the log-log PSD and saves a 2D image into an output folder
    :param freq: Array of frequency values of the PSD
    :param power: Array of power values from the PSD
    :param designation: Integer value used to indicate classe labels if needed
    :return: None
    '''
    if designation is None and numax is None:
        output_folder = output_folder_path + '/%s.png' %star_id
    elif numax is not None and designation is None:
        output_folder = output_folder_path + '/1/%s-%.2f.png' %(star_id, numax)
    elif designation is not None and numax is None:
        output_folder = output_folder_path + '/%s/%s.png' %(designation, star_id)
    else:
        output_folder = output_folder_path + '/%s/%s-%.2f.png' % (designation, star_id, numax)

    fig = Figure(figsize=(256 / 85, 256 / 85), dpi=96)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.loglog(freq, power, c='w')
    ax.set_xlim([3., 283]) # Boundary ranges for plotting
    ax.set_ylim([3, 3e7])
    fig.tight_layout(pad=0.01)
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    canvas.draw()  # draw the canvas, cache the renderer
    canvas.print_figure(output_folder, bbox_inches='tight', pad_inches=0,facecolor='black')
    plt.close()

    return None

def generate_single_image(freq, power, star_id, output_path, label, numax):
    '''
    Generates an image from the PSD of a single star.
    :param freq: Array of frequency values for the PSD
    :param power: Array of power values for the PSD
    :param star_id: The identifier of the star for file naming purposes
    :param output_path: Image output path
    :param label: Classification label for star for classifier training,
     typically 0 for nondet and 1 for positive detection
    :param numax: Numax value for star for regressor training
    :return: None
    '''
    if label is None and numax is None:
        print_images(freq, power, star_id, output_path)
    elif label is not None and numax is None:
        if not os.path.exists(output_path+'/1/'):
            os.mkdir(output_path+'/1/')
        if not os.path.exists(output_path+'/0/'):
            os.mkdir(output_path+'/0/')
        print_images(freq, power, star_id, output_path, designation=label)
    elif numax is not None and label is None:
        if not os.path.exists(output_path+'/1/'):
            os.mkdir(output_path+'/1/')
        print_images(freq, power, star_id, output_path, numax=numax)
    else:
        if not os.path.exists(output_path+'/1/'):
            os.mkdir(output_path+'/1/')
        if not os.path.exists(output_path+'/0/'):
            os.mkdir(output_path+'/0/')
        print_images(freq, power, star_id, output_path, numax=numax, designation=label)


def generate_images(input_folder_path, output_folder_path, star_list=None, label_list=None, numax_list=None):
    '''
    Generates images from PSD in an input folder. Handles two column files with frequency as one column and power as the other.
    For ease of naming files, source files should be named with the Star ID.
    :param input_folder_path: The folder containing all the PSD
    :param output_folder_path: The folder containing all the PSD
    :param star_list: For generating images for a training set, a list to cross-match with known parameters
    :param label_list: List of ground truth detection values for classifier training set creation
    :param numax_list: List of ground truth numax values for regressor training set creation
    :return: None
    '''

    for filename in os.listdir(input_folder_path):
        merge = os.path.join(input_folder_path, filename)
        if merge.endswith('.csv'):  # Read in file, change format and IO if required
            df = pd.read_csv(merge)
            freq = df.iloc[:,0].values
            power = df.iloc[:,1].values
        elif merge.endswith('.fits'):
            with fits.open(merge) as data: # Rafa fits files
                df = pd.DataFrame(data[0].data)
                freq = df.iloc[:, 0].values * 1E6
                power = df.iloc[:, 1].values
        else:
            df = pd.read_table(merge, header=None, delim_whitespace=True)
            freq = df.iloc[:,0].values
            power = df.iloc[:,1].values
        star_id = int(re.search(r'\d+', filename).group()) # get ID from filename


        if star_list is not None: # so we have a training set
            if label_list is not None and numax_list is None:
                if not os.path.exists(output_folder_path + '/1/'):
                    os.mkdir(output_folder_path + '/1/')
                if not os.path.exists(output_folder_path + '/0/'):
                    os.mkdir(output_folder_path + '/0/')

                training_label = np.array(label_list)[np.where(np.array(star_list) == star_id)][0]
                print_images(freq, power, star_id, output_folder_path, designation=training_label)
            elif numax_list is not None and label_list is None:
                if not os.path.exists(output_folder_path + '/1/'):
                    os.mkdir(output_folder_path + '/1/')

                train_numax = np.array(numax_list)[np.where(np.array(star_list) == star_id)][0]
                print_images(freq, power, star_id, output_folder_path, numax=train_numax)
            elif numax_list is not None and label_list is not None:
                if not os.path.exists(output_folder_path + '/1/'):
                    os.mkdir(output_folder_path + '/1/')
                if not os.path.exists(output_folder_path + '/0/'):
                    os.mkdir(output_folder_path + '/0/')

                training_label = np.array(label_list)[np.where(np.array(star_list) == star_id)][0]
                train_numax = np.array(numax_list)[np.where(np.array(star_list) == star_id)][0]
                print_images(freq, power, star_id, output_folder_path, numax=train_numax, designation=training_label)
            else:
                raise FileNotFoundError('Please include training labels or numax!')
        else:
            print_images(freq, power, star_id, output_folder_path)


def img_to_array(im_path, normalize=True):
    '''
    Converts an image to a 128x128 2D grayscale pixel array
    :param im_path: Path to image
    :param normalize: If True, normalize values to lie between 0 and 1
    :return: img_array: 2D grayscale pixel array
    '''
    if not os.path.exists(im_path):
        raise ValueError('Target path does not exist!')
    img = pil_image.open(im_path).convert('L')
    img = img.resize((128, 128), pil_image.NEAREST)
    img_array = np.array(img, dtype=K.floatx())
    if normalize:
        img_array *= 1 / 255.0

    return img_array

def default_classifier_model():
    '''
    Default classifier model architecture
    :return: model: untrained classifier model
    '''
    reg = l2(7.5E-4)
    adam = Adam(clipnorm=1.)
    input1 = Input(shape=(128, 128, 1))
    drop0 = Dropout(0.25)(input1)
    conv1 = Conv2D(4, kernel_size=(7, 7), padding='same', kernel_initializer='glorot_uniform',
                   kernel_regularizer=reg)(drop0)
    lrelu1 = LeakyReLU(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu1)
    conv2 = Conv2D(8, kernel_size=(5, 5), padding='same', kernel_initializer='glorot_uniform',
                   kernel_regularizer=reg)(pool1)
    lrelu2 = LeakyReLU(0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu2)
    conv3 = Conv2D(16, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform',
                   kernel_regularizer=reg)(pool2)
    lrelu3 = LeakyReLU(0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu3)

    flat = Flatten()(pool3)
    drop1 = Dropout(0.5)(flat)
    dense1 = Dense(128, kernel_initializer='glorot_uniform', activation='relu', kernel_regularizer=reg)(drop1)
    output = Dense(2, kernel_initializer='glorot_uniform', activation='softmax')(dense1)
    model = Model(input1, output)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def default_regressor_model():
    '''
    Default regressor model architecture.
    :return: model: untrained regressor model
    '''
    reg = l2(7.5E-4)
    input1 = Input(shape=(128, 128, 1))
    drop0 = Dropout(0.25)(input1)
    conv1 = Conv2D(4, kernel_size=(5, 5), padding='same', kernel_initializer='glorot_uniform',
                   kernel_regularizer=reg)(drop0)
    lrelu1 = LeakyReLU(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu1)
    conv2 = Conv2D(8, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform',
                   kernel_regularizer=reg)(pool1)
    lrelu2 = LeakyReLU(0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu2)
    conv3 = Conv2D(16, kernel_size=(2, 2), padding='same', kernel_initializer='glorot_uniform',
                   kernel_regularizer=reg)(pool2)
    lrelu3 = LeakyReLU(0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu3)
    flat = Flatten()(pool3)
    drop1 = Dropout(0.5)(flat)

    dense1 = Dense(1024, kernel_initializer='glorot_uniform', activation='relu', kernel_regularizer=reg)(drop1)
    dense2 = Dense(128, kernel_regularizer=reg, kernel_initializer='glorot_uniform', activation='relu')(dense1)
    output = Dense(1, kernel_initializer='glorot_uniform')(dense2)
    model = Model(input1, output)
    # 1024-128-1 has 5 mse
    # 1024-1 had 7 mse
    model.compile(optimizer='Nadam', loss=weighted_mean_squared_error, metrics=['mae'])
    return model

def default_regressor_model_aleatoric():
    '''
    Default prototype regressor model architecture that uses aleatoric loss.
    :return: model: untrained regressor model
    '''
    reg = l2(7.5E-4)
    input1 = Input(shape=(128, 128, 1))
    # pad = ZeroPadding2D(8)(input1)
    drop0 = Dropout(0.25)(input1)
    conv1 = Conv2D(4, kernel_size=(5, 5), padding='same', kernel_initializer='glorot_uniform',
                   kernel_regularizer=reg)(drop0)
    lrelu1 = LeakyReLU(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu1)
    conv2 = Conv2D(8, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform',
                   kernel_regularizer=reg)(pool1)
    lrelu2 = LeakyReLU(0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu2)
    conv3 = Conv2D(16, kernel_size=(2, 2), padding='same', kernel_initializer='glorot_uniform',
                   kernel_regularizer=reg)(pool2)
    lrelu3 = LeakyReLU(0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='valid')(lrelu3)
    flat = Flatten()(pool3)
    drop1 = Dropout(0.5)(flat)

    dense1 = Dense(1024, kernel_initializer='glorot_uniform', activation='relu', kernel_regularizer=reg)(drop1)
    dense2 = Dense(128, kernel_regularizer=reg, kernel_initializer='glorot_uniform', activation='relu')(dense1)
    output = Dense(1, kernel_initializer='glorot_uniform', name='prediction')(dense2)
    output_var = Dense(1, kernel_initializer='glorot_uniform', name='variance')(dense2)
    pred_var = concatenate([output, output_var], name='pred_var')

    model = Model(input1, [output, pred_var])
    # 1024-128-1 has 5 mse
    # 1024-1 had 7 mse
    model.compile(optimizer='Nadam', loss={'prediction': weighted_mean_squared_error,
                                           'pred_var': aleatoric_loss}, metrics={'prediction': 'mae'},
                  loss_weights={'prediction': 1., 'pred_var': .2})
    return model

def numax_generator(generator):
    '''
    Converts a flow_from_directory generator to a generator that takes values from the filenames and outputs numax
    :param generator:
    :return:
    '''
    conversion_a = 3  # constants for conversion from pixel coordinate to frequency in uHz
    conversion_b = (1. / 128.) * np.log(283. / 3.)
    for x in generator:
        idx = (generator.batch_index - 1) * generator.batch_size
        names = generator.filenames[idx:idx + generator.batch_size]
        if len(names) == 0:
            continue
        numaxes = np.zeros(len(names))
        print(numaxes)
        for i in range(len(names)):
            numaxes[i] = (float(names[i].split("-", 1)[1][:-4]))
        numaxes = (1 / conversion_b) * np.log(numaxes / conversion_a)  # convert to pixel coordinates
        yield x, np.array(numaxes).reshape((np.array(numaxes).shape[0], 1))

def numax_generator_aleatoric(generator):
    '''
    Converts a flow_from_directory generator to a generator that takes values from the filenames and outputs numax for
    aleatoric models
    :param generator:
    :return:
    '''
    conversion_a = 3  # constants for conversion from pixel coordinate to frequency in uHz
    conversion_b = (1. / 128.) * np.log(283. / 3.)
    for x in generator:
        idx = (generator.batch_index - 1) * generator.batch_size
        names = generator.filenames[idx:idx + generator.batch_size]
        if len(names) == 0:
            continue
        numaxes = np.zeros(len(names))
        for i in range(len(names)):
            numaxes[i] = (float(names[i].split("-", 1)[1][:-4]))  # get numax from filename
        numaxes = (1 / conversion_b) * np.log(numaxes / conversion_a)  # convert to pixel coordinates
        numaxes = np.array(numaxes).reshape((np.array(numaxes).shape[0], 1))
        yield x, [numaxes, numaxes]


def aleatoric_loss(y_true, pred_var):
    '''
    Aleatoric loss function for heteroscedatic noise estimation in deep learning models. Needed for model imports.
    :param y_true: Ground truth
    :param pred_var: Prediction appended with variance
    :return: Aleatoric Loss
    '''
    y_pred = pred_var[:, 0] # here pred_var should be [prediction, variance], y_true is true numax
    log_var = pred_var[:, 1]
    loss = (K.abs(y_true - y_pred)) * (K.exp(-log_var)) + log_var
    return K.mean(loss)

def weighted_mean_squared_error(y_true, y_pred):
    '''
    Custom loss function for training the regressor. Prioritizes getting low/high numax predictions correct.
    :param y_true: Ground truth
    :param y_pred: Model predicted value
    :return: Weighted MSE loss
    '''
    return K.mean((K.square(y_pred - y_true))*K.square(y_true-64), axis=-1)
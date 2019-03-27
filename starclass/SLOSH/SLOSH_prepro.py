#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for SLOSH (2D deep learning methods).

.. codeauthor::  Marc Hon <mtyh555@uowmail.edu.au>
"""

import matplotlib as mpl
import keras
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


class npy_generator(keras.utils.Sequence):
    """
    Generator that loads numpy arrays from a folder for training a deep learning model. This version has been tailored
    for a classifier, with the training labels taken from the subfolder. Indices of training/validation can be passed
    to indicate which files to partition for each set.
    Written by  Marc Hon (mtyh555@uowmail.edu.au)
    """
    def __init__(self, root, batch_size, dim, extension = '.npz', shuffle = True, indices=[]):
        self.root = root # root folder containing subfolders
        self.batch_size = batch_size
        self.extension = extension # file extension
        self.filenames = []
        self.subfolder_labels = [] # for binary classification
        self.shuffle = shuffle # shuffles data after every epoch
        self.dim=dim # image/2D array dimensions

        for dirpath, dirnames, filenames in os.walk(root):
            for file in filenames:
                if file.endswith(extension) & dirpath[-1].isdigit(): # I infer the class label '0' or '1' according to subfolder names
                    self.filenames.append(os.path.join(dirpath, file))
                    self.subfolder_labels.append(int(dirpath[-1]))
        if len(indices) == 0: # otherwise pass a list of training/validation indices
            self.indexes = np.arange(len(self.filenames))
        else:
            self.indexes = np.array(indices)

    def __len__(self):
        return int(np.ceil(len(self.indexes) / float(self.batch_size)))
    def __getitem__(self, index):
        # Generate indexes of the batch
        batch_indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Get a list of filenames of the batch
        batch_filenames = [self.filenames[k] for k in batch_indices]
        batch_labels = [self.subfolder_labels[k] for k in batch_indices]
        # Generate data
        X, y = self.__data_generation(batch_filenames, batch_labels)
        return X, keras.utils.to_categorical(y, num_classes=2)

    def on_epoch_end(self):
        # Shuffles indices after every epoch
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_filenames, batch_labels):
        # Generates data - this example is repurposed for .npy files
        X = np.empty((len(batch_filenames), self.dim[0], self.dim[1]))
        y = np.empty((len(batch_filenames)), dtype=int)

        for i, ID in enumerate(batch_filenames):
            X[i, :] = np.load(ID)['im']
            y[i] = batch_labels[i]
        return np.expand_dims(X,-1), y

def local_maxima(grid, search_radius):
    moving_max_vec = np.zeros(len(grid))

    for i in range(len(grid)):
        if i < 0.8*len(grid):
            radius = search_radius
            q = 95
        else:
            radius = int(search_radius/2)
            q = 98

        if (i + radius) > len(grid) - 1:
            upper_bound = len(grid) - 1
            lower_bound = (i - radius) - ((i + radius) - (len(grid) - 1))
        elif (i - radius) < 0:
            lower_bound = 0
            upper_bound = (i + radius) + i
        else:
            upper_bound = i + radius
            lower_bound = i - radius

        moving_max_vec[i] = np.percentile(grid[lower_bound: upper_bound], q=q)

    return moving_max_vec


def squeeze(arr, minval, maxval, axis=0):
    """
    Returns version of 1D arr with values squeezed to range [minval,maxval]
    """
    #array is 1D
    minvals = np.ones(arr.shape)*minval
    maxvals = np.ones(arr.shape)*maxval

    #assure above minval first
    squeezed = np.max(np.vstack((arr,minvals)),axis=0)
    squeezed = np.min(np.vstack((squeezed,maxvals)),axis=0)

    return squeezed


def ps_to_array(freq, power, nbins=128, supersample=1,
                minfreq=3., maxfreq=283., minpow=3., maxpow=3e7):
    """
    Produce 2D array representation of power spectrum that is similar to Marc Hon's 2D images
    Written by Keaton Bell (bell@mps.mpg.de)
    This should be faster and more precise than writing plots to images
    Returns nbin x nbins image-like representation of the data
    freq and power are from power spectrum
    min/max freqs/powers define the array edges in same units as input spectrum
    if supersample == 1, result is strictly black and white (1s and 0s)
    if supersample > 1, returns grayscale image represented spectrum "image" density
    """
    # make sure integer inputs are integers
    nbins = int(nbins)
    supersample = int(supersample)
    # Set up array for output
    output = np.zeros((nbins, nbins))
    if supersample > 1:  # SUPERSAMPLE
        # Call yourself and flip orientation again
        supersampled =  ps_to_array(freq, power, nbins=nbins * supersample, supersample=1,
                                        minfreq=minfreq, maxfreq=maxfreq, minpow=minpow, maxpow=maxpow)[::-1]
        for i in range(supersample):
            for j in range(supersample):
                output += supersampled[i::supersample, j::supersample]
        output = output / (supersample ** 2.)
    else:  # don't supersample
        # Do everything in log space
        logfreq = np.log10(freq)
        logpower = np.log10(power)
        minlogfreq = np.log10(minfreq)
        maxlogfreq = np.log10(maxfreq)
        minlogpow = np.log10(minpow)
        maxlogpow = np.log10(maxpow)

        # Define bins

        xbinedges = np.linspace(np.log10(minfreq), np.log10(maxfreq), nbins + 1)
        xbinwidth = xbinedges[1] - xbinedges[0]
        ybinedges = np.linspace(np.log10(minpow), np.log10(maxpow), nbins + 1)
        ybinwidth = ybinedges[1] - ybinedges[0]

        # resample at/near edges of bins and at original frequencies

        smalloffset = xbinwidth / (10. * supersample)  # to get included in lower-freq bin
        interpps = interp1d(logfreq, logpower, fill_value=(0,0), bounds_error=False)
        poweratedges = interpps(xbinedges)
        logfreqsamples = np.concatenate((logfreq, xbinedges, xbinedges - smalloffset))
        powersamples = np.concatenate((logpower, poweratedges, poweratedges))

        sort = np.argsort(logfreqsamples)
        logfreqsamples = logfreqsamples[sort]
        powersamples = powersamples[sort]

        # Get maximum and minimum of power in each frequency bin
        maxpow = binned_statistic(logfreqsamples, powersamples, statistic='max', bins=xbinedges)[0]
        minpow = binned_statistic(logfreqsamples, powersamples, statistic='min', bins=xbinedges)[0]
        # Convert to indices of binned power

        # Fix to fall within power range
        minpowinds = np.floor((minpow - minlogpow) / ybinwidth)
        minpowinds = squeeze(minpowinds, 0, nbins).astype('int')
        maxpowinds = np.ceil((maxpow - minlogpow) / ybinwidth)
        maxpowinds = squeeze(maxpowinds, 0, nbins).astype('int')

        # populate output array
        for i in range(nbins):
            output[minpowinds[i]:maxpowinds[i], i] = 1.
            if maxpowinds[i] - minpowinds[i] != np.sum(output[minpowinds[i]:maxpowinds[i], i]):
                print(i, "!!!!!!")
                print(minpowinds[i])
                print(maxpowinds[i])
    # return result, flipped to match orientation of Marc's images
    return output[::-1]


def generate_single_image(freq, power):
    '''
    Generates an image from the PSD of a single star.
    :param freq: Array of frequency values for the PSD
    :param power: Array of power values for the PSD
    :return: image: 2D binary array containing the PSD 'image'
    '''
    image = ps_to_array(freq, power)
    return image


def generate_train_images(freq, power, star_id, output_path, label):
    '''
    Generates images from PSD in an input folder. Handles two column files with frequency as one column and power as the other.
    For ease of naming files, source files should be named with the Star ID.
    :param freq: Frequency values for the PSD
    :param power: Power values for the PSD
    :param star_list: For generating images for a training set, a list to cross-match with known parameters
    :param output_path: Image output path
    :param label: Training label
    :param numax: Numax value for star for regressor training (for later implementation)
    :return: None
    '''

    if label is None:
        image = generate_single_image(freq, power)
        np.savez_compressed(file = output_path+'/%s' %star_id, im=image)
    else:
        if not os.path.exists(output_path + '/1/'):
            os.mkdir(output_path + '/1/')
        if not os.path.exists(output_path + '/0/'):
            os.mkdir(output_path + '/0/')
        np.savez_compressed(file=output_path+'/%s/%s'%(label, star_id))


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

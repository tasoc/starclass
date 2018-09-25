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
        output_folder = output_folder_path + '/%s' %star_id
    elif numax is not None:
        output_folder = output_folder_path + '/1/%s-%.2f' %(star_id, numax)
    elif designation is not None:
        output_folder = output_folder_path + '/%s/%s' %(designation, star_id)

    fig = Figure(figsize=(256 / 85, 256 / 85), dpi=96)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.loglog(freq, power, c='w')
    ax.set_xlim([3., 283])
    ax.set_ylim([3, 3e7])
    fig.tight_layout(pad=0.01)
    ax.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    canvas.draw()  # draw the canvas, cache the renderer
    canvas.print_figure(output_folder, bbox_inches='tight', pad_inches=0,facecolor='black')
    plt.close()

    return None

def generate_images(input_folder_path, output_folder_path, star_list=None, label_list=None, numax_list=None):
    '''
    Generates images from PSD in an input folder. Handles two column files with frequency as one column and power as the other.
    For ease of naming files, source files should be named with the Star ID.
    :param input_folder_path: The folder containing all the PSD
    :param output_folder_path: The folder containing all the PSD
    :param star_list: For generating images for a training set, a list to cross-match with known parameters
    :param label_list: Ground truth detection values for classifier training set creation
    :param numax_list: Ground truth numax values for regressor training set creation
    :return: None
    '''

    for filename in os.listdir(input_folder_path):
        merge = os.path.join(input_folder_path, filename)
        if merge.endswith('.csv'):  # Read in file, change format and IO if required
            df = pd.read_csv(merge)
        elif merge.endswith('.fits'):
            with fits.open(merge) as data:
                df = pd.DataFrame(data[0].data)
        else:
            df = pd.read_table(merge, header=None, delim_whitespace=True)
        star_id = int(re.search(r'\d+', filename).group()) # get ID from filename

        df.columns = ['Frequency', 'Power']

        freq = df['Frequency'].values
        power = df['Power'].values

        if star_list is not None: # so we have a training set
            if label_list is not None:
                training_label = label_list[np.where(star_list == star_id)]
                print_images(freq, power, star_id, output_folder_path, designation=training_label)
            elif numax_list is not None:
                train_numax = numax_list[np.where(star_list == star_id)]
                print_images(freq, power, star_id, output_folder_path, numax=train_numax)
            else:
                raise FileNotFoundError('Please include training labels or numax!')
        else:
            print_images(freq, power, star_id, output_folder_path)


def aleatoric_loss(y_true, pred_var):
    '''
    Aleatoric loss function for heteroscedatic noise estimation in deep learning models. Needed for model imports.
    :param y_true: Ground truth
    :param pred_var: Model predicted value
    :return: Aleatoric Loss
    '''
    y_pred = pred_var[:, 0] # here pred_var should be [prediction, variance], y_true is true numax
    log_var = pred_var[:, 1]
    loss = (K.abs(y_true - y_pred)) * (K.exp(-log_var)) + log_var
    return K.mean(loss)
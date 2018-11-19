#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 09:05:24 2018

@author: fingeza
"""
import pandas as pd
import numpy as np 
import feets
import upsilon.utils as utils
#import glob
import warnings
import os
warnings.simplefilter("ignore", RuntimeWarning)
import logging
logger = logging.getLogger(__name__)


Features_file_path = '~/Documents/starclass/starclass/data/L1'

def lc_norm(lc, linflatten = False):
    """
    Preprocess light curves using sigma clipping and normalise them with the median
    #This is is done under the assumption that missing (nan) have been removed
    """
    #nan_values = (lc == 0) | np.isnan(lc)
    nancut = (lc.flux==0) | np.isnan(lc.flux)
    t = lc.time[~nancut]
    mag = lc.flux[~nancut]
    dmag = lc.flux_err[~nancut]
    lc_clipped = np.array(utils.sigma_clipping(t,mag,dmag,threshold=2.5,iteration=1))
    mag_median = np.median(lc_clipped[1])
    mag_norm = lc_clipped[1]/mag_median 
    error_norm = lc_clipped[2]/ mag_median
    t = lc_clipped[0]
    lc_f = np.array((t,mag_norm, error_norm))
    if linflatten:
        mag_norm = mag_norm - np.polyval(np.polyfit(t,mag_norm,1),t) + 1
    return lc_f


def feature_extract_single(features,linflatten = True):
    """
    Calculate features for a single light curve
    """
    light_curve = features('lightcurve')
    #print(light_curve)
    #feature_list = np.zeros(22)
    lc = lc_norm(light_curve, linflatten)
    fs =feets.FeatureSpace(exclude=['AndersonDarling','Color','Eta_color','Q31_color',
                                    'StetsonK','StetsonJ','PercentDifferenceFluxPercentile',
                                    'SlottedA_length','Autocor_length','Con','Mean','Eta_e',
                                    'StructureFunction_index_21','StructureFunction_index_31',
                                    'StructureFunction_index_32','Freq1_harmonics_rel_phase_0',
                                    'FluxPercentileRatioMid20','FluxPercentileRatioMid35',
                                    'FluxPercentileRatioMid50','FluxPercentileRatioMid65',
                                    'FluxPercentileRatioMid80','Freq2_harmonics_rel_phase_0',
                                    'Freq3_harmonics_rel_phase_0','Period_fit','StetsonK_AC','StetsonL',
                                    'PercentAmplitude','Freq1_harmonics_amplitude_1','Freq1_harmonics_amplitude_2', 
                                    'Freq1_harmonics_amplitude_3','Freq2_harmonics_amplitude_0',
                                    'Freq2_harmonics_amplitude_1','Freq2_harmonics_amplitude_2',
                                    'Freq2_harmonics_amplitude_3','Freq3_harmonics_amplitude_0',
                                    'Freq3_harmonics_amplitude_1','Freq3_harmonics_amplitude_2',
                                    'Freq3_harmonics_amplitude_3','Gskew','MaxSlope',
                                    'MedianAbsDev','MedianBRP','SmallKurtosis','Q31','Std','Psi_eta'])
    Feature_ID, values = fs.extract(*lc)
    features_dict = dict(zip(Feature_ID,values))
    Features_final = pd.DataFrame(features_dict, index=[0])
    return Features_final


def feature_extract(features,linflatten = True):
    Features_all = pd.DataFrame()
    for obj in features:
        lc = lc_norm(obj['lightcurve'], linflatten)
        #print(lc)
        #Features_final = feature_extract_single(features,linflatten = True)
        fs =feets.FeatureSpace(exclude=['AndersonDarling','Color','Eta_color','Q31_color',
                                    'StetsonK','StetsonJ','PercentDifferenceFluxPercentile',
                                    'SlottedA_length','Autocor_length','Con','Mean','Eta_e',
                                    'StructureFunction_index_21','StructureFunction_index_31',
                                    'StructureFunction_index_32','Freq1_harmonics_rel_phase_0',
                                    'FluxPercentileRatioMid20','FluxPercentileRatioMid35',
                                    'FluxPercentileRatioMid50','FluxPercentileRatioMid65',
                                    'FluxPercentileRatioMid80','Freq2_harmonics_rel_phase_0',
                                    'Freq3_harmonics_rel_phase_0','Period_fit','StetsonK_AC','StetsonL',
                                    'PercentAmplitude','Freq1_harmonics_amplitude_1','Freq1_harmonics_amplitude_2', 
                                    'Freq1_harmonics_amplitude_3','Freq2_harmonics_amplitude_0',
                                    'Freq2_harmonics_amplitude_1','Freq2_harmonics_amplitude_2',
                                    'Freq2_harmonics_amplitude_3','Freq3_harmonics_amplitude_0',
                                    'Freq3_harmonics_amplitude_1','Freq3_harmonics_amplitude_2',
                                    'Freq3_harmonics_amplitude_3','Gskew','MaxSlope',
                                    'MedianAbsDev','MedianBRP','SmallKurtosis','Q31','Std','Psi_eta'])
        Feature_ID, values = fs.extract(*lc)
        features_dict = dict(zip(Feature_ID,values))
        Features_final = pd.DataFrame(features_dict, index=[0])
        Features_all = Features_all.append(Features_final)
        Features_all.to_csv(os.path.join(Features_file_path, 'feets_features.csv'), index=False)
        #Features_all['ID'] = ID
        #Features_all.set_index('ID', inplace=True)
    return Features_all
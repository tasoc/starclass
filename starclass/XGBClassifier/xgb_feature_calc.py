#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 09:05:24 2018

@author: fingeza
"""
import pandas as pd
import numpy as np
import feets
#import glob
import warnings
import os
import types

from tqdm import tqdm
from ..RFGCClassifier import RF_GC_featcalc

warnings.simplefilter("ignore", RuntimeWarning)
import logging
logger = logging.getLogger(__name__)


#Features_file_path = '~/Documents/starclass/starclass/data/L1'

def lc_norm(lc, linflatten = False):
    """
    Preprocess light curves using sigma clipping and normalise them with the median
    #This is is done under the assumption that missing (nan) have been removed
    """
    #nan_values = (lc == 0) | np.isnan(lc)
    lc = lc.remove_nans().remove_outliers(sigma=2.5)
    #nancut = (lc.flux==0) | np.isnan(lc.flux)
    #t = lc.time[~nancut]
    lc = lc[lc.flux != 0]
    lc.flux = lc.flux*1e-6 + 1
    lc.flux_err = lc.flux_err * 1e-6
    #mag = lc.flux[~nancut] * 1e-6 + 1
    #dmag = lc.flux_err[~nancut] * 1e-6
    #lc_clipped = np.array(utils.sigma_clipping(t,mag,dmag,threshold=2.5,iteration=1))

    #mag_median = np.median(lc_clipped[1])
    #mag_norm = lc_clipped[1]/mag_median
    #error_norm = lc_clipped[2]/ mag_median
    #t = lc_clipped[0]
    #lc_f = np.array((t,mag, dmag))

    if linflatten:
        #lc_f[:,1] = lc_f[:,1] - np.polyval(np.polyfit(lc_f[:,0],lc_f[:,1],1),lc_f[:,0]) + 1
        lc.flux = lc.flux - np.polyval(np.polyfit(lc.time,lc.flux ,1),lc.time) + 1
    return lc

def feature_extract(features, savefeat=None, linflatten=False, recalc=False):
    featout = pd.DataFrame()
    if not isinstance(features, types.GeneratorType):
        features = [features]
    for idx, obj in tqdm(enumerate(features)):

        precalc = False
        if savefeat is not None:
            featfile = os.path.join(savefeat, str(obj['priority'])+'.txt')
            if os.path.exists(featfile) and not recalc:
                objfeatures = pd.read_csv(featfile)
                #print(np.shape(objfeatures))
                #print(objfeatures)
                precalc = True
                featout = featout.append(objfeatures)

        if not precalc:
            lc = lc_norm(obj['lightcurve'], linflatten)



            #Features_final = feature_extract_single(features,linflatten = True)
            fs =feets.FeatureSpace(exclude=[#'Amplitude', 'Beyond1Std',
                                            'Freq1_harmonics_amplitude_0',
                                            'Freq1_harmonics_rel_phase_1', 'Freq1_harmonics_rel_phase_2',
                                            'Freq1_harmonics_rel_phase_3', 'Freq2_harmonics_rel_phase_1',
                                            'Freq2_harmonics_rel_phase_2', 'Freq2_harmonics_rel_phase_3',
                                            'Freq3_harmonics_rel_phase_1', 'Freq3_harmonics_rel_phase_2',
                                            'Freq3_harmonics_rel_phase_3',
                                            #'LinearTrend', 'Meanvariance',
                                            #'PairSlopeTrend',
                                            'PeriodLS',
                                            'Psi_CS', 'Rcs',
                                            #'Skew',
                                            'CAR_mean', 'CAR_sigma', 'CAR_tau',
                                        'AndersonDarling','Color','Eta_color','Q31_color',
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



            Feature_ID, values = fs.extract(*np.array([lc.time, lc.flux, lc.flux_err]))

            features_dict = dict(zip(Feature_ID,values))


            forbiddenfreqs=[13.49/4.]
            periods, usedfreqs = checkfrequencies(obj, 6, 6,
            									 forbiddenfreqs, lc.time)
            amp21,amp31 = freq_ampratios(obj,usedfreqs)
            pd21,pd31 = freq_phasediffs(obj,usedfreqs)

            features_dict['PeriodLS'] = periods[0]
            if len(usedfreqs)>0:
                features_dict['Freq_amp_0'] = obj['amp'+str(usedfreqs[0]+1)]
            else:
                features_dict['Freq_amp_0'] = 0.

            features_dict['Freq_ampratio_21'] = amp21
            features_dict['Freq_ampratio_31'] = amp31
            features_dict['Freq_phasediff_21'] = pd21
            features_dict['Freq_phasediff_31'] = pd31

            # phase-fold lightcurve on dominant period
            folded_lc = phase_fold_lc(lc, periods[0])
            # Compute phi_rcs and rcs features
            rcs = Rcs(lc.flux)
            psi_rcs = Rcs(folded_lc)

            features_dict['Rcs'] = rcs
            features_dict['psi_Rcs'] = psi_rcs

            objfeatures = pd.DataFrame(features_dict, index=[0])
            if savefeat is not None:
                objfeatures.to_csv(featfile, index=False)
            featout = featout.append(objfeatures)

            #Features_all.to_csv(os.path.join(Features_file_path, 'feets_features.csv'), index=False)
            #Features_all['ID'] = ID
            #Features_all.set_index('ID', inplace=True)
        #featout = np.vstack((featout, objfeatures.values))
    return featout

def Rcs(flux):
    """
    Rcs from feet but without overhead
    """
    sigma = np.std(flux)
    N = len(flux)
    m = np.mean(flux)
    s = np.cumsum(flux - m) * 1.0 / (N * sigma)
    R = np.max(s) - np.min(s)
    return R

def phase_fold_lc(lc, per):
    """
    Uses functions from RF_GC_featcalc to compute phase folded light curve
    """
    # Compute additional features
    time, flux = lc.time.copy(), lc.flux.copy()

    EBper = RF_GC_featcalc.EBperiod(time, flux, per)
    phase = RF_GC_featcalc.phasefold(time, EBper)
    return flux[np.argsort(phase)]


def checkfrequencies(featdictrow, nfreqs, providednfreqs, forbiddenfreqs, time):
    """
    Cuts frequency data down to desired number of frequencies, and removes harmonics
    of forbidden frequencies

    Inputs
    -----------------


    Returns
    -----------------
    freqs: ndarray [self.nfreqs]
         array of frequencies
    """
    freqs = []
    usedfreqs = []
    j = 0
    while len(freqs)<nfreqs:
        freqdict = featdictrow['freq'+str(j+1)]
        if np.isfinite(freqdict):
            freq = 1./(freqdict*1e-6)/86400.  #convert to days

            #check to cut bad frequencies
            cut = False
            if (freq < 0) or (freq > np.max(time)-np.min(time)):
                cut = True
            for freqtocut in forbiddenfreqs:
                for k in range(4):  #cuts 4 harmonics of frequency, within +-3% of given frequency
                    if (1./freq > (1./((k+1)*freqtocut))*(1-0.01)) & (1./freq < (1./((k+1)*freqtocut))*(1+0.01)):
                        cut = True
            if not cut:
                freqs.append(freq)
                usedfreqs.append(j)
        j += 1
        if j >= providednfreqs:
            break
    #fill in any unfilled frequencies with negative numbers
    gap = nfreqs - len(freqs)
    if gap > 0:
        for k in range(gap):
            freqs.append(np.max(time)-np.min(time))
    return np.array(freqs), np.array(usedfreqs)

def freq_ampratios(featdictrow, usedfreqs):
    """
    Amplitude ratios of frequencies

    Inputs
    -----------------


    Returns
    -----------------
    amp21, amp31: float, float
        ratio of 2nd to 1st and 3rd to 1st frequency amplitudes

    """
    if len(usedfreqs) >= 2:
        amp21 = featdictrow['amp'+str(usedfreqs[1]+1)]/featdictrow['amp'+str(usedfreqs[0]+1)]
    else:
        amp21 = 0
    if len(usedfreqs) >= 3:
        amp31 = featdictrow['amp'+str(usedfreqs[2]+1)]/featdictrow['amp'+str(usedfreqs[0]+1)]
    else:
        amp31 = 0
    return amp21,amp31

def freq_phasediffs(featdictrow, usedfreqs):
    """
    Phase differences of frequencies

    Inputs
    -----------------

    Returns
    -----------------
    phi21, phi31: float, float
        phase difference of 2nd to 1st and 3rd to 1st frequencies

    """
    if len(usedfreqs) >= 2:
        phi21 = featdictrow['phase'+str(usedfreqs[1]+1)] - 2*featdictrow['phase'+str(usedfreqs[0]+1)]
    else:
        phi21 = 0
    if len(usedfreqs) >= 3:
        phi31 = featdictrow['phase'+str(usedfreqs[2]+1)] - 3*featdictrow['phase'+str(usedfreqs[0]+1)]
    else:
        phi31 = 0
    return phi21,phi31

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities for the RF-GC classifier (general random forest).

.. codeauthor::  David Armstrong <d.j.armstrong@warwick.ac.uk>
"""

import numpy as np
import os
import glob
import lightkurve
import logging
logger = logging.getLogger(__name__)

def featcalc(features, som, 
					providednfreqs=6, nfrequencies=6, forbiddenfreqs=[13.49/4.],
					cardinality=64, linflatten=False, savefeat=None, recalc=False):
	"""
	Calculates features for set of lightcurves
	"""
	featout = np.zeros([1,nfrequencies+16])
	if isinstance(features,dict): #trick for single features
		features = [features]
        
	for obj in features:   
		precalc = False
		if savefeat is not None:
			featfile = os.path.join(savefeat,str(obj['priority'])+'.txt')
			if os.path.exists(featfile) and not recalc:
				#logger.info(str(obj['priority'])+": Loading precalculated features...")
				objfeatures = np.loadtxt(featfile,delimiter=',')
				precalc = True
		
		if not precalc:	
			objfeatures = np.zeros(nfrequencies+16)
			lc = prepLCs(obj['lightcurve'],linflatten)
			periods, usedfreqs = checkfrequencies(obj, nfrequencies, providednfreqs,
        									 forbiddenfreqs, lc.time)
			objfeatures[:nfrequencies] = periods
			objfeatures[nfrequencies:nfrequencies+2] = freq_ampratios(obj,usedfreqs)
			objfeatures[nfrequencies+2:nfrequencies+4] = freq_phasediffs(obj,usedfreqs)
			EBper = EBperiod(lc.time, lc.flux, periods[0], linflatten=linflatten-1)
			objfeatures[0] = EBper #overwrites top period
			objfeatures[nfrequencies+4:nfrequencies+6] = SOMloc(som, lc.time, lc.flux, EBper, 
        													cardinality)
			objfeatures[nfrequencies+6:nfrequencies+8] = phase_features(lc.time, lc.flux,EBper)
			objfeatures[nfrequencies+8:nfrequencies+10] = p2p_features(lc.flux)
			psi, zc = compute_hocs(lc.time, lc.flux, 5)
			objfeatures[nfrequencies+10] = psi
			objfeatures[nfrequencies+11] = zc[0]
			objfeatures[nfrequencies+12:] = obj['Fp07'], obj['Fp7'], obj['Fp20'], obj['Fp50']
			if savefeat is not None:
				np.savetxt(featfile,objfeatures,delimiter=',')
		featout = np.vstack((featout,objfeatures))
	return featout[1:,:]

def prepLCs(lc,linflatten=False):
    """
    Nancut lightcurve, converts from ppm to relative flux and centres around 1. 
    Optionally removes linear trend.
    Assumes LCs come in in normalised ppm with median zero.
    """
    nancut = (lc.flux==0) | np.isnan(lc.flux)
    
    #norm = np.median(lc.flux[~nancut])
    #lc.flux /= norm
    #lc.flux_err /= norm
        
    lc.time = lc.time[~nancut]
    lc.flux = lc.flux[~nancut]
    lc.flux_err = lc.flux_err[~nancut]
    
    lc.flux *= 1e-6
    lc.flux_err *= 1e-6
    lc.flux += 1
    
    if linflatten:
        lc.flux = lc.flux - np.polyval(np.polyfit(lc.time,lc.flux,1),lc.time) + 1
            
    return lc


def makeSOM(features,outfile,overwrite=False,cardinality=64,dimx=1,dimy=400,
				nsteps=300,learningrate=0.1):
	"""
	Top level function for training a SOM.
	"""
	logger.info('Preparing lightcurves for SOM')
	SOMarray = SOM_alldataprep(features,cardinality=cardinality)
	logger.info(str(SOMarray.shape[0])+' lightcurves prepared. Training SOM')
	som = SOM_train(SOMarray,outfile,overwrite,cardinality,dimx,dimy,nsteps,learningrate)
	logger.info('SOM trained.')
	return som

def loadSOM(somfile):
    """
    Loads a previously trained SOM.
        
    Inputs
    -----------------
    somfile: 		str
    	Filepath to saved SOM (saved using self.kohonenSave)
        	
    Returns
    -----------------
    som:	 object
         Trained som object
    """
    with open(somfile,'r') as f:
        lines = f.readlines()
    newshape = lines[0].strip('\n').split(',')
    dimx,dimy,cardinality = int(newshape[0]),int(newshape[1]),int(newshape[2])
           
    def Init(sample):
        '''
        Initialisation function for SOM.
        '''
        return np.random.uniform(0,1,size=(dimx,dimy,cardinality))

    from . import selfsom
    som = selfsom.SimpleSOMMapper((dimx,dimy),1,initialization_func=Init,learning_rate=0.1)
    loadk = kohonenLoad(somfile)
    som.train(loadk)  #purposeless but tricks the SOM into thinking it's been trained. Don't ask.
    som._K = loadk
    return som


def kohonenLoad(infile):
    """
    Loads a 3d array saved with self.kohonenSave(). Auto-detects dimensions.
        
    Inputs
    -----------------
    infile: str
        Filepath to load
        	
    Returns
    -----------------
    out: ndarray, size [i,j,k]
         Loaded array.
    """
    with open(infile,'r') as f:
        lines = f.readlines()
    newshape = lines[0].strip('\n').split(',')
    out = np.zeros([int(newshape[0]),int(newshape[1]),int(newshape[2])])
    for i in range(int(newshape[0])):
        for j in range(int(newshape[1])):
            line = lines[1+(i*int(newshape[1]))+j].strip('\n').split(',')
            for k in range(int(newshape[2])):
                out[i,j,k] = float(line[k])
    return out

def kohonenSave(layer,outfile):  #basically a 3d >> 2d saver
    """
    Takes a 3d array and saves it to txt file in a recoverable way.
        
    Inputs
    -----------------
    layer: 		ndarray, 3 dimensional, size [i,j,k]
        Array to save.
        	
    outfile: 	str
        Filepath to save to.
    """
    with open(outfile,'w') as f:
        f.write(str(layer.shape[0])+','+str(layer.shape[1])+','+str(layer.shape[2])+'\n')
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                for k in range(layer.shape[2]):
                    f.write(str(layer[i,j,k]))
                    if k < layer.shape[2]-1:
                        f.write(',')
                f.write('\n')


def SOM_alldataprep(features, outfile=None, cardinality=64):
    ''' Function to create an array of normalised lightcurves to train a SOM
        
    Parameters
    ----------------
    lightcurves
    frequencies
	outfile:		str, optional
		Filepath to save array to. If not populated, just returns array
	cardinality
		            
    Returns
    -----------------
    SOMarray:		np array, [n_lightcurves, cardinality]
        Array of phase-folded, binned lightcurves
    '''    
    SOMarray = np.ones(cardinality)
    for obj in features: 
        lc = obj['lightcurve']
        lc = prepLCs(lc,linflatten=True)
        
        freq = obj['freq1']
        if np.isfinite(freq):

            time, flux = lc.time.copy(), lc.flux.copy()

            #check double period
            per = 1./(freq*1e-6)/86400. #convert to days    
            EBper = EBperiod(time, flux, per)  
            if EBper > 0: #ignores others
                binlc,range = prepFilePhasefold(time, flux, EBper,cardinality)
                SOMarray = np.vstack((SOMarray,binlc))
    if outfile is not None:
        np.savetxt(outfile,SOMarray[1:,:])
    return SOMarray[1:,:] #drop first line as this is just ones
    
def SOM_train(SOMarray, outfile=None, overwrite=False, cardinality=64, dimx=1, dimy=400, 
				nsteps=300, learningrate=0.1):
    ''' Function to train a SOM
        
    Parameters
    ----------------
	SOMarrayfile:	str
		Filepath to txt file containing SOMarray

	outfile:		str, optional
		Filepath to save array to. If not populated, just returns array
	overwrite
	dimx
	dimy         
    nsteps:			int, optional
        number of training steps for SOM
        
    learningrate:	float, optional
        parameter for SOM, controls speed at which it changes. Between 0 and 1.
        	
    Returns
    -----------------
    som object:		object
        Trained som
    '''
    from . import selfsom
        
    cardinality = SOMarray.shape[1]
        
    def Init(sample):
        return np.random.uniform(0,1,size=(dimx,dimy,cardinality))
        
    som = selfsom.SimpleSOMMapper((dimx,dimy),nsteps,initialization_func=Init,
        							learning_rate=learningrate)
    som.train(SOMarray)
    if outfile:
        if not os.path.exists(outfile) or overwrite:
            kohonenSave(som.K,outfile)
    return som

def EBperiod(time, flux, per, cut_outliers=0, linflatten=True):
    """
    Tests for phase variation at double the current prime period,
    to correct EB periods.

    Inputs
    -----------------
    time
    flux
    per: 			float
        Period to phasefold self.lc at.
    cut_outliers:	float
        outliers ignored if difference from median in bin divided by the MAD is 
        greater than cut_outliers.
        	        
    Returns
    -----------------
    corrected period: float
        Either initial period or double      
    """
    if per < 0:
        return per
    if linflatten:
        flux_flat = flux - np.polyval(np.polyfit(time,flux,1),time) + 1
    else:
        flux_flat = flux
        
    phaselc2P = np.zeros([len(time),2])
    phaselc2P = phasefold(time,per*2)
    idx = np.argsort(phaselc2P)
    phaselc2P = phaselc2P[idx]
    flux = flux_flat[idx]
    binnedlc2P = binPhaseLC(phaselc2P,flux,64,cut_outliers=5) #ADD OUTLIER CUTS?

    minima = np.argmin(binnedlc2P[:,1])
    posssecondary = np.mod(np.abs(binnedlc2P[:,0]-np.mod(binnedlc2P[minima,0]+0.5,1.)),1.)
    #within 0.05 either side of phase 0.5 from minima
    posssecondary = np.where((posssecondary<0.05) | (posssecondary > 0.95))[0]  

    pointsort = np.sort(flux)
    top10points = np.median(pointsort[-30:])
    bottom10points = np.median(pointsort[:30])
        
    periodlim = (np.max(time)-np.min(time))/2. #no effective limit, could be changed
    if np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1] > 0.0025 \
    and np.min(binnedlc2P[posssecondary,1]) - binnedlc2P[minima,1] \
        > 0.03*(top10points-bottom10points) \
    and per*2<=periodlim:  
        return 2*per
    else:
        return per

def phasefold(time,per,t0=0):
    return np.mod(time-t0,per)/per

def binPhaseLC(phase, flux, nbins, cut_outliers=0):
    """
    Bins a lightcurve, typically phase-folded.

    Inputs
    -----------------
    phase: 			ndarray, N
        Phase data (could use a time array instead)
    flux:			ndarray, N
        Flux data
    nbins:			int
        Number of bins to use
    cut_outliers:	float
        If not zero, cuts outliers where (difference to median)/MAD > cut_outliers 
        	        
    Returns
    -----------------
    binnedlc:		ndarray, (nbins, 2)    
        Array of (bin phases, binned fluxes)
    """
    bin_edges = np.arange(nbins)/float(nbins)
    bin_indices = np.digitize(phase,bin_edges) - 1
    binnedlc = np.zeros([nbins,2])
    #fixes phase of all bins - means ignoring locations of points in bin
    binnedlc[:,0] = 1./nbins * 0.5 +bin_edges  
    for bin in range(nbins):
        if np.sum(bin_indices==bin) > 0:
            inbin = np.where(bin_indices==bin)[0]
            if cut_outliers and np.sum(bin_indices==bin)>2:
                mad = np.median(np.abs(flux[inbin]-np.median(flux[inbin])))
                outliers = np.abs((flux[inbin] - np.median(flux[inbin])))/mad <= cut_outliers
                inbin = inbin[outliers]
            binnedlc[bin,1] = np.mean(flux[inbin])  
        else:
            #bit awkward this, but only alternative is to interpolate?
            binnedlc[bin,1] = np.mean(flux)  
    return binnedlc

def prepFilePhasefold(time, flux, period, cardinality):
    """
    Prepares a lightcurve for using with the SOM.

    Inputs
    -----------------
    time
    flux
    period: 			float
        Period to phasefold self.lc at
    cardinality:		int
        Number of bins used in SOM
        	        
    Returns
    -----------------
    binnedlc:		ndarray, (cardinality, 2)    
        Array of (bin phases, binned fluxes)
    range:			float
        Max - Min if binned lightcurve
    """
    phase = phasefold(time,period)
    idx = np.argsort(phase)
    binnedlc = binPhaseLC(phase[idx],flux[idx],cardinality)
    #normalise to between 0 and 1
    minflux = np.min(binnedlc[:,1])
    maxflux = np.max(binnedlc[:,1])
    if maxflux != minflux:
        binnedlc[:,1] = (binnedlc[:,1]-minflux) / (maxflux-minflux)
    else:
        binnedlc[:,1] = np.ones(cardinality)
    #offset so minimum is at phase 0
    binnedlc[:,0] = np.mod(binnedlc[:,0]-binnedlc[np.argmin(binnedlc[:,1]),0],1)
    binnedlc = binnedlc[np.argsort(binnedlc[:,0]),:]
    return binnedlc[:,1],maxflux-minflux

def SOMloc(som, time, flux, per, cardinality): 
    """
    Returns location on the current som for given lc,
    and binned amplitude.
        
    Inputs
    -----------------
    som
    per: 			float
        Period to phasefold the lightcurve at.
    time
    flux
    cardinality

    Returns
    -----------------
    map: 	int
         Location on SOM (assumes 1d SOM).
    range: float
        Amplitude of the binned phase-folded lightcurve
    """
    if per < 0:
        return -10
    SOMarray,range = prepFilePhasefold(time, flux, per, cardinality)
    SOMarray = np.vstack((SOMarray,np.ones(len(SOMarray)))) #tricks som code into thinking we have more than one
    map = som(SOMarray)
    map = map[0,1]
    return map, range
    
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

def phase_features(time, flux, per):
    """
    Returns p2p features connected to phase fold

    Inputs
    -----------------
    time
    flux
    per: 			float
        Period to phasefold lc at.
          	
    Returns
    -----------------
    p2p 98th percentile: 	float
        98th percentile of point-to-point differences of phasefold
    p2p mean:				float
        Mean of point-to-point differences of phasefold
         	
    """
    phase = phasefold(time,per)
    p2p = np.abs(np.diff(flux[np.argsort(phase)]))
    return np.percentile(p2p,98),np.mean(p2p)

def p2p_features(flux):
    """
    Returns p2p features on lc

    Inputs
    -----------------
    flux
    
    Returns
    -----------------
    p2p 98th percentile: 	float
        98th percentile of point-to-point differences of lightcurve
    p2p mean:				float
        Mean of point-to-point differences of lightcurve
         	
    """
    p2p = np.abs(np.diff(flux))
    return np.percentile(p2p,98),np.mean(p2p)

def compute_fill(x):
   cadence = np.median(np.diff(x))
   full_npts = (x[-1] - x[0]) / cadence
   return len(x[np.isfinite(x)]) / full_npts

def compute_k_crossings(y, k):
    y = np.diff(y, k)
    window = y.copy()
    window[window >= 0] = 1
    window[window < 0] = 0
    return np.sum(np.diff(window)**2)

def sigmoid_(x, t):
    # Normalisation factor functions
    return (1 / (1 + np.exp(-t*x))) - 0.5

def sig_single(x, t):
    return sigmoid_(x, t) / sigmoid_(1,t)

def correction_factor(x, t, k):
    sig = sig_single(x, t)
    return sig# * self.coeffs[k]


def compute_hocs(x, y, k):
    """
    Compute higher order crossings (HOC)
    Parameters
    -----------
    k (int) : number of k crossings to compute (inclusive)
    """
    y_offset = y - np.median(y)
    
    sigmoid_coeffs = np.array([3.625418060200669,
                           5.250836120401338,
                           7.117056856187291,
                           8.862876254180602,
                           10.608695652173914,
                           12.234113712374581])

    zc_gauss = np.array([ 0.49912 ,  0.666367,  0.732079,  0.769869,  0.795083,  0.81284 ,
    0.827098,  0.838291,  0.847576,  0.855565,  0.862341,  0.868062,
    0.873031,  0.877556,  0.881678])
    zc = np.zeros(k)
    fill = compute_fill(x)
    for i in range(k):
        zc[i] = compute_k_crossings(y_offset, i) / (len(y_offset)-i)/correction_factor(fill, sigmoid_coeffs[i], i)
    delta_k = zc[0]
    delta_k_gauss = zc_gauss[0]
    for i in range(k-1):
        delta_k = np.append(delta_k, zc[i+1] - zc[i])
        delta_k_gauss = np.append(delta_k_gauss, zc_gauss[i+1] - zc_gauss[i])
    psi = np.sum((delta_k - delta_k_gauss[:len(delta_k)])**2 / delta_k_gauss[:len(delta_k)])
    return psi, zc
    

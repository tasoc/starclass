#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:17:00 2018

@author: Lisa Bugnet
@contact: lisa.bugnet@cea.fr
This code is the property of L. Bugnet (please see and cite Bugnet et al.,2018).

The user should use the FLIPER class to calculate FliPer values
from 0.2,0.7,7,20 and 50 muHz.

A calling example is reported at the beginning of the code
"""


from astropy.io import fits
import numpy as np
import os, os.path
from math import *


### CALLING SEQUENCE:
# Fliper_values=FLIPER().Fp(star_tab_psd)
# print(Fliper_values.fp07)
# print(Fliper_values.fp7)
# print(Fliper_values.fp20)
# print(Fliper_values.fp50)

class FLIPER:
#"""Class defining the FliPer"""

    def __init__(self):
        self.nom = "FliPer"
        self.id=[]
        self.fp07=[]
        self.fp7=[]
        self.fp20=[]
        self.fp02=[]
        self.fp50=[]


    def Fp(self, star_tab_psd, kepmag=0):
        """
        Compute FliPer values from 0.7, 7, 20, & 50 muHz
        star_tab_psd[0] contains frequencies in muHz
        star_tab_psd[1] contains power density in ppm2/muHz
        star_tab_psd can be computed using the LC_TO_PSD class if needed
        """
        #star_tab_psd=   self.APODIZATION(star_tab_psd)
        end         =   277# muHz
        noise       =   self.HIGH_NOISE(star_tab_psd)
        Fp07_val    =   self.REGION(star_tab_psd, 0.7, end) - noise
        Fp7_val     =   self.REGION(star_tab_psd, 7, end)   - noise
        Fp20_val    =   self.REGION(star_tab_psd, 20, end)  - noise
        Fp50_val    =   self.REGION(star_tab_psd, 50, end)  - noise

        self.fp07.append(Fp07_val)
        self.fp7.append(Fp7_val)
        self.fp20.append(Fp20_val)
        self.fp50.append(Fp50_val)
        return self

    def REGION(self,star_tab_psd,inic,end):
        """
        Function that calculates the average power in a given frequency range on PSD
        """
        x       =   np.float64(star_tab_psd[0]) # convert frequencies in muHz
        y       =   np.float64(star_tab_psd[1])
        ys      =   y[np.where((x >= inic) & (x <= end))]
        average =   np.mean(ys)
        return average

    def HIGH_NOISE(self,star_tab_psd):
        """
        Function that computes photon noise from last 100 bins of the spectra
        """
        data_arr_freq   =   star_tab_psd[0]
        data_arr_pow    =   star_tab_psd[1]
        siglower        =   np.mean(data_arr_pow[-100:])
        return siglower

    def APODIZATION(self,star_tab_psd):
        """
        Function that corrects the spectra from apodization
        """
        power   =   star_tab_psd[1]
        freq    =   star_tab_psd[0]
        nq      =   max(freq)
        nu      =   np.sin(np.pi/2.*freq/nq) / (np.pi/2.*freq/nq)
        power   =   power/(nu**2)

        star_tab_psd    =   list(star_tab_psd)
        star_tab_psd[1] =   power
        star_tab_psd[0] =   freq
        return star_tab_psd

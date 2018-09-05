
import numpy as np
import gatspy
from gatspy.periodic import LombScargleFast
import matplotlib.pyplot as plt
from astropy.stats import mad_std

class Dataset(object):
    def __init__(self, epic, data_file=None):
        ''' Initial contruct for the Dataset object
        
        Parameters
        ----------------
        epic: int
            The epic number for the source
            
        data_file: str
            The path to the file containg the data
            
        Returns
        -----------------
        NA
        
        Examples
        -----------------
        
        This is just creating the object, so for epic='2001122017' and 
        data file of '/home/davies/Data/ktwo_2001122017_llc.pow' you would run:
        
        >>> import K2data
        >>> star = K2data.Dataset(2001122017, '/home/davies/Data/ktwo_2001122017_llc.pow')
        
        '''
        
        self.epic = epic
        self.data_file = data_file
        self.time = []
        self.flux = []
        self.time_fix = []
        self.flux_fix = []
        self.freq = []
        self.power = []
        self.smoo_freq = []
        self.smoo_power = []

    def read_timeseries(self, verbose=False, sigma_clip=5):
        '''  Reads in a timeseries from the file stored in data file.
        This works for ascii files that can be read by np.genfromtxt. The 
        function assumes that time is in the zero column and flux is in the
        one column.
        
        Time should be in units of days.  
        
        The data is read in, zero values are removed, and stored in the time and flux.
        
        A sigma clip is performed on the flux to remove extreme values.  The 
        level of the sigma clip can be adjusted with the sigma_clip parameter.
        The results of the sigma clip are stored in time_fix and flux_fix.
                
        
        Parameters
        ------------------
        verbose: Bool(False)
            Set to true to produce verbose output.
            
        sigma_clip: Float
            The level at which to perform the sigma clip.  If sigma_clip=0 
            then no sigma is performed.
            
            
        Returns
        ------------------
        NA
        
        Examples
        ------------------
        
        To load in the time series with a 5 sigma clip, one would run:
            
        >>> import K2data
        >>> star = K2data.Dataset(2001122017, '/home/davies/Data/ktwo_2001122017_llc.pow')
        >>> star.read_timeseries()
        
        '''
        data = np.genfromtxt(self.data_file)
        self.time = (data[:,0] - data[0,0]) * 24.0 * 3600.0
        self.flux = ((data[:,1] / np.nanmedian(data[:,1]) - 1.0) *1e6)
        self.time = self.time[np.isfinite(self.flux)]
        self.flux = self.flux[np.isfinite(self.flux)]
        
        self.time = self.time[self.flux != 0]
        self.flux = self.flux[self.flux != 0]

        self.flux = self.flux[np.argsort(self.time)]
        self.time = self.time[np.argsort(self.time)]
        
        self.flux_fix = self.flux
        sel = np.where(np.abs(self.flux_fix) < mad_std(self.flux_fix) * sigma_clip)
        self.flux_fix = self.flux_fix[sel]
        self.time_fix = self.time[sel]  
        if verbose:
            print("Read file {}".format(self.data_file))
            print("Data points : {}".format(len(self.time)))

    def read_psd(self):
        ''' This function reads in a power spectrum for self.data_file.  This 
        module currently supports .txt, .pow, and .fits. The frequencies and 
        power are stored in the self.freq and self.power object properties.
        
        .txt files must have frequency in the zero column and power in the one
        column.  Frequency is expected to be in units of Hz and will be stored
        as muHz.
        
        .pow files must have frequency in the zero column and power in the one column.  Frequency is expected to be in units of muHz and will be stored as muHz.
        
        .fits files must have frequency in the zero column and power in the one column of the data object.  Frequency is expected to be in units of Hz and will be stored as muHz.
        
        Parameters
        ----------------
        
        NA
        
        Returns
        ----------------
        NA
        
        Examples
        ----------------
        
        To read in a power spectrum:
        
        >>> import K2data
        >>> star = K2data.Dataset(2001122017, '/home/davies/Data/ktwo_2001122017_llc_psd_v1.pow')
        >>> star.read_psd()
            
        '''
        if self.data_file.endswith('.txt'):
            data = np.genfromtxt(self.data_file)
            self.freq = data[:,0]*1e6
            self.power = data[:,1]
        elif self.data_file.endswith('.pow'):
            data = np.genfromtxt(self.data_file)
            self.freq = data[:,0]
            self.power = data[:,1]
        elif self.data_file.endswith('.fits'):
            import pyfits
            data = pyfits.getdata(self.data_file)
            data = np.array(data)
            self.freq = data[:,0]*1e6
            self.power = data[:,1]
        else:
            print("File type not supported!")
    
    def power_spectrum(self, verbose=False, noise=0.0, \
                       length=-1):
        ''' This function computes the power spectrum from the timeseries.
        
        The function checks to see if the timeseries has been read in, and if not it calls the read_timeseries function.
        
        The porperties of the power spectrum can be altered for a given timeseries via the noise, and length parameters.
        
        The frequency and power are stored in the object atributes self.freq and self,power.
        
        Parameters
        ----------------
        verbose: Bool(False)
            Provide verbose output if set to True.
            
        noise: Float
            If noise is not zero then additional noise is added to the timeseries where the value of noise is the standard deviation of the additional noise.

        length: Int
            If length is not -1 then a subset of the timeseries is selected when n points will equal length.  The subset of data is taken from the start of the time series.  TODO this can be updated if neccessary.

        Returns
        ----------------
        
        NA
        
        Examples
        ----------------
        
        To read in a data set and create the power spectrum one need only run:
            
        >>> import K2data
        >>> star = K2data.Dataset(2001122017, '/home/davies/Data/ktwo_2001122017_llc.pow')
        >>> star.power_spectrum()
        
        '''
        if len(self.time) < 1:
            self.read_timeseries(verbose=True)
        if noise > 0.0:
            self.flux_fix[:length] += np.random.randn(len(self.time_fix[:length])) * noise
        dtav = np.mean(np.diff(self.time_fix[:length]))
        dtmed = np.median(np.diff(self.time_fix[:length]))
        if dtmed == 0:
            dtmed = dtav
        fmin = 0
        N = len(self.time_fix[:length]) #n-points
        df = 1.0 / dtmed / N #bw
        model = LombScargleFast().fit(self.time_fix[:length], \
                                    self.flux_fix[:length], \
                                      np.ones(N))
        power = model.score_frequency_grid(fmin, df, N/2)
        freqs = fmin + df * np.arange(N/2)
        var = np.std(self.flux_fix[:length])**2
        power /= np.sum(power)
        power *= var
        power /= df * 1e6
        if len(freqs) < len(power):
            power = power[0:len(freqs)]
        if len(freqs) > len(power):
            freqs = freqs[0:len(power)]
        self.freq = freqs * 1e6
        self.power = power
        if verbose:
            print("Frequency resolution : {}".format(self.freq[1]))
            print("Nyquist : ~".format(self.freq.max()))


    def plot_power_spectrum(self, smoo=0, plog=True, ax=[]):
        ''' Plots the power spectrum '''
        if len(self.freq) < 0:
            self.power_spectrum()
        if ax == []:
            fig, ax = plt.subplots()
        ax.plot(self.freq, self.power, 'b-')
        if plog:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel(r'Frequency ($\rm \mu Hz$)')
        ax.set_ylabel(r'Power ($\rm ppm^{2} \, \mu Hz^{-1}$)')
        ax.set_xlim([self.freq.min(),self.freq.max()])
        ax.set_title('KIC ' + str(self.epic))
        if smoo > 0:
            self.rebin_quick(smoo)
            ax.plot(self.smoo_freq, self.smoo_power, 'k-', linewidth=4)
        if ax == []:
            fig.savefig('power_spectrum_' + str(self.epic) + '.png')

    def plot_timeseries(self):
        ''' Plots the time series '''
        if len(self.time) < 0:
            self.read_data()
        fig, ax = plt.subplots()
        ax.plot(self.time, self.flux, 'b.')
        ax.plot(self.time_fix, self.flux_fix, 'k.')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Flux (ppm)')
        ax.set_title('KIC ' + str(self.epic))
        fig.savefig('timeseries_' + str(self.epic) + '.png')

    def rebin_quick(self, smoo):
        ''' TODO Write DOC strings '''
        if smoo < 1:
            return f, p
        if self.freq == []:
            self.power_spectrum()
        self.smoo = int(smoo)
        m = int(len(self.power) / self.smoo)
        self.smoo_freq = self.freq[:m*self.smoo].reshape((m,self.smoo)).mean(1)
        self.smoo_power = self.power[:m*self.smoo].reshape((m,self.smoo)).mean(1)


        

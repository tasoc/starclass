#star_path_LC='/Users/lbugnet/DATA/METRIC/TESS/clean_files_TDA3_1/Star3.clean'
#star_number=3
#flux=CONVERT().get_ts(star_path_LC, star_number)[1]
#time=CONVERT().get_ts(star_path_LC, star_number)[0]
#star_tab_psd=CONVERT().compute_ps(time, flux, star_path_LC, star_number)

#plt.loglog(star_tab_psd[0],star_tab_psd[1])
#plt.show()

#np.savetxt('/Users/lbugnet/DATA/TABLES/share_spectrum_cl3.txt', np.c_[star_tab_psd[0],star_tab_psd[1]], fmt=['%-s','%-s'], delimiter=' ',comments='')
import numpy as np
import gatspy.periodic as gp
from gatspy.periodic import LombScargleFast

class CONVERT:
    def get_ts(self, star_path_LC, ii):
        """
        Import data
        """
        # Open text file
        time, flux, flag= np.loadtxt(star_path_LC, unpack=True)
        tottime=np.max(time)-np.min(time)
        flux=((flux/np.nanmedian(flux))-1)*1e6 # convert flux from e- to ppm !!!!
        # Remove nans and prepare time array
        sel = np.where(np.isnan(flux)==True)
        flux[sel] = 0.0
        time -= time[0]
        time *= 86400.0#put in sec
        cadence=np.median(np.diff(time))
        # plt.plot(time, flux)
        # plt.show()
        return time, flux, cadence, tottime

    def normalise(self, time, flux, f, p, bw):
        """
        Normalise according to Parseval's theorem
        """
        rhs = 1.0 / len(flux) * np.sum(flux**2.0)
        lhs = p.sum()
        ratio = rhs / lhs
        return p * ratio / bw / 1e6

    def compute_ps(self, time, flux, dt, tottime):
        """
        Compute power spectrum using gatspy fast lomb scargle
        """
        #print(dt)
        dt=float(dt)
        tottime=float(tottime)*86400.0#put in sec
        # Nyquist frequency
        nyq = 1.0 / (2.0*dt)
        # Frequency bin width
        df = 1.0 / tottime
        # Number of frequencies to compute
        Nf = nyq / df
        # Compute psd
        f, p = gp.lomb_scargle_fast.lomb_scargle_fast(time, flux,
                                                      f0=df,
                                                      df=df, Nf=Nf,
                                                      use_fft=True)
        # Calibrate power
        p = self.normalise(time, flux, f, p, df)
        return f*1e6, p

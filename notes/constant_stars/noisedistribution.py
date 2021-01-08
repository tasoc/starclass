import astropy.constants as const
import astropy.units as u
from astroquery.mast import Catalogs
import noiseestimation as ne
import numpy as np
import glob, os

#TESS white noise lightcurve simulator
def TIC_byID(ID):
    """
    """
    catTable = Catalogs.query_criteria(ID=ID, catalog="Tic")

    return catTable['ID','ra','dec','Tmag','Teff']
    

#generate random integer
num = 0
cadence = 30*60. #30 min cadence
whitenoise = []
teffdist = []
tmagdist = []
radist = []
decdist = []
#
while num < 1000:
    id = np.random.randint(0,472000000.)
    catdata = TIC_byID(id)
    if len(catdata)>1:
        print(catdata)
    if len(catdata==1):
        if catdata['Tmag'] < 15 and catdata['Teff'] < 9700. and catdata['Teff'] > 2450:
            PARAM = {}
            PARAM['RA'] = catdata['ra']
            PARAM['DEC'] = catdata['dec']
            noise, PARAM = ne.phot_noise(catdata['Tmag'], catdata['Teff'], cadence, PARAM)
            totnoise = np.sqrt(noise[0]**2+noise[1]**2+noise[2]**2+noise[3]**2)
            whitenoise.append(totnoise[0])
            tmagdist.append(catdata['Tmag'][0])
            teffdist.append(catdata['Teff'][0])
            radist.append(catdata['ra'][0])
            decdist.append(catdata['dec'][0])
            num += 1      
            print(num)  
    else:
        print('ID not in TIC')

whitenoise = np.array(whitenoise)
tmagdist = np.array(tmagdist)
teffdist = np.array(teffdist)
radist = np.array(radist)
decdist = np.array(decdist)


np.savetxt('whitenoisedist.txt',whitenoise)
np.savetxt('tmagdist.txt',tmagdist)
np.savetxt('teffdist.txt',teffdist)
np.savetxt('radist.txt',radist)
np.savetxt('decdist.txt',decdist)

kepdata = '/Users/davidarmstrong/Software/Python/TESS/batch06_noisy/'
keplist = glob.glob(os.path.join(kepdata,'*.noisy'))
import pylab as p
p.ion()

for i,lcjit in enumerate(whitenoise):
    print(i)
    #load random kepler lc.
    kepfile = np.genfromtxt(np.random.choice(keplist))
    timestamps = kepfile[:,0]
    lcjit_relflux = lcjit*1e-6
    flux = np.random.normal(1,lcjit_relflux,len(timestamps))
    flux_err = np.ones(len(timestamps)) * lcjit_relflux
    output = np.array([timestamps,flux,flux_err]).T
    #p.figure(1)
    #p.clf()
    #p.errorbar(timestamps,flux,yerr=flux_err,fmt='b.')
    #p.pause(1)
    #input('a')
    np.savetxt('constantlcs/constant_'+str(i)+'.txt',output)

#get Tmag, Teff, RA, DEC



#add noise sources

#define time baseline

#make curve






#nsamples = 1000
#Tmagdist = np.random.uniform(5.,15.,nsamples)
#Teffdist = np.random.uniform(2450.,9700.,nsamples)
#RAdist = np.random.uniform(0.,360.,nsamples)
#DECdist = np.random.uniform(-89.,-20.,nsamples)

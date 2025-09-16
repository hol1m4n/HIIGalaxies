# Reads and fit the redshift and flux from sdss spectra
import matplotlib
matplotlib.use('Agg')

import os
from os import listdir
from os.path import isfile, join
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord

def spread(spath):
    hdul = fits.open(spath)
    data = hdul[1].data

    # print hdul.info()
    # print(repr(hdul[1].header))
    RA = hdul[1].header['RAOBJ']
    DEC = hdul[1].header['DECOBJ']

    vflux = data.flux
    vwave = data.wavelength

    c = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
    jname =  'J{0}{1}'.format(c.ra.to_string(unit=u.hourangle, sep='' \
          , precision=0, pad=True), c.dec.to_string(sep='', precision=0 \
          , alwayssign=True, pad=True))
    return jname, vwave, vflux

# Main Code
mcd = os.path.dirname(os.path.abspath(__file__))
dpath = mcd + '/dat/'

spath = dpath + 'indat/spec/DR8spectra/'
nspec = [f for f in listdir(spath) if isfile(join(spath, f))]

for i in nspec:
    ispath = spath + i
    jname, vx, vy = spread(ispath)
    print jname
    # break

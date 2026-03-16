# Reads and fit the redshift and flux from sdss spectra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from os import listdir
from os.path import isfile, join
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
import numpy as np
from scipy.signal import find_peaks

def spread(spath):
    hdul = fits.open(spath)
    data = hdul[1].data

    # print hdul.info()
    # print(repr(hdul[1].header))
    RA = hdul[1].header['RAOBJ']
    DEC = hdul[1].header['DECOBJ']
    Z = hdul[1].header['Z']
    ZERR = hdul[1].header['Z_ERR']

    vflux = data.flux
    vwave = data.wavelength

    c = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
    jname =  'J{0}{1}'.format(c.ra.to_string(unit=u.hourangle, sep='' \
          , precision=0, pad=True), c.dec.to_string(sep='', precision=0 \
          , alwayssign=True, pad=True))
    return jname, vwave, vflux

def redshift(dpath, lines):
    lpath = dpath + 'indat/lines.dat'
    ldat = Table.read(lpath, format='csv')
    elin = ldat['Air']
    olin = sorted(lines, reverse = True)
    ix = min(len(ldat), len(olin))
    vz = (olin[0:ix] - elin[0:ix])/elin[0:ix]
    z = np.mean(vz)
    zErr = np.std(vz)
    return z, zErr

# Main Code
mcd = os.path.dirname(os.path.abspath(__file__))
dpath = mcd + '/dat/'

spath = dpath + 'indat/spec/DR8spectra/'
nspec = [f for f in listdir(spath) if isfile(join(spath, f))]

for i in nspec:
    ispath = spath + i
    # print ispath
    jname, vx, vy = spread(ispath)
    print jname
    peaks, _ = find_peaks(vy, height = 0.15*max(vy))

    olines = vx[peaks]

    z, zErr = redshift(dpath, olines)

    # Ploting
    path = dpath + 'results/sp' + jname + '.pdf'

    xmin = 3500.0
    xmax = 9500.0
    ymin = min(vy) - 0.05*max(vy)
    ymax = max(vy) + 0.05*max(vy)

    xl = "$\lambda\ (\mathrm{\AA})$"
    yl = "$f(\lambda)\ (10^{-17} \mathrm{erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$)"

    plt.figure()
    plt.plot(vx, vy, '-', lw = 1, c = 'black', drawstyle='steps')
    plt.plot(vx[peaks], vy[peaks], "x", markersize=8)
    plt.text(3550.0, ymax - 0.05*ymax, r'z = ' + str("%5.4f" % z) + \
             r'$\pm$' + str("%5.5f" % zErr))
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel(xl, fontsize=14)
    plt.ylabel(yl, fontsize=14)
    plt.savefig(path)
    plt.close()
    # break

import os
from astropy.table import Table
from astropy.cosmology import FlatwCDM
import numpy as np
import scipy.optimize as op


def lnlike(theta, x, y, z, xerr, yerr, zerr):
    alpha, beta, h0 = theta

    Om = 0.3
    w0 = -1.0

    cosmo = FlatwCDM(H0=h0*100, Om0=Om, w0=w0)

    ixG = np.where(z > 10)
    ixH = np.where(z < 10)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2)

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    llq = -0.5*xsq
    return (llq, xsq, R, Mum)


def h2gdat(dpath):
    tpath = dpath + 'indat/h2data.dat'
    data = Table.read(tpath, format='ascii')

    vx = data['col4']
    vxErr = data['col5']
    vy = data['col6']
    vyErr = data['col7']
    vz = data['col2']
    vzErr = data['col3']
    return vx, vxErr, vy, vyErr, vz, vzErr


# Main Code
mcd = os.path.dirname(os.path.abspath(__file__))
dpath = mcd + '/dat/'

# Reading Data
vx, vxErr, vy, vyErr, vz, vzErr = h2gdat(dpath)

# First guess
ialpha = 32.0
ibeta = 5.0
ih0 = 0.75

# Maximum Likelihood
nll = lambda *args: -lnlike(*args)[0]
result = op.minimize(nll, [ialpha, ibeta, ih0],
                     args=(vx, vy, vz, vxErr, vyErr, vzErr))
alpha_ml, beta_ml, h0_ml = result["x"]

print 'max llq'
print alpha_ml
print beta_ml
print h0_ml

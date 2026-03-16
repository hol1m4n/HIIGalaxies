import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from astropy.table import Table
from astropy.cosmology import FlatwCDM
import emcee
import numpy as np
import corner
import scipy.optimize as op


def lnlike(theta, x, y, z, xerr, yerr, zerr):
    alpha, beta, h0, Om = theta

    if Om <= 0.01: Om = 0.01

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


def lnprior(theta):
    alpha, beta, h0, Om = theta
    if (0.0 <= beta <= 10.0 and 20.0 <= alpha <= 40.0 and 0.5 <= h0 <= 1.0
            and 0.01 <= Om <= 1.0):
        return 0.0
    return -np.inf


def lnprob(theta, x, y, z, xerr, yerr, zerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, z, xerr, yerr, zerr)[0]


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
iOm = 0.2

# Maximum Likelihood
nll = lambda *args: -lnlike(*args)[0]
result = op.minimize(nll, [ialpha, ibeta, ih0, iOm],
                     args=(vx, vy, vz, vxErr, vyErr, vzErr))
alpha_ml, beta_ml, h0_ml, Om_ml = result["x"]

print 'max llq'
print alpha_ml
print beta_ml
print h0_ml
print Om_ml

# MCMC
ndim, nwalkers = 4, 50
pos = [[alpha_ml, beta_ml, h0_ml, Om_ml] + 1e-5*np.random.randn(ndim)
       for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                args=(vx, vy, vz, vxErr, vyErr, vzErr),
                                threads=8)

# Run 100 steps as a burn-in.
pos, prob, state = sampler.run_mcmc(pos, 100)

# Reset the chain to remove the burn-in samples.
sampler.reset()

# Starting from the final position in the burn-in chain
sampler.run_mcmc(pos, 500, rstate0=state)

print("Mean acceptance fraction: {0:.3f}"
      .format(np.mean(sampler.acceptance_fraction)))

samples = sampler.flatchain
lnprob = sampler.flatlnprobability

alpha_mcmc, beta_mcmc, h0_mcmc, Om_mcmc = map(
                                 lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                 zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
print 'mcmc'
print alpha_mcmc
print beta_mcmc
print h0_mcmc
print Om_mcmc
print len(vx)

# Plot
path = dpath+'results/trploth0.pdf'
fig = corner.corner(samples,
                    labels=["$\\alpha$", "$\\beta$", "$h_0$", "$\Omega_m$"],
                    truths=[alpha_mcmc[0],  beta_mcmc[0], h0_mcmc[0],
                            Om_mcmc[0]],
                    quantiles=[0.16, 0.84],
                    plot_contours='true',
                    levels=1.0 - np.exp(
                        -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                    smooth=1.0,
                    bins=100,
                    color='black',
                    show_titles=1)
plt.savefig(path)

#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  :Name:     mch2gv69.py
#  :Purpose:  MCMC for HII galaxies KMOS sample
#  :Author:   Ricardo Chavez -- Jun 8th, 2020 -- Morelia, Mexico
#  :Notes: testing samples, sampling_efficiency: 0.8, subsample testing
#  ------------------------------------------------------------------------------
from astropy.table import Table
from astropy.visualization import hist
from astropy.cosmology import FlatwCDM
from astropy.cosmology import Flatw0waCDM
from astropy.cosmology import LambdaCDM
import scipy.stats as st
import numpy as np
from astropy import constants as const
from pymultinest.solve import solve
import corner
from getdist import plots, MCSamples
import datetime
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import pandas as pd

c = const.c.to('km/s').value
k_b = const.k_B.to('J/K').value
m_p = const.m_p.to('kg').value

EBVG = 0.184*1.2
EBVGErr = 0.048

EBVC = 0.1212*1.2
EBVCErr = 0.037

LSL = 1.83  # 1.84
x0 = 1.79
x0Err = 0.08


def lsprior(cube):
    cube[0] = cube[0]*2. + 32.5
    cube[1] = cube[1]*1. + 4.5
    return cube


def lslike(cube):
    alpha, beta = cube[0], cube[1]
    theta = alpha, beta
    xsq = lschsq(theta)[0]
    return -0.5*xsq


def lschsq(theta):
    alpha, beta = theta
    x, y, xerr, yerr = lsdata

    ym = beta*x + alpha
    ymerr = np.sqrt(beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            # + 0.22**2
            )

    R = (y - ym)
    W = 1.0/(yerr**2 + ymerr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if lsfr == 2:
        return (xsq2, R, ym, ymerr)
    else:
        return (xsq, R, ym, ymerr)


def lsfit(dpath, cpath, fend, x, xerr, y, yerr):
    global lsdata
    global lsfr
    lsfr = 1

    lsdata = x, y, xerr, yerr
    parameters = [r"\alpha",r"\beta"]
    parametersc = [r"$\alpha$",r"$\beta$"]
    n_params = len(parameters)

    fend = fend + '_LS_'
    prefix = cpath + fend

    sps = 10000
    vbs = 1
    prs = 0

    result = solve(LogLikelihood=lslike, Prior=lsprior,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'LS'
    # ploter(dpath, fend, result, parameters, parametersc, tag, ctag)

    path = dpath+'results/GDtr'+fend+'.pdf'
    GDsamples = MCSamples(samples=result['samples'], names=parameters,
                          labels=parameters, name_tag=tag #,
                          # ranges={r"\Omega_m":[0.0, None]}
                          )

    g = plots.getSubplotPlotter() #width_inch=4
    g.settings.num_plot_contours = 2
    g.triangle_plot(GDsamples, filled=True, title_limit=1,
            contour_colors=['crimson'],
            param_limits={r"\alpha":[32.8, 33.7]
                        # , r"\beta":[4.7, 5.35]
                        # , r"h":[0.6, 0.85]
                        # , r"\Omega_m":[0.0, 0.52]
                        # , r"w_0":[-2.1, -0.2]
                        }
            )
    g.export(path)

    t = GDsamples.getTable(limit=1).tableTex()
    theta = GDsamples.getMeans()

    # Print parameter values
    print('parameter values:')
    for name, col in zip(parameters, result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

    print(t)

    fy = theta[1]*x + theta[0]

    return fy


def ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    fchsq = 'chsq' + str(ctag)
    n_params = len(parameters)
    nobs = len(x)
    print(fchsq, nobs)

    # Plot getdist
    path = dpath+'results/GDtr'+fend+'.pdf'
    GDsamples = MCSamples(samples=result['samples'], names=parameters,
                          labels=parameters, name_tag=tag,
                          ranges={r"\Omega_m":[0.0, None]}
                          )

    g = plots.getSubplotPlotter() #width_inch=4
    g.settings.num_plot_contours = 2
    g.triangle_plot(GDsamples, filled=True, title_limit=1,
            contour_colors=['crimson'],
            param_limits={r"\alpha":[32.8, 33.7]
                        # , r"\beta":[4.7, 5.35]
                        , r"h":[0.6, 0.85]
                        # , r"\Omega_m":[0.0, 0.52]
                        # , r"w_0":[-2.1, -0.2]
                        }
            )
    g.export(path)

    t = GDsamples.getTable(limit=1).tableTex()
    theta = GDsamples.getMeans()

    chsq = globals()[fchsq](theta)[0]
    res = globals()[fchsq](theta)[1]

    # Cov Matix and FoM
    cpars=[r"\Omega_m", r"w_0"]
    # FoM = 1./np.sqrt(np.linalg.det(GDsamples.cov(pars=cpars)))
    FoM = 1./np.sqrt(np.linalg.det(GDsamples.cov()))
    print('FoM = ', FoM)

    # Plot Corner
    path = dpath+'results/tr'+fend+'.pdf'
    fig = corner.corner(result['samples'],
                        labels=parametersc,
                        plot_contours='true',
                        levels=1.0 - np.exp(
                            -0.5 * np.arange(1.0, 3.1, 1.0) ** 2),
                        smooth=1.0,
                        bins=100,
                        color='black',
                        show_titles=1)
    plt.savefig(path)

    # Print parameter values
    print('parameter values:')
    for name, col in zip(parameters, result['samples'].transpose()):
        print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

    print(t)

    # print(GDsamples.getBestFit(max_posterior=True))
    # print(GDsamples.getBestFit(max_posterior=False))

    dof = nobs - n_params

    print('Chsq min:')
    print(theta)
    print(chsq)
    print(nobs)
    print(chsq/dof)

    # Test
    DMu = np.mean(res)
    print(len(res))
    nres = (res - DMu)/np.std(res)
    ixn = np.where(abs(nres) > 3.)
    print('N3=', vTg[ixn])

    path = dpath+'results/Hnres'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(nres, bins='knuth', histtype='stepfilled',
         alpha=0.2)
    plt.savefig(path)

    ksD, ksP = st.kstest(nres, 'norm')

    print(ksD, ksP)
    print(np.std(res))

    # GDsamples.getGelmanRubin()

    # Print to log file
    f = open(dpath+'results/GDlog'+fend+'.txt', 'w')
    f.write(t)
    f.write('--------------------------------------\n')
    f.write(str(chsq)+'\n')
    f.write(str(chsq/dof)+'\n')
    f.write(str(ksD)+','+str(ksP)+'\n')
    f.write(str(np.std(res))+'\n')
    f.write(str(nobs)+'\n')
    f.write(str(FoM)+'\n')
    f.write(str(start)+'\n')
    f.write(str(datetime.datetime.now()))
    f.close

    # print(getMargeStats(include_bestfit=True))

    # L-Sigma
    # Params
    # dmu = 1
    #
    # if dmu == 1:
    #     Nms = 1000
    #     ah = np.random.choice(result['samples'][:, 2], Nms, replace=False)
    #     aOm = np.random.choice(result['samples'][:, 3], Nms, replace=False)
    #     aw0 = np.random.choice(result['samples'][:, 4], Nms, replace=False)
    #
    #     print(np.mean(aw0), np.std(aw0), len(aw0))
    #
    #     path = dpath+'results/HPP'+fend+'.pdf'
    #     fig = plt.subplots(1)
    #     hist(aw0, bins='knuth', histtype='stepfilled', alpha=0.2)
    #     plt.savefig(path)
    #
    #     ixG = np.where(z>10.0)
    #     ixH = np.where(z<10.0)
    #
    #     Mum = z*0.0
    #     MumErr = z*0.0
    #
    #     Mum[ixG] = z[ixG]
    #     MumErr[ixG] = zerr[ixG]
    #
    #     print(len(ixH[0]))
    #     for i in ixH[0]:
    #         print(i, z[i])
    #         aMum = np.zeros(Nms)
    #         aMumErr = np.zeros(Nms)
    #
    #         for j in range(len(ah)):
    #             cosmo = FlatwCDM(H0=ah[j]*100.0, Om0=aOm[j], w0=aw0[j])
    #             aMum[j] = 5.0*np.log10(cosmo.luminosity_distance(z[i]).value) + 25.0
    #
    #         Mum[i] = np.mean(aMum)
    #         MumErr[i] = np.std(aMum)
    #
    #     vMu = Mum
    #     vMuErr = MumErr
    #
    #     with open(dpath+'indat/distmod.pickle', 'wb') as handle:
    #         print('saving ...')
    #         pickle.dump([vMu, vMuErr], handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # else:
    #     with open(dpath+'indat/distmod.pickle', 'rb') as handle:
    #         vMu, vMuErr = pickle.load(handle)
    #
    # path = dpath+'results/LS'+fend+'.pdf'
    #
    # xp = x
    # xsig = xerr
    # # yp = y + 0.4*(vMu + DMu) + 40.08
    # yp = y + 0.4*(vMu) + 40.08
    # ysig = np.sqrt(yerr**2 + 0.4**2*vMuErr**2)
    #
    # if ((zs == 1) or (zs == 6)):
    #     fy = beta*x + alpha
    # elif zs == 2:
    #     fy = theta[0]*x
    # elif zs == 3:
    #     fy = 5.0*x
    # else:
    #     fy = theta[1]*x + theta[0]
    #
    # fy = lsfit(dpath, cpath, fend, xp, xsig, yp, ysig)
    # lsres = yp - fy
    #
    # frms = np.std(lsres)
    # nob = len(x)
    #
    # xmin = min(xp)- 0.1
    # xmax = max(xp)+ 0.1
    # ymin = min(yp)- 0.2
    # ymax = max(yp)+ 1.1
    #
    # xl = "$\log \sigma (\mathrm{H}\\beta)$ $\mathrm{(km\ s^{-1})}$"
    # yl = "$\log L(\mathrm{H}\\beta)$$\mathrm{(erg\ s^{-1})}$"
    #
    # ixz = np.where((z > 0.2) & (z < 10))
    # ix0 = np.arange(len(x))
    # # ixn = np.where(ix0 > 188)
    #
    # plt.figure(figsize=(8, 8))
    # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    # ax1 = plt.subplot(gs[0])
    # plt.errorbar(xp, yp, xerr=xsig, yerr=ysig, fmt='o', markersize=0.8, lw=0.8)
    # # plt.errorbar(xp[ixn], yp[ixn], xerr=xsig[ixn], yerr=ysig[ixn], fmt='ro', markersize=0.5)
    # # plt.plot(xp[ixn], yp[ixn], 'r*', markersize=0.4)
    # plt.plot(xp, fy, lw=2, c='red')
    #
    # plt.annotate("$rms = $"+str("%5.3f" % frms)
    #              + "$, N = $"+str("%5.0f" % nob), xy=(xmin+0.05, ymax-0.5),
    #              xytext=(xmin+0.05, ymax-0.5), fontsize=14)
    #
    # for i in range(len(xp) - 1):
    #     plt.annotate(vTg[i], xy=(xp[i], yp[i]), fontsize=2)
    #
    # plt.axis([xmin, xmax, ymin, ymax])
    # # plt.xlabel(xl, fontsize=18)
    # plt.ylabel(yl, fontsize=18)
    # plt.setp(ax1.get_xticklabels(), visible=False)
    #
    # ax2 = plt.subplot(gs[1], sharex=ax1)
    # res = yp - fy
    # plt.errorbar(xp, res, yerr=ysig, fmt='o', markersize=0.8, lw=0.8)
    # plt.plot(xp, fy-fy, lw=2, c='red')
    #
    # plt.xlabel(xl, fontsize=18)
    # # plt.ylabel(yl, fontsize=18)
    # plt.savefig(path)
    return


def lnprior1(cube):
    cube[0] = cube[0]*4. + 3.
    return cube


def lnlike1(cube):
    beta = cube[0]
    theta = beta
    xsq = chsq1(theta)[0]
    return -0.5*xsq


def chsq1(theta):
    beta = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    apha = 0.0
    w0 = -1
    Om = 0.3
    cosmo = FlatwCDM(H0=100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case1(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\beta"]
    parametersc = [r"$\beta$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike1, Prior=lnprior1,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior3(cube):
    cube[0] = cube[0]*2. + 32.5
    cube[1] = cube[1]*1. + 4.5
    cube[2] = cube[2]*0.4 + 0.5
    return cube


def lnlike3(cube):
    alpha, beta, h0 = cube[0], cube[1], cube[2]
    theta = alpha, beta, h0
    xsq = chsq3(theta)[0]
    return -0.5*xsq


def chsq3(theta):
    alpha, beta, h0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    w0 = -1
    Om = 0.3
    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case3(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\alpha", r"\beta", r"h_0"]
    parametersc = [r"$\alpha$", r"$\beta$", r"$h_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike3, Prior=lnprior3,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior4(cube):
    cube[0] = cube[0]*2. + 32.5
    cube[1] = cube[1]*1. + 4.5
    cube[2] = cube[2]*0.4 + 0.5
    cube[3] = cube[3]
    return cube


def lnlike4(cube):
    alpha, beta, h0, Om = cube[0], cube[1], cube[2], cube[3]
    theta = alpha, beta, h0, Om
    xsq = chsq4(theta)[0]
    return -0.5*xsq


def chsq4(theta):
    alpha, beta, h0, Om = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    w0 = -1
    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case4(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\alpha", r"\beta", r"h", r"\Omega_m"]
    parametersc = [r"$\alpha$", r"$\beta$", r"h", r"$\Omega_m$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike4, Prior=lnprior4,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior41(cube):
    cube[0] = cube[0]
    return cube


def lnlike41(cube):
    Om = cube[0]
    theta = Om
    xsq = chsq41(theta)[0]
    return -0.5*xsq


def chsq41(theta):
    Om = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    w0 = -1
    cosmo = FlatwCDM(H0=100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.*np.log10((cosmo.luminosity_distance(z[ixH]).value*100.)/c) + 25.
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = Muo
    MuErr = MuoErr

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case41(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\Omega_m"]
    parametersc = [r"$\Omega_m$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike41, Prior=lnprior41,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior42(cube):
    cube[0] = cube[0]*0.5 + 0.5
    cube[1] = cube[1]
    return cube


def lnlike42(cube):
    h0, Om = cube[0], cube[1]
    theta = h0, Om
    xsq = chsq42(theta)[0]
    return -0.5*xsq


def chsq42(theta):
    h0, Om = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    w0 = -1
    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = Muo
    MuErr = MuoErr

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case42(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"h", r"\Omega_m"]
    parametersc = [r"h", r"$\Omega_m$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike42, Prior=lnprior42,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior5(cube):
    cube[0] = cube[0]*2. + 32.5
    cube[1] = cube[1]*1. + 4.5
    cube[2] = cube[2]*0.4 + 0.5
    cube[3] = cube[3]
    cube[4] = cube[4]*2. - 2.
    return cube


def lnlike5(cube):
    alpha, beta, h0, Om, w0 = cube[0], cube[1], cube[2], cube[3], cube[4]
    theta = alpha, beta, h0, Om, w0
    xsq = chsq5(theta)[0]
    return -0.5*xsq


def chsq5(theta):
    alpha, beta, h0, Om, w0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0)
    #---------------------------------------------------------------------------
    ixG = np.where(z>10.0)
    ixH = np.where(z<10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            # + 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case5(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    parameters = [r"\alpha",r"\beta", r"h", r"\Omega_m", r"w_0"]
    parametersc = [r"$\alpha$",r"$\beta$", r"$h$", r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike5, Prior=lnprior5,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior51(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]*2. - 2.
    return cube


def lnlike51(cube):
    Om, w0 = cube[0], cube[1]
    theta = Om, w0
    xsq = chsq51(theta)[0]
    return -0.5*xsq


def chsq51(theta):
    Om, w0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = FlatwCDM(H0=100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.*np.log10((cosmo.luminosity_distance(z[ixH]).value*100.)/c) + 25.
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = Muo
    MuErr = MuoErr

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case51(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\Omega_m", r"w_0"]
    parametersc = [r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike51, Prior=lnprior51,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior52(cube):
    cube[0] = cube[0]*4. + 3.
    cube[1] = cube[1]
    cube[2] = cube[2]*2. - 2.
    return cube


def lnlike52(cube):
    beta, Om, w0 = cube[0], cube[1], cube[2]
    theta = beta, Om, w0
    xsq = chsq52(theta)[0]
    return -0.5*xsq


def chsq52(theta):
    beta, Om, w0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    alpha = 0.0
    h = 100.0
    cosmo = FlatwCDM(H0=h, Om0=Om, w0=w0)
    # ---------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    vZ = 73.2*10**(0.2*(z[ixG] - 25.))/c

    Mum[ixG] = 5.*np.log10((cosmo.luminosity_distance(vZ).value*h)/c) + 25.
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.*np.log10((cosmo.luminosity_distance(z[ixH]).value*h)/c) + 25.
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case52(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\beta", r"\Omega_m", r"w_0"]
    parametersc = [r"$\beta$", r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike52, Prior=lnprior52,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior53(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]*2. - 2.
    return cube


def lnlike53(cube):
    Om, w0 = cube[0], cube[1]
    theta = Om, w0
    xsq = chsq53(theta)[0]
    return -0.5*xsq


def chsq53(theta):
    Om, w0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    alpha = 0.0
    beta = 5.0
    cosmo = FlatwCDM(H0=100.0, Om0=Om, w0=w0)
    # ---------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.*np.log10((cosmo.luminosity_distance(z[ixH]).value*100.)/c) + 25.
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case53(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\Omega_m", r"w_0"]
    parametersc = [r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike53, Prior=lnprior53,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior54(cube):
    cube[0] = cube[0]*0.5 + 0.5
    cube[1] = cube[1]
    cube[2] = cube[2]*2.0 - 2.0
    return cube


def lnlike54(cube):
    h0, Om, w0 = cube[0], cube[1], cube[2]
    theta = h0, Om, w0
    xsq = chsq54(theta)[0]
    return -0.5*xsq


def chsq54(theta):
    h0, Om, w0 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = Muo
    MuErr = MuoErr

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case54(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"h", r"\Omega_m", r"w_0"]
    parametersc = [r"h", r"$\Omega_m$", r"$w_0$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike54, Prior=lnprior54,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIG'
    ploter(dpath, cpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior6(cube):
    cube[0] = cube[0]*10 + 25
    cube[1] = cube[1]*8
    cube[2] = cube[2] + 0.3
    cube[3] = cube[3]
    cube[4] = cube[4]*2 - 2
    cube[5] = cube[5]*3 - 2
    return cube


def lnlike6(cube):
    alpha, beta, h0, Om, w0, w1 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5]
    theta = alpha, beta, h0, Om, w0, w1
    xsq = chsq6(theta)[0]
    return -0.5*xsq


def chsq6(theta):
    alpha, beta, h0, Om, w0, w1 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = Flatw0waCDM(H0=h0*100.0, Om0=Om, w0=w0, wa=w1)
    #---------------------------------------------------------------------------
    ixG = np.where(z>10.0)
    ixH = np.where(z<10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case6(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\alpha",r"\beta", r"h", r"\Omega_m", r"w_0", r"w_a"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike6, Prior=lnprior6,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, fend, result, parameters, tag, ctag)
    return


def lnprior61(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]*2 - 2
    cube[2] = cube[2]*3 - 2
    return cube


def lnlike61(cube):
    Om, w0, w1 = cube[0], cube[1], cube[2]
    theta = Om, w0, w1
    xsq = chsq61(theta)[0]
    return -0.5*xsq


def chsq61(theta):
    Om, w0, w1 = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = Flatw0waCDM(H0=100.0, Om0=Om, w0=w0, wa=w1)
    # -------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.*np.log10((cosmo.luminosity_distance(z[ixH]).value*100.)/c) + 25.
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = Muo
    MuErr = MuoErr

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case61(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\Omega_m", r"w_0", r"w_a"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike61, Prior=lnprior61,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, fend, result, parameters, tag, ctag)
    return


def lnprior7(cube):
    cube[0] = cube[0]*0.5 + 0.5
    cube[1] = cube[1]
    cube[2] = cube[2]
    return cube


def lnlike7(cube):
    h0, Om, Ol = cube[0], cube[1], cube[2]
    theta = h0, Om, Ol
    xsq = chsq7(theta)[0]
    return -0.5*xsq


def chsq7(theta):
    h0, Om, Ol = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = LambdaCDM(H0=h0*100.0, Om0=Om, Ode0=Ol)
    #---------------------------------------------------------------------------
    ixG = np.where(z>10.0)
    ixH = np.where(z<10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case7(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"h", r"\Omega_m", r"\Omega_{\Lambda}"]
    parametersc = [r"$h$", r"$\Omega_m$", r"$\Omega_{\Lambda}$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike7, Prior=lnprior7,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior71(cube):
    cube[0] = cube[0]
    cube[1] = cube[1]
    return cube


def lnlike71(cube):
    Om, Ol = cube[0], cube[1]
    theta = Om, Ol
    xsq = chsq71(theta)[0]
    return -0.5*xsq


def chsq71(theta):
    Om, Ol = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = LambdaCDM(H0=100.0, Om0=Om, Ode0=Ol)
    #---------------------------------------------------------------------------
    ixG = np.where(z > 10.0)
    ixH = np.where(z < 10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.*np.log10((cosmo.luminosity_distance(z[ixH]).value*100.)/c) + 25.
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = Muo
    MuErr = MuoErr

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsqA = np.sum(R**2*W)
    xsqB = np.sum(R*W)
    xsqC = np.sum(W)

    xsq = xsqA - xsqB**2/xsqC
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MuErr)
    else:
        return (xsq, R, Mum, MuErr)


def case71(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\Omega_m", r"\Omega_{\Lambda}"]
    parametersc = [r"$\Omega_m$", r"$\Omega_{\Lambda}$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike71, Prior=lnprior71,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, fend, result, parameters, parametersc, tag, ctag)
    return


def lnprior72(cube):
    cube[0] = cube[0]*2. + 32.5
    cube[1] = cube[1]*1. + 4.5
    cube[2] = cube[2]*0.5 + 0.5
    cube[3] = cube[3]
    cube[4] = cube[4]
    return cube


def lnlike72(cube):
    alpha, beta, h0, Om, Ol = cube[0], cube[1], cube[2], cube[3], cube[4]
    theta = alpha, beta, h0, Om, Ol
    xsq = chsq72(theta)[0]
    return -0.5*xsq


def chsq72(theta):
    alpha, beta, h0, Om, Ol = theta
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    cosmo = LambdaCDM(H0=h0*100.0, Om0=Om, Ode0=Ol)
    #---------------------------------------------------------------------------
    ixG = np.where(z>10.0)
    ixH = np.where(z<10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])

    Mu = 2.5*(beta*x + alpha) - 2.5*y - 100.195
    MuErr = 2.5*np.sqrt((yerr)**2 + beta**2*(xerr)**2
            # - 2.0*beta*Rxy*xerr*yerr
            #+ 0.22**2
    )

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2*W)
    xsq2 = xsq - np.sum(np.log(W))

    if fr == 2:
        return (xsq2, R, Mum, MumErr)
    else:
        return (xsq, R, Mum, MumErr)


def case72(cpath, dpath, fend, vbs, sps, prs, ctag):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata
    parameters = [r"\alpha", r"\beta", r"h", r"\Omega_m", r"\Omega_{\Lambda}"]
    parametersc = [r"$\alpha$", r"$\beta$", r"$h$", r"$\Omega_m$",
            r"$\Omega_{\Lambda}$"]
    n_params = len(parameters)

    prefix = cpath + fend

    result = solve(LogLikelihood=lnlike72, Prior=lnprior72,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=vbs,
                   n_live_points=sps, resume=prs, init_MPI=False,
                   sampling_efficiency=0.8)

    tag = 'HIIGx'
    ploter(dpath, fend, result, parameters, parametersc, tag, ctag)
    return


def hd(cpath, dpath, fend):
    x, y, z, xerr, yerr, zerr, Rxy, Muo, MuoErr, vTg = Mdata

    h0 = 0.7
    Om = 0.3
    w0 = -1.0
    cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=w0)

    ixG = np.where(z>10.0)
    ixH = np.where(z<10.0)

    Mum = z*0.0
    MumErr = z*0.0

    Mum[ixG] = z[ixG]
    MumErr[ixG] = zerr[ixG]

    Mum[ixH] = 5.0*np.log10(cosmo.luminosity_distance(z[ixH]).value) + 25.0
    MumErr[ixH] = (5.0/np.log(10.0))*(zerr[ixH]/z[ixH])


    print(x, Mum)
    lS = x
    Mu = Mum

    rxz = np.corrcoef(lS, Mu)
    print(rxz)
    return


def union2020(ve, dpath, drd, obs):
    Tpath = dpath+'indat/Union2020v16.dat'
    data = Table.read(Tpath, format='ascii', comment='#')

    vix = data['col1']

    vx = data['col2']
    vy = data['col3']
    vz = data['col4']

    vxErr = data['col5']
    vyErr = data['col6']
    vzErr = data['col7']

    vsp = data['col8']

    rXY = np.corrcoef(vx, vy)
    vRxy = vx*0.0 + rXY[0,1]

    ix = np.where((vx - vxErr) <= LSL)
    # ix = np.where((vx + vxErr) <= LSL)
    # ix = np.where((vx - vxErr) <= 1.8)

    return (vix[ix], vx[ix], vy[ix], vz[ix], vxErr[ix], vyErr[ix], vzErr[ix],
            vRxy[ix], vsp[ix])


def gehrz(ve, dpath, drd):
    #H0 = h0*100
    H0 = 73.2
    H0Err = 3.1

    # H0 = 69.1
    # H0Err = 3.0

    # Reading GEH2R Data
    TpathLS = dpath+'indat/tablaregiones2018.dat' #gehr2017n2.csv #gehr2017n2t22.csv'
    # TpathLS = dpath+'indat/gehr2017n2.csv'
    tGHN = Table.read(TpathLS, format='ascii')

    vANX = tGHN['logSsi']
    vANXErr = tGHN['erlogSsi']

    # vANX = tGHN['LogSigma']
    # vANXErr = tGHN['ERLogSigma']

    if drd == 0:
        vANY = np.log10(tGHN['FSC'])
        vANYErr = (1.0/np.log(10))*(tGHN['erFSC']/tGHN['FSC'])
        # vANY = np.log10(tGHN['FsinCG'])
        # vANYErr = (1.0/np.log(10))*(tGHN['ERRFsinCG']/tGHN['FsinCG'])
    elif drd == 1:
        vANY = np.log10(tGHN['FCC'])
        vANYErr = (1.0/np.log(10))*(tGHN['erFCC']/tGHN['FCC'])
    elif drd == 2:
        vANY = np.log10(tGHN['FCG'])
        vANYErr = (1.0/np.log(10))*(tGHN['erFCG']/tGHN['FCG'])

    vANZ = H0*10**(0.2*(tGHN['disM'] - 25.0))/c
    vANZErr = np.sqrt((vANZ/H0)**2*H0Err**2+(0.2*np.log(10)*vANZ)**2*tGHN['erdiM']**2)/c

    vANW = vANX*0.0
    vANWErr = vANXErr*0.0

    rXY = np.corrcoef(vANX, vANY)
    vANRxy = vANX*0.0 + rXY[0,1]

    return vANX, vANXErr, vANY, vANYErr, vANZ, vANZErr, vANW, vANWErr, vANRxy


def gehr(ve, dpath, drd):
    # Reading GEH2R Data
    #TpathLS = dpath+'indat/gehr2017n.csv'
    # TpathLS = dpath+'indat/GEHRv10.dat'
    TpathLS = dpath+'indat/tablaregiones2018.dat'
    tGHN = Table.read(TpathLS, format='ascii')

    vANX = tGHN['logSsi']
    vANXErr = tGHN['erlogSsi']

    if drd == 0:
        vANY = np.log10(tGHN['FSC'])
        vANYErr = (1.0/np.log(10))*(tGHN['erFSC']/tGHN['FSC'])
        # vANY = np.log10(tGHN['FsinCG'])
        # vANYErr = (1.0/np.log(10))*(tGHN['ERRFsinCG']/tGHN['FsinCG'])
    elif drd == 1:
        vANY = np.log10(tGHN['FCC'])
        vANYErr = (1.0/np.log(10))*(tGHN['erFCC']/tGHN['FCC'])
    elif drd == 2:
        vANY = np.log10(tGHN['FCG'])
        vANYErr = (1.0/np.log(10))*(tGHN['erFCG']/tGHN['FCG'])

    vANZ = tGHN['disM']
    vANZErr = tGHN['erdiM']

    vANW = np.log10(tGHN['EW'])
    vANWErr = (1.0/np.log(10))*(tGHN['erEWC']/tGHN['EW'])

    # vANR = tGHN['Ru']
    # vANRErr = tGHN['Ruerr']

    # ixS3 = np.where(vANR < 0.0)
    # vANZ[ixS3] = -99

    rXY = np.corrcoef(vANX, vANY)
    vANRxy = vANX*0.0 + rXY[0,1]

    return vANX, vANXErr, vANY, vANYErr, vANZ, vANZErr, vANW, vANWErr, vANRxy


def lh2g(ve, dpath, drd):
    # TpathH2gEc = dpath+'indat/HIIGxCh14v0817.dat'
    # TpathH2gEc = dpath+'indat/HIIGxv10.dat'
    TpathH2gEc = dpath+'indat/tablagalaxias2018.dat'

    tHII = Table.read(TpathH2gEc, format='ascii')

    vBX = tHII['logSsi']
    vBXErr = tHII['erlogSsi']

    # print np.mean(tHII['AVC'])
    # # print np.std(tHII['AVG'])
    # print np.mean(tHII['eraAVC'])

    # print np.mean((np.log10(tHII['FCG']) - np.log10(tHII['FSC']))/(0.4*3.33249))
    # print np.std((np.log10(tHII['FCG']) - np.log10(tHII['FSC']))/(0.4*3.33249))
    # print np.mean(tHII['erFCG']/tHII['FCG'])
    # print np.mean(tHII['erFSC']/tHII['FSC'])

    if drd == 0:
        vBY = np.log10(tHII['FSC'])
        vBYErr = (1.0/np.log(10)) * (tHII['erFSC']/tHII['FSC'])
    elif drd == 1:
        vBY = np.log10(tHII['FCC'])
        vBYErr = (1.0/np.log(10)) * (tHII['erFCC']/tHII['FCC'])
    elif drd == 2:
        vBY = np.log10(tHII['FCG'])
        vBYErr = (1.0/np.log(10)) * (tHII['erFCG']/tHII['FCG'])

    vBZ = tHII['z']
    vBZErr = tHII['z']*1e-2 #tHII['erZ']
    vBZErr = np.sqrt(vBZErr**2 + 0.0005**2)

    vBW = np.log10(tHII['EWCh14'])
    vBWErr = (1.0/np.log(10)) * (tHII['erEWCh14']/tHII['EWCh14'])

    # vBR = tHII['Ru']
    # vBRErr = tHII['RuErr']

    vBR = vBW*0.0
    vBRErr = vBWErr*0.0

    ixS = np.where(vBX < 0.0)
    vBZ[ixS] = -99

    # ixS2 = np.where((vBX - vBXErr) > 1.8) #1.84, 1.8
    ixS2 = np.where((vBX - vBXErr) >= LSL)
    vBZ[ixS2] = -99

    # ixS3 = np.where(vBR < 0.0)
    # vBZ[ixS3] = -99

    rXY = np.corrcoef(vBX, vBY)
    vBRxy = vBX*0.0 + rXY[0,1]

    return vBX, vBXErr, vBY, vBYErr, vBZ, vBZErr, vBW, vBWErr, vBR, vBRErr, vBRxy


def hzh2g(ve, dpath, drd, obs):
    # Reading HZ Data
    TpathLS = dpath+'indat/hzh2g.txt'
    tHzH2G = Table.read(TpathLS, format='ascii')

    index = tHzH2G['col1']

    vHX = tHzH2G['col3']
    vHXErr = tHzH2G['col4']

    vY = np.log10(tHzH2G['col5'])
    vYErr = (1.0/np.log(10)) * (tHzH2G['col6']/tHzH2G['col5'])

    vEBV = tHzH2G['col7']
    vEBVErr = tHzH2G['col8']

    # ixo = np.where(vEBV == 1000)
    # ixn = np.where(index >= 1000)

    # vEBV[ixn] = np.mean(vEBV[ixo])
    # vEBVErr[ixn] = np.mean(vEBVErr[ixo])

    # print np.mean(vEBV)
    # print np.median(vEBV)
    # print np.std(vEBV)
    # print np.mean(vEBVErr)
    # print np.median(vEBVErr)

    if drd == 0:
        vHY = vY
        vHYErr = vYErr
    elif drd == 1:
        vHY = vY + 0.4*4.598*EBVC
        vHYErr = np.sqrt(vYErr**2 + (0.4*4.598)**2*EBVCErr**2)
    elif drd == 2:
        vHY = vY + 0.4*3.33249*EBVG
        vHYErr = np.sqrt(vYErr**2 + (0.4*3.33249)**2*EBVGErr**2)
    else:
        print('error')
        # vHY = vY + 0.4*3.33249*0.19
        # vHYErr = np.sqrt(vYErr**2 + (0.4*3.33249)**2*0.10**2)
        # vHY = vY + 0.4*3.33249*vEBV
        # Calzetti
        # vHY = vY + 0.4*4.598*vEBV
        # vHYErr = np.sqrt(vYErr**2 + (0.4*4.598)**2*vEBVErr**2)
        # vHY = vY + 0.4*4.598*0.15
        # vHYErr = np.sqrt(vYErr**2 + (0.4*4.598)**2*0.06**2)

        # vHYErr = vHY*0.2

    # print 'EBV'
    # print np.mean(vEBV)
    # print np.mean(vEBVErr)

    vHZ = tHzH2G['col2']
    vHZErr = tHzH2G['col2']*0.01 #OJO

    ixS2 = np.where((vHX - vHXErr) >= LSL) #1.84
    index[ixS2] = -1

    ixS3 = np.where(index >= 1000)
    index[ixS3] = 0

    if obs == 0:
        ixr = np.where(index >= 0)
    elif obs == 1:
        ixr = np.where(index >= 1)
    elif obs == 2:
        ixr = np.where((index >= 1)&(index < 200))
    elif obs == 3:
        ixr = np.where((index >= 0)&(index < 200))
    else:
        ixr = np.where(index >= 200)

    rXY = np.corrcoef(vHX, vHY)
    vHRxy = vHX*0.0 + rXY[0,1]

    return vHX[ixr], vHXErr[ixr], vHY[ixr], vHYErr[ixr], vHZ[ixr], vHZErr[ixr], vHRxy[ixr]


def hzmfh2g(ve, dpath, drd, obs):
    if drd == 0:
        print('error')
    elif drd == 1:
        Tpath = dpath+'indat/Union2018Cv2.txt'
        tHzH2Gc = Table.read(Tpath, format='ascii')

        vID = tHzH2Gc['col1']
        vHX = tHzH2Gc['col2']
        vHY = tHzH2Gc['col3']
        vHZ = tHzH2Gc['col4']
        vHXErr = tHzH2Gc['col5']
        vHYErr = tHzH2Gc['col6']
        vHZErr = tHzH2Gc['col7']
    elif drd == 2:
        Tpath = dpath+'indat/Union2018Gv5.txt'
        tHzH2Gg = Table.read(Tpath, format='ascii')

        vID = tHzH2Gg['col1']
        vHX = tHzH2Gg['col2']
        vHY = tHzH2Gg['col3']
        vHZ = tHzH2Gg['col4']
        vHXErr = tHzH2Gg['col5']
        vHYErr = tHzH2Gg['col6']
        vHZErr = tHzH2Gg['col7']
    else:
        print('error')

    ixi = np.where(vID >= 139)
    vHX = vHX[ixi]
    vHY = vHY[ixi]
    vHZ = vHZ[ixi]
    vHXErr = vHXErr[ixi]
    vHYErr = vHYErr[ixi]
    vHZErr = vHZErr[ixi]

    ixr = np.where((vHX - vHXErr) <= LSL) #test 1.84

    rXY = np.corrcoef(vHX, vHY)
    vHRxy = vHX*0.0 + rXY[0,1]

    return vHX[ixr], vHXErr[ixr], vHY[ixr], vHYErr[ixr], vHZ[ixr], vHZErr[ixr], vHRxy[ixr]


def hzh2g2020(ve, dpath, drd, obs):
    Tpath = dpath+'indat/hzh2g2020v2.dat'
    data = Table.read(Tpath, format='ascii', comment='#')

    vsp = data['flag']

    vix = data['ID']
    vName = data['Target']

    sigO = data['sigma_obs']
    sigOErr = data['err_sigma_obs']

    sigI = data['sigma_RES_INST']
    sigIErr = data['err_sigma_RES_INST']

    sigfs = data['sigma_fs']
    sigth = data['sigma_th']

    ixmf = np.where(vsp == 4)
    sigIErr[ixmf] = 22.6816746*0.03 # MOSFIE inst. borad. uncertainty fixed at 3%

    ixs = np.where(sigIErr > 0.)

    sigC = np.sqrt(sigO[ixs]**2 - sigI[ixs]**2 - sigfs[ixs]**2 - sigth[ixs]**2)
    sigCErr = np.sqrt((sigO[ixs]/sigC)**2*sigOErr[ixs]**2
                      + (sigI[ixs]/sigC)**2*sigIErr[ixs])

    lsig = np.log10(sigC)
    lsigErr = 1./(np.log(10.))*(sigCErr/sigC)

    ixr = np.where((lsig - lsigErr) <= LSL)

    vixf = vix[ixs][ixr]
    vNamef = vName[ixs][ixr]

    KX = lsig[ixr]
    KXErr = lsigErr[ixr]

    vfxo = data['FHx_obs'][ixs][ixr]*1.e-17
    vfxoErr = data['err_FHx_obs'][ixs][ixr]*1.e-17

    vfg = data['note'][ixs][ixr]
    vk = data['Kapha_Hx'][ixs][ixr]
    vAv = data['Av'][ixs][ixr]
    vAvErr = data['err_Av'][ixs][ixr]
    Rv = 2.77

    vfc = vfxo*0.0
    vfcErr = vfxo*0.0
    for i in range(len(vfc)):
        if vfg[i] == 2:
            vfc[i] = vfxo[i]*10.**((0.4*vAv[i]*vk[i])/(Rv))
            vfcErr[i] = vfc[i]*np.sqrt((vfxoErr[i]/vfxo[i])**2
                                       + ((0.4*vk[i]*np.log(10.))/Rv)**2*vAvErr[i]**2)
        else:
            vfc[i] = (vfxo[i]*10.**((0.4*vAv[i]*vk[i])/(Rv)))/2.86
            vfcErr[i] = vfc[i]*np.sqrt((vfxoErr[i]/vfxo[i])**2
                                       + ((0.4*vk[i]*np.log(10.))/Rv)**2*vAvErr[i]**2)

    KY = np.log10(vfc)
    KYErr = (1./np.log(10.))*(vfcErr/vfc)

    KZ = data['z'][ixs][ixr]
    KZErr = data['err_z'][ixs][ixr]

    vspf = data['flag'][ixs][ixr]

    rXY = np.corrcoef(KX, KY)
    vKRxy = KX*0.0 + rXY[0,1]

    # lsigt = data['logsigma_corr'][ixs][ixr]
    # lsigtErr = data['logerr_sigma_corr'][ixs][ixr]
    # lfxc = data['logFHb_corr'][ixs][ixr]
    # lfxcErr = data['logerr_FHb_corr'][ixs][ixr]
    # print(vfc)

    # for i in range(len(KX)-1):
        # print(vixf[i], KZ[i], KZErr[i], KX[i], KXErr[i], vKRxy[i], vspf[i])
    # print(len(KX))

    return KX, KXErr, KY, KYErr, KZ, KZErr, vKRxy, vspf


def mch2gv69(ve, dpath, cpath, clc=0, opt=0, spl=0, zps=0, prs=1, drd=0,
             obs=0, nwk=1000, sps=10000, tds=6, vbs=0, fr0=1,
             a=0, aErr=0, b=0, bErr=0):
    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('mch2gv69: ' + str(datetime.datetime.now()))
    print('+++++++++++++++++++++++++++++++++++++++++++')

    # ============= Parameters ======================================
    global Mdata
    # global lsdata
    global alpha
    global alphaErr
    global beta
    global betaErr
    global fr
    global x0
    global x0Err
    global start
    global zs

    zs = zps

    alpha = a
    alphaErr = aErr
    beta = b
    betaErr = bErr

    print(alpha, beta)

    x0 = x0
    x0Err = x0Err

    fr = fr0
    fg0 = clc
    fg1 = opt
    fsim = 1

    fend = 'mchv69_' + str(ve) + '_' + str(spl) + str(zps) + str(fg1) + \
        str(drd) + str(obs) + str(fr0)
    print(fend)
    print(datetime.datetime.now())
    start = datetime.datetime.now()
    # ============= Main Body =======================================
    if spl == 1:
        # Union2020
        # 0: GEHRs, 1:local HIIG, 2: Literatura, 3: XShooter, 4: MOSFIRE, 5: KMOS
        vTg, vX, vY, vZ, vXErr, vYErr, vZErr, vRxy, vsp = union2020(ve, dpath, drd, obs)
        if (zps == 0):
            ix = np.where(vX > 0)
        elif (zps == 1 or zps == 2 or zps == 3):
            ix = np.where(vsp != 0)
        elif (zps == 6):
            ix = np.where((vsp != 0) & (vsp != 2))
        elif (zps == 11):
            ix = np.where(vsp == 1)
        elif (zps == 22):
            ix = np.where(vsp != 2)
        elif (zps == 23):
            ix = np.where((vsp == 0) | (vsp == 1) | (vsp == 5))
        elif (zps == 24):
            ix = np.where(vsp != 5)
        elif (zps == 25):
            ix = np.where((vsp != 5) & (vsp != 2))
        else:
            print('zps option error')
    elif spl == 2:
        # GEHR Data zero point
        vzpX, vzpXErr, vzpY, vzpYErr, vzpZ, vzpZErr, vzpW, vzpWErr, vzpRxy = \
                gehr(ve, dpath, drd)
        ixzp = vzpX*0 + 0
        # GEHR Data
        vANX, vANXErr, vANY, vANYErr, vANZ, vANZErr, vANW, vANWErr, vANRxy = \
                gehrz(ve, dpath, drd)
        ixAN = vANX*0 + 0
        # Local HIIgx Data
        vBX, vBXErr, vBY, vBYErr, vBZ, vBZErr, vBW, vBWErr, vBR, vBRErr, vBRxy = \
                lh2g(ve, dpath, drd)
        ixB = vBX*0 + 1
        # Reading HZ Datatest
        vHX, vHXErr, vHY, vHYErr, vHZ, vHZErr, vHRxy = hzh2g(ve, dpath, drd, obs)
        ixH = vHX*0 + 2

        # Reading HZ Data Mosfire
        vHMX, vHMXErr, vHMY, vHMYErr, vHMZ, vHMZErr, vHMRxy = \
                hzmfh2g(ve, dpath, drd, obs)
        ixHM = vHMX*0 + 3

        zlim = 0.0
        ixAZ = np.where(vANZ >= zlim)
        ixBZ = np.where(vBZ <= 0.2)
        ixHMZ = np.where(vHMZ <= 3.0)

        # Joining samples
        if zps == 1:
            print(len(vANX[ixAZ]), len(vBX[ixBZ]), len(vHX), len(vHMX))

            vX = np.concatenate((vANX[ixAZ], vBX[ixBZ], vHX, vHMX))
            vXErr = np.concatenate((vANXErr[ixAZ], vBXErr[ixBZ], vHXErr, vHMXErr))

            vY = np.concatenate((vANY[ixAZ], vBY[ixBZ], vHY, vHMY))
            vYErr = np.concatenate((vANYErr[ixAZ], vBYErr[ixBZ], vHYErr, vHMYErr))

            vZ = np.concatenate((vANZ[ixAZ], vBZ[ixBZ], vHZ, vHMZ))
            vZErr = np.concatenate((vANZErr[ixAZ], vBZErr[ixBZ], vHZErr, vHMZErr))

            vRxy = np.concatenate((vANRxy[ixAZ], vBRxy[ixBZ], vHRxy, vHMRxy))

            vsp = np.concatenate((ixAN[ixAZ], ixB[ixBZ], ixH, ixHM))
            vTg = vsp

            ix = np.where(vX > 0)
        elif zps == 2:
            print(len(vBX[ixBZ]), len(vHX), len(vHMX))

            vX = np.concatenate((vBX[ixBZ], vHX, vHMX))
            vXErr = np.concatenate((vBXErr[ixBZ], vHXErr, vHMXErr))

            vY = np.concatenate((vBY[ixBZ], vHY, vHMY))
            vYErr = np.concatenate((vBYErr[ixBZ], vHYErr, vHMYErr))

            vZ = np.concatenate((vBZ[ixBZ], vHZ, vHMZ))
            vZErr = np.concatenate((vBZErr[ixBZ], vHZErr, vHMZErr))

            vRxy = np.concatenate((vBRxy[ixBZ], vHRxy, vHMRxy))

            vsp = np.concatenate((ixAN[ixAZ], ixB[ixBZ], ixH, ixHM))
            vTg = vsp

            ix = np.where(vX > 0)
        elif zps == 5:
            print(len(vzpX), len(vBX[ixBZ]), len(vHX), len(vHMX[ixHMZ]))

            vX = np.concatenate((vzpX, vBX[ixBZ], vHX, vHMX[ixHMZ]))
            vXErr = np.concatenate((vzpXErr, vBXErr[ixBZ], vHXErr, vHMXErr[ixHMZ]))

            vY = np.concatenate((vzpY, vBY[ixBZ], vHY, vHMY[ixHMZ]))
            vYErr = np.concatenate((vzpYErr, vBYErr[ixBZ], vHYErr, vHMYErr[ixHMZ]))

            vZ = np.concatenate((vzpZ, vBZ[ixBZ], vHZ, vHMZ[ixHMZ]))
            vZErr = np.concatenate((vzpZErr, vBZErr[ixBZ], vHZErr, vHMZErr[ixHMZ]))

            vRxy = np.concatenate((vzpRxy, vBRxy[ixBZ], vHRxy, vHMRxy[ixHMZ]))

            vsp = np.concatenate((ixzp[ixAZ], ixB[ixBZ], ixH, ixHM))
            vTg = vsp

            ix = np.where(vX > 0)
    elif spl == 3:
        # GEHR Data zero point
        vzpX, vzpXErr, vzpY, vzpYErr, vzpZ, vzpZErr, vzpW, vzpWErr, vzpRxy = \
                gehr(ve, dpath, drd)
        ixzp = vzpX*0 + 0
        # GEHR Data
        vANX, vANXErr, vANY, vANYErr, vANZ, vANZErr, vANW, vANWErr, vANRxy = \
                gehrz(ve, dpath, drd)
        ixAN = vANX*0 + 0
        # Local HIIgx Data
        vBX, vBXErr, vBY, vBYErr, vBZ, vBZErr, vBW, vBWErr, vBR, vBRErr, vBRxy = \
                lh2g(ve, dpath, drd)
        ixB = vBX*0 + 1
        # Reading HZ Datatest
        vHX, vHXErr, vHY, vHYErr, vHZ, vHZErr, vHRxy = hzh2g(ve, dpath, drd, obs)
        ixH = vHX*0 + 2

        # Reading HZ Data Mosfire
        vHMX, vHMXErr, vHMY, vHMYErr, vHMZ, vHMZErr, vHMRxy = \
                hzmfh2g(ve, dpath, drd, obs)
        ixHM = vHMX*0 + 3

        # Reading KMOS (hzh2g2020v2) data
        vKX, vKXErr, vKY, vKYErr, vKZ, vKZErr, vKRxy, vKTg = \
                hzh2g2020(ve, dpath, drd, obs)

        zlim = 0.0
        ixAZ = np.where(vANZ >= zlim)
        ixBZ = np.where(vBZ <= 0.2)
        ixHMZ = np.where(vHMZ <= 3.0)

        # Joining samples
        if zps == 1:
            print(len(vzpZ[ixAZ]), len(vBX[ixBZ]), len(vHX), len(vKX))

            vX = np.concatenate((vzpX[ixAZ], vBX[ixBZ], vHX, vKX))
            vXErr = np.concatenate((vzpXErr[ixAZ], vBXErr[ixBZ], vHXErr, vKXErr))

            vY = np.concatenate((vzpY[ixAZ], vBY[ixBZ], vHY, vKY))
            vYErr = np.concatenate((vzpYErr[ixAZ], vBYErr[ixBZ], vHYErr, vKYErr))

            vZ = np.concatenate((vzpZ[ixAZ], vBZ[ixBZ], vHZ, vKZ))
            vZErr = np.concatenate((vzpZErr[ixAZ], vBZErr[ixBZ], vHZErr, vKZErr))

            vRxy = np.concatenate((vzpRxy[ixAZ], vBRxy[ixBZ], vHRxy, vKRxy))

            vsp = np.concatenate((ixzp[ixAZ], ixB[ixBZ], ixH, vKTg))
            vTg = vsp

            ix = np.where(vX > 0)
        elif zps == 2:
            print(len(vzpZ[ixAZ]), len(vBX[ixBZ]), len(vKX))

            vX = np.concatenate((vzpX[ixAZ], vBX[ixBZ], vKX))
            vXErr = np.concatenate((vzpXErr[ixAZ], vBXErr[ixBZ], vKXErr))

            vY = np.concatenate((vzpY[ixAZ], vBY[ixBZ], vKY))
            vYErr = np.concatenate((vzpYErr[ixAZ], vBYErr[ixBZ], vKYErr))

            vZ = np.concatenate((vzpZ[ixAZ], vBZ[ixBZ], vKZ))
            vZErr = np.concatenate((vzpZErr[ixAZ], vBZErr[ixBZ], vKZErr))

            vRxy = np.concatenate((vzpRxy[ixAZ], vBRxy[ixBZ], vKRxy))

            vsp = np.concatenate((ixzp[ixAZ], ixB[ixBZ], vKTg))
            vTg = vsp

            ix = np.where(vX > 0)
    else:
        print('spl option error')
        return

    # return

    vTg = vTg[ix]
    vX = vX[ix]
    vY = vY[ix]
    vZ = vZ[ix]
    vXErr = vXErr[ix]
    vYErr = vYErr[ix]
    vZErr = vZErr[ix]
    vRxy = vRxy[ix]
    vsp = vsp[ix]

    Muo = 2.5*(beta*vX + alpha) - 2.5*vY - 100.195
    MuoErr = 2.5*np.sqrt((vYErr)**2 + beta**2*(vXErr)**2
                 + betaErr**2*vX**2
                 + alphaErr**2
                 # - 2.0*beta*vRxy*vXErr*vYErr
                 # + 0.22**2
                 )
    # n = 100000
    # aalpha = np.random.normal(alpha, alphaErr, n)
    # abeta  = np.random.normal(beta, betaErr, n)
    # vMuo = vX*0
    # vMuoErr = vX*0
    # for i in range(0, len(vX)):
        # aX = np.random.normal(vX[i], vXErr[i], n)
        # aY = np.random.normal(vY[i], vYErr[i], n)
        # # S = 0.22/beta
        # S = 0.22/5.0
        # aYN = np.random.normal(0.0, S, n)
        # aXN = np.random.normal(0.0, S, n)
        # aMuo = 2.5*(abeta*(aX+aXN) + aalpha - aY) - 100.195
        # vMuo[i] = np.mean(aMuo)
        # vMuoErr[i] = np.std(aMuo)
        # # print np.mean(aMuo), Muo[i]
        # # print np.std(aMuo), MuoErr[i], np.std(aMuo) - MuoErr[i]

    # ixf = np.where((vXErr <= 0.1) & (vYErr <= 0.2))
    # print(len(vX[ixf]))
    # Mdata = [vX[ixf], vY[ixf], vZ[ixf], vXErr[ixf], vYErr[ixf], vZErr[ixf], vRxy[ixf], Muo[ixf], MuoErr[ixf]]

    Mdata = np.array([vX, vY, vZ, vXErr, vYErr, vZErr, vRxy, Muo, MuoErr, vTg])

    # df = pd.DataFrame(np.transpose(Mdata))
    # print(df)
    # dfs = df.sort_values(by=[2], ascending=True)
    # print(dfs)

    # path = dpath+'results/Union2020Sv1.csv'
    # dfs.to_csv(path, columns = [9, 2, 0, 3, 1, 4], index=False)

    # return 

    for i in range(0, len(vX)):
        print(vTg[i], vZ[i], vZErr[i], vX[i], vXErr[i], vY[i], vYErr[i], vsp[i])

    print(len(vTg))
    # hd(cpath, dpath, fend)
    # return

    path = dpath+'results/HLSig'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(vX, bins='knuth', histtype='stepfilled',
        alpha=0.2)
    plt.xlabel("$log \sigma$", fontsize=18)
    plt.ylabel("$n$", fontsize=18)
    plt.savefig(path)

    path = dpath+'results/HLESig'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(vXErr, bins='knuth', histtype='stepfilled',
        alpha=0.2)
    plt.xlabel("$\epsilon log \sigma$", fontsize=18)
    plt.ylabel("$n$", fontsize=18)
    plt.savefig(path)

    path = dpath+'results/HLfx'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(vY, bins='knuth', histtype='stepfilled',
        alpha=0.2)
    plt.xlabel("$log f$", fontsize=18)
    plt.ylabel("$n$", fontsize=18)
    plt.savefig(path)

    path = dpath+'results/HLEfx'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(vYErr, bins='knuth', histtype='stepfilled',
        alpha=0.2)
    plt.xlabel("$\epsilon log f$", fontsize=18)
    plt.ylabel("$n$", fontsize=18)
    plt.savefig(path)


    ixZ = np.where(vZ <= 5.0)
    path = dpath+'results/HZ'+fend+'.pdf'
    fig = plt.subplots(1)
    hist(vZ[ixZ], bins='knuth', histtype='stepfilled',
        alpha=0.2)
    plt.xlabel("$z$", fontsize=18)
    plt.ylabel("$n(z)$", fontsize=18)
    plt.savefig(path)
    plt.clf()

    # return

    ctag = str(fg1)
    # Cases
    if fg1 == 1:
        case1(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 3:
        case3(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 4:
        case4(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 41:
        case41(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 42:
        case42(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 5:
        case5(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 51:
        case51(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 52:
        case52(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 53:
        case53(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 54:
        case54(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 6:
        case6(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 61:
        case61(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 7:
        case7(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 71:
        case71(cpath, dpath, fend, vbs, sps, prs, ctag)
    elif fg1 == 72:
        case72(cpath, dpath, fend, vbs, sps, prs, ctag)

    print(datetime.datetime.now())
    return

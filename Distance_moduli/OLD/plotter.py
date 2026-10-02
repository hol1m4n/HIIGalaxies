from __future__ import annotations
import os
import json
from astropy.cosmology import FlatwCDM


import os, sys
print(sys.executable)
print(os.environ.get("LD_LIBRARY_PATH","<empty>"))
import pymultinest
print("OK")

import re
import os
import numpy as np
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
import matplotlib
matplotlib.use('Agg') # O plt.switch_backend('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from matplotlib.ticker import AutoMinorLocator
from astropy.constants import L_sun
from matplotlib import gridspec
import astropy.units as u
from math import pi
import gc
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from astropy.table import Column
from scipy.stats import bootstrap



import pymultinest
import corner
from getdist import plots, MCSamples
import scipy.optimize as op
from scipy import stats
from scipy.stats import norm
from astropy.time import Time
import matplotlib.dates as mdates

def weighted_average(modulus,error):
    modulus,error = np.array(modulus),np.array(error)
    err_sq_inv = 1 / ((error)**2)
    num = np.sum(err_sq_inv * modulus)
    dem = np.sum(err_sq_inv)

    return num/dem

def weighted_error(error):
    error = np.array(error)
    err_sq_inv = 1 / ((error)**2)
    dem = np.sum(err_sq_inv)

    return 1 / np.sqrt(dem)

def bootstrap_error2(modulus,error,N=200000,group = '',Galaxy=''):

    PATH = f'Bootstrap_resampling/{group}/{Galaxy}.txt'

    if os.path.exists(PATH) == True:
        bootstrap_statistics = np.loadtxt(PATH)
        print(f'Reading bootstrap for {Galaxy}\n')
    if os.path.exists(PATH) == False:
        print(f'Running bootstrap for {Galaxy}\n')
        modulus,error = np.array(modulus),np.array(error)
        size = len(modulus)

        n_bootstraps = N  # Number of bootstrap iterations
        bootstrap_statistics = []

        for i in range(n_bootstraps):
            # Create a resample with replacement, same size as original data
            bootstrap_sample = np.random.choice(size, size=size, replace=True)

            modulus_sample = modulus[bootstrap_sample]
            error_sample = error[bootstrap_sample]

            sample_mean = weighted_average(modulus_sample,error_sample)
            bootstrap_statistics.append(sample_mean)

        bootstrap_statistics = np.array(bootstrap_statistics)

        np.savetxt(fname = PATH, X = bootstrap_statistics)

    N_dist = len(bootstrap_statistics)

    br_mean = np.mean(bootstrap_statistics)

    quad_br = (bootstrap_statistics-br_mean)**2
    
    lower_bound = np.percentile(bootstrap_statistics, 15.87)
    upper_bound = np.percentile(bootstrap_statistics, 84.13)

    return [np.sqrt(((1)/(N_dist-1)) * np.sum(quad_br)),
           [br_mean - lower_bound, upper_bound - br_mean],
            bootstrap_statistics]

    
def cochran_error(modulus,error):
    Mu_i,Er_i = np.array(modulus),np.array(error)
    n = len(Mu_i)
    w_i = 1 / ((Er_i)**2)

    Mu_w = weighted_average(Mu_i,Er_i)
    w_mean = np.mean(w_i)

    s1 = n / ((n-1) * (np.sum(w_i))**2)
    s2 = (w_i * Mu_i  - (w_mean * Mu_w))**2
    s3 = (w_i - w_mean) * ((w_i*Mu_i) - (w_mean*Mu_w)) 
    s4 = (w_i - w_mean)**2

    return np.sqrt(s1 * (np.sum(s2) - (2 * Mu_w * np.sum(s3)) +    (Mu_w**2 * np.sum(s4)) ) )


def nestedsampling_error(modulus,error,steps=10000, Name='', group = ''):

    def prior_transform(x):
        mu = 20 * x + 20
        return mu
    
    def lnlike(theta, m, merr):
        mu_pi = theta
        R = (mu_pi - m)
        W = 1.0/(merr**2)
        xsq = np.sum(R**2 * W)
        L = -0.5*xsq
        return L
    
    def loglike_for_mnest(theta):
        return lnlike(theta, modulus, error)
    
    outdir = os.path.join('Nested_sampling', group)
    os.makedirs(outdir, exist_ok=True)
    prefix = os.path.join(outdir, Name)

    n_dims = 1
    result = pymultinest.solve(
        LogLikelihood=loglike_for_mnest,
        Prior=prior_transform,
        n_dims=n_dims,
        outputfiles_basename=prefix,
        evidence_tolerance=0.5,
        n_live_points=steps,
        multimodal=True,
        verbose=False,
    )

    samples = result["samples"] 
    T_sample = samples.T


    # calcula loglike para cada muestra
    logL = np.array([loglike_for_mnest(s) for s in samples])

    # punto MAP (aprox): el de mayor logL
    #theta_map = samples[np.argmax(logL)]

    # chi^2 en MAP
    chi2_map = -2.0*np.max(logL)

    N = len(modulus)   # número de datos
    k = n_dims         # parámetros del modelo
    nu = N - k
    chi2_red = chi2_map / nu

    sigma2_corr = T_sample.std() * chi2_red

    return [T_sample.std(),sigma2_corr]




def plotter_statRESULTS(FILE='modulus_tracker_TABLES.fits',LIST='',data_split=''):

    S = fits.open(FILE)

    for z in range(len(LIST)):

        Galaxia = LIST[z]
        GRUPO = data_split

        if GRUPO == 'sm_r' or GRUPO == 'cm_r' or GRUPO == 'sm_t' or GRUPO == 'cm_t':
            metodo = 'CEPHEIDS'
        if GRUPO == 't_r':
            metodo = 'TRGB'

        for hdu in S:
            header = hdu.header
            if ('EXTNAME' in header and header['EXTNAME'] == Galaxia) and \
                ('METHOD' in header and header['METHOD'] == metodo):
                hdu_encontrado = hdu
                T = Table.read(hdu)
                if GRUPO == 'sm_r':
                    T = T[(T['Zcorr']==False) & (T['e_R']>= 0)]
                    DF_t = pd.read_csv('Results_tables/sm_r.csv')
                    DF_t = DF_t[DF_t['Galaxia'] == Galaxia]
                    title_label = r'$PLrC^{ \times Z}_{Random}$'
                if GRUPO == 'cm_r':
                    T = T[(T['Zcorr']==True) & (T['e_R']>= 0)]
                    DF_t = pd.read_csv('Results_tables/cm_r.csv')
                    DF_t = DF_t[DF_t['Galaxia'] == Galaxia]
                    title_label = r'$PLrC^{Z}_{Random}$'
                if GRUPO == 'sm_t':
                    T = T[(T['Zcorr']==False)  &  (T['e_T']>= 0)]
                    DF_t = pd.read_csv('Results_tables/sm_t.csv')
                    DF_t = DF_t[DF_t['Galaxia'] == Galaxia]
                    title_label = r'$PLrC^{ \times Z}_{Total}$'
                if GRUPO == 'cm_t':
                    T = T[(T['Zcorr']==True)  &  (T['e_T']>= 0)]
                    DF_t = pd.read_csv('Results_tables/cm_t.csv')
                    DF_t = DF_t[DF_t['Galaxia'] == Galaxia]
                    title_label = r'$PLrC^{Z}_{Total}$'
                if GRUPO == 't_r':
                    T = T
                    DF_t = pd.read_csv('Results_tables/t_r.csv')
                    DF_t = DF_t[DF_t['Galaxia'] == Galaxia]
                    title_label = r'$TRGB_{Random}$'
                break 


        outdir = os.path.join('Nested_sampling', GRUPO) 
        prefix = os.path.join(outdir, Galaxia)
        samples  = pymultinest.analyse.Analyzer(n_params = 1, outputfiles_basename=prefix)
        mu_pi_samples = samples.get_equal_weighted_posterior()[:,0]
        bootstrap_pahist = bootstrap_error2(modulus = [0],error = [0],N=200000,group=GRUPO,Galaxy=Galaxia)[2]
        binS = np.histogram_bin_edges(bootstrap_pahist, bins='doane')

        
        #width_double_column = 176 / 25.4 
        #height = 5.0 # Puedes ajustar la altura según necesites

        #fig = plt.figure(figsize=(width_double_column, height))


        fig = plt.figure(figsize=(20, 10))

        gs = gridspec.GridSpec(
            2, 2,
            figure=fig,
            width_ratios=[1.0, 1.0],
            height_ratios=[1.0, 0.2],
            wspace=0.1,
            hspace=0.2
        )
        gs.update(left=0.05, bottom=0.05, right=0.98, hspace=0.2)
        ax1 = fig.add_subplot(gs[0, 0])


        #ax1.set_title(f'{Galaxia} - {title_label}', fontsize=30)
        ax1.set_ylabel(r'PDF[%]', fontsize=15)
        ax1.set_xlabel(r'$\mu$', fontsize=15)
        ax1.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.5)
        ax1.minorticks_on()
        ax1.hist(mu_pi_samples, bins=binS, density=True, 
                color='#1f77b4', alpha=0.7, label=r'$\mathcal{P} \,( \mu |D)$',histtype='step',linewidth=3)
        ax1.hist(bootstrap_pahist, bins=binS, density=True, 
                color="#3ab41f", alpha=0.7, label=r'$\mathcal{B} \,( \mu_w)$',histtype='step',linewidth=3)

        mu_w = float(DF_t['mu_w'].values[0])
        mu_L = float(samples.get_stats()['modes'][0]['mean'][0])
        sigma_w = float(DF_t['sigma_w'].values[0])
        sigmabr = float(DF_t['sigma_br'].values[0])
        sigmaC = float(DF_t['sigma_C'].values[0])
        sigmacl_plus = float(DF_t['sigma_cl+'].values[0])
        sigmacl_minus = float(DF_t['sigma_cl-'].values[0])
        sigmaL = float(DF_t['sigma_L'].values[0])
        sigmaLcorr = float(DF_t['sigma_Lcorr'].values[0])


        ax1.axvline(mu_w, color='black', linestyle='dotted', linewidth=3, alpha = 0.7, 
                    label=r'${\mu}_{w}$: ' + f'{mu_w:.3f}')
        ax1.axvline(mu_L, color='red', linestyle='dotted', linewidth=3, alpha = 0.7, 
                    label=r'${\mu}_{\mathcal{L}}$: ' + f'{mu_L:.3f}')


        sigmas_loc = [0.6,0.5,0.4,0.3,0.2,0.1]
        sigmas_color = ["#b41f1f","#1f30b4","#b41faa","#090a08","#994607","#0d9edc"]


        for e in range(6):
            if e == 0:
                sigma_val = sigma_w
                label_sigma = r'$\sigma_{w}$'
            if e == 1:
                sigma_val = sigmabr
                label_sigma = r'$\sigma_{Br}$'
            if e == 2:
                sigma_val = sigmaC
                label_sigma = r'$\sigma_{C}$'
            if e == 3:
                #sigma_val = sigmacl_plus
                label_sigma = r'$\sigma_{cl{\pm}}$'
            if e == 4:
                sigma_val = sigmacl_minus
                label_sigma = r'$\sigma_{\mathcal{L}}$'
            if e == 5:
                sigma_val = sigmaLcorr
                label_sigma = r'$\sigma_{\mathcal{L},corr}$'

            s_loc = sigmas_loc[e]
            s_col = sigmas_color[e]

            line_height = ax1.get_ylim()[1] * sigmas_loc[e]
                
            if e == 3:
                lower_limit = mu_w - sigmacl_minus
                upper_limit = mu_w + sigmacl_plus
                ax1.hlines(y=line_height, 
                    xmin=lower_limit, 
                    xmax=upper_limit,
                    alpha = 0.6,
                    linewidth=5,color = s_col, label = rf"{label_sigma}: $^{{+{sigmacl_plus:.3f}}}_{{-{sigmacl_minus:.3f}}}$")
            else:
                lower_limit = mu_w - sigma_val
                upper_limit = mu_w + sigma_val
                ax1.hlines(y=line_height, 
                    xmin=lower_limit, 
                    xmax=upper_limit,
                    alpha = 0.6,
                    linewidth=5,color = s_col, label=f'{label_sigma}: {sigma_val:.3f}')

            ax1.vlines(x=lower_limit, 
                    ymin=line_height - (ax1.get_ylim()[1]*0.02), 
                    ymax=line_height + (ax1.get_ylim()[1]*0.02), 
                    alpha = 0.6,linestyles='-',
                    linewidth=3,color = s_col)
            ax1.vlines(x=upper_limit, 
                    ymin=line_height - (ax1.get_ylim()[1]*0.02), 
                    ymax=line_height + (ax1.get_ylim()[1]*0.02), alpha = 0.6,
                    linewidth=3,color = s_col)

        ax1.legend(fontsize=15,ncol=2)

        ax2 = fig.add_subplot(gs[0, 1])

        ax2.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.5)
        ax2.minorticks_on()
        ax2.set_ylabel(r'PDF[%]', fontsize=15)
        ax2.set_xlabel(r'$\mu$', fontsize=15)
        ax2.hist(bootstrap_pahist, bins=binS, density=True, 
                color="#3ab41f", alpha=0.7, label=r'$\mathcal{B} \,( \mu_w)$',histtype='step',linewidth=3)

        br_hist_ran = np.linspace(binS.min(), binS.max(), 1000)

        cmap = plt.cm.tab10   # buen colormap categórico
        author_color = {}     # autor -> color
        plotted_authors = set()

        for u in range(len(T)):

            author = T['Citation'][u]

            # Asignar color si es la primera vez que aparece el autor
            if author not in author_color:
                author_color[author] = cmap(len(author_color) % cmap.N)

            color = author_color[author]

            # Elegir sigma según el grupo
            if GRUPO in ('sm_t', 'cm_t'):
                sigma = T['e_T'][u]
            elif GRUPO in ('sm_r', 'cm_r', 't_r'):
                sigma = T['e_R'][u]

            pdf = norm.pdf(br_hist_ran, loc=T['mu_0'][u], scale=sigma)

            # Solo agregar label la primera vez
            label = author if author not in plotted_authors else None

            ax2.plot(
                br_hist_ran,
                pdf,
                lw=2,
                color=color,
                alpha=0.9,
                label=label
            )

            plotted_authors.add(author)

        ax2.legend(fontsize=10)

        ax3 = fig.add_subplot(gs[1, :])

        ax3.set_ylabel(r'$\mu$', fontsize=15)
        ax3.set_xlabel(r'Year', fontsize=15)


        ax3.axhline(mu_w, color='black', linestyle='dotted', linewidth=3, alpha = 0.7, 
                    label=r'${\mu}_{w}$: ' + f'{mu_w:.3f}')


        def jitter_same_time(t, base_buffer=15):
            """
            Si hay repetidos en t, los separa con pequeños offsets en el eje x.
            base_buffer en unidades de t (si t es JD, esto son días).
            """
            t = np.asarray(t, dtype=float)
            t_out = t.copy()

            # buffer automático: ~0.5% de la mediana del espaciado entre fechas únicas
            if base_buffer is None:
                tu = np.unique(t)
                if len(tu) > 1:
                    dt = np.diff(np.sort(tu))
                    base_buffer = 0.005 * np.median(dt)  # ajusta (0.005–0.02) según densidad
                else:
                    base_buffer = 1.0  # 1 día si todo cae en la misma fecha

            # para cada fecha repetida, asigna offsets simétricos: -k,...,0,...,+k
            for val in np.unique(t):
                idx = np.where(t == val)[0]
                if len(idx) > 1:
                    k = len(idx)
                    offsets = (np.arange(k) - (k - 1)/2.0) * base_buffer
                    t_out[idx] = val + offsets

            return t_out

        # --- tus datos ---
        t = np.array(T['JD'])  # JD
        x = np.array(T['mu_0'])  # modulus

        if GRUPO == 'sm_r' or GRUPO == 'cm_r' or GRUPO == 't_r':
            s = np.array(T['e_R'])
        if GRUPO == 'sm_t' or GRUPO == 'cm_t':
            s = np.array(T['e_T'])


        ordr = np.argsort(t)
        t_sorted = np.asarray(t)[ordr]
        x_sorted = np.asarray(x)[ordr]
        s_sorted = np.asarray(s)[ordr]

        t_jit = jitter_same_time(t_sorted)
        t_datetime = Time(t_jit, format='jd').to_datetime()

        cmap = plt.cm.tab10   # buen colormap categórico
        author_color = {}     # autor -> color
        plotted_authors = set()

        for u in range(len(T)):

            author = T['Citation'][u]

            # Asignar color si es la primera vez que aparece el autor
            if author not in author_color:
                author_color[author] = cmap(len(author_color) % cmap.N)

            color = author_color[author]

            # Elegir sigma según el grupo
            if GRUPO in ('sm_t', 'cm_t'):
                sigma = T['e_T'][u]
            elif GRUPO in ('sm_r', 'cm_r', 't_r'):
                sigma = T['e_R'][u]

            pdf = norm.pdf(br_hist_ran, loc=T['mu_0'][u], scale=sigma)

            # Solo agregar label la primera vez
            label = author if author not in plotted_authors else None

            ax3.errorbar(
                t_datetime[u], x_sorted[u], yerr=s_sorted[u],
                fmt='o', capsize=2,color=color
            )

            plotted_authors.add(author)

        ax3.fill_between(
            t_datetime,
            x_sorted - s_sorted,
            x_sorted + s_sorted,
            alpha=0.2
        )

        os.makedirs(f'Distr_plots/{GRUPO}',exist_ok = True)
        fig.suptitle(f'{Galaxia.split('.')[1]} - {title_label} ({len(T)})', fontsize=30)
        fig.savefig(f'Distr_plots/{GRUPO}/{Galaxia}.png', dpi=70, bbox_inches='tight', transparent=False)
        plt.close(fig)

    S.close()


sm_r_lista = os.listdir('Bootstrap_resampling/sm_r')
sm_r_lista = [x.split('.txt')[0] for x in sm_r_lista if x.endswith('.txt')]

plotter_statRESULTS(LIST=sm_r_lista,data_split='sm_r')

cm_r_lista = os.listdir('Bootstrap_resampling/cm_r')
cm_r_lista = [x.split('.txt')[0] for x in cm_r_lista if x.endswith('.txt')]

plotter_statRESULTS(LIST=cm_r_lista,data_split='cm_r')

sm_t_lista = os.listdir('Bootstrap_resampling/sm_t')
sm_t_lista = [x.split('.txt')[0] for x in sm_t_lista if x.endswith('.txt')]

plotter_statRESULTS(LIST=sm_t_lista,data_split='sm_t')

cm_t_lista = os.listdir('Bootstrap_resampling/cm_t')
cm_t_lista = [x.split('.txt')[0] for x in cm_t_lista if x.endswith('.txt')]

plotter_statRESULTS(LIST=cm_t_lista,data_split='cm_t')


t_r_lista = os.listdir('Bootstrap_resampling/t_r')
t_r_lista = [x.split('.txt')[0] for x in t_r_lista if x.endswith('.txt')]

plotter_statRESULTS(LIST=t_r_lista,data_split='t_r')

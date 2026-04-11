### Importing required libraries...

import re
import os
import numpy as np
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
import matplotlib
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
import warnings

### Defining global variables

#global home
#global name
#global hdr_params
#global hdr_titles
#global cat_line

class file_path:
    def __init__(self,home,file): # Input file name & file directory
        self.home = home
        self.file = file
        path = os.path.join(self.home,self.file)
        if os.path.exists(path) != True:
            raise FileNotFoundError('Result spectrum is not in folder or does not exist. Try relocating the file or changing the name')
    def sy_spectra(self):
        pass
    def pop_vector(self):
        pass





def load_mask_ranges(filepath):
    """
    Lee un archivo de máscara con formato:
    N
    wl1  wl2  flag  Name  [comentarios...]
    Devuelve una lista de tuplas (wl1, wl2, flag)
    """
    mask_ranges = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:  # saltar la primera línea (número de regiones)
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        wl1 = float(parts[0])
        wl2 = float(parts[1])
        flag = float(parts[2])
        mask_ranges.append((wl1, wl2, flag))

    return mask_ranges


def spec_pop_STAR_mod(id,
    home = os.path.expanduser("~") + '/gdrive/DataHII/HIIGs/STARLIGHT/',mask_loc = [str('0'),str('0')]):
    name = ''
    name = home + id


    mask_path_folder = mask_loc[0]
    mask_path = mask_path_folder + mask_loc[1]

    if mask_path == '00':
        mask_path = '0'

    if os.path.exists(name) != True:
        raise FileNotFoundError('File is not in folder. Try relocating the file or changing the name')
    
    FITS = fits.open(name)


    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[3, 1.7],
        height_ratios=[2.0, 3.0, 0.5],
        wspace=0.1,
        hspace=0.0
    )
    gs.update(left=0.05, bottom=0.05, right=0.98, hspace=0.0)

    ax1 = fig.add_subplot(gs[0:2, 0])

    TABLE = Table.read(FITS[2])

    l_obs = TABLE['l_obs']
    f_obs = TABLE['f_obs']
    f_syn = TABLE['f_syn']


    ax1.plot(l_obs, f_obs, label='Observed', color='blue', alpha=0.4)
    if (mask_path != '0'):
        MASK = load_mask_ranges(mask_path)
        c2 = 0
        c0 = 0
        for e in MASK:

            chunk = TABLE[(TABLE['l_obs']>=e[0]) & (TABLE['l_obs']<=e[1])]
            

            if e[2] == 2.0:
                if c2 == 0:
                    ax1.plot(chunk['l_obs'], chunk['f_obs'], color='red', alpha=0.4, label= r'$w^{masks}_{\lambda}$ = 2.0')
                else:
                    ax1.plot(chunk['l_obs'], chunk['f_obs'], color='red', alpha=0.4)
                c2 += 1
                
            if e[2] == 0.0:
                if c0 == 0:
                    ax1.plot(chunk['l_obs'], chunk['f_obs'], color='green', alpha=0.4, label= r'$w^{masks}_{\lambda}$ = 0.0')
                else:
                    ax1.plot(chunk['l_obs'], chunk['f_obs'], color='green', alpha=0.4)
                c0 += 1





    
    ax1.plot(l_obs, f_syn, label='Best model', color='black', alpha=1.0, linewidth=2)
    ax1.minorticks_on()
    ax1.tick_params(axis='x',which='major',labelbottom='off')
    ax1.set_ylim(0,max(f_syn)+0.1)
    ax1.set_title(str(FITS[1].header['ARQ_OBS']).replace('.tex',''),fontsize=20)
    ax1.legend(loc='upper right',ncol=2, title = 'STARLIGHT', fontsize=13,
               title_fontproperties = {'weight':'bold', "size":15})
    ax1.set_ylabel(r'$F_\lambda$', fontsize=15)
    ax1.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.5)





    

    
    ax2 = fig.add_subplot(gs[2, 0])
    residual = (f_obs - f_syn)
    ax2.plot(l_obs, residual, color = 'g', alpha = 0.3)
    if (mask_path != '0'):
        MASK = load_mask_ranges(mask_path)
        for e in MASK:
            chunk = TABLE[(TABLE['l_obs']>=e[0]) & (TABLE['l_obs']<=e[1])]
            sub_l_obs = chunk['l_obs']
            sub_f_obs = chunk['f_obs']
            sub_f_syn = chunk['f_syn']
    
            sub_residual = (sub_f_obs - sub_f_syn)
            if e[2] == 2.0:
                ax2.plot(sub_l_obs, sub_residual, color='red', alpha=0.4)
            if e[2] == 0.0:
                ax2.plot(sub_l_obs, sub_residual, color='green', alpha=0.4)




    ax2.minorticks_on()
    minorLocator = AutoMinorLocator(2)
    ax2.yaxis.set_minor_locator(minorLocator)
    ax2.set_ylim(-0.5,0.5)

    
    ax2.set_ylabel(r'${O}_{\lambda} \, - \, {M}_{\lambda}$', fontsize=15)
    ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=15)

    
    ax2.grid(True, which="both", ls="--", color = 'gray', linewidth = 0.5)

        
        
    def smooth_in_logage(edades, x_age_percent, h_dex=0.20, ngrid=800, renorm=True):
        """
        edades: array de log10(age/yr) (centros)
        x_age_percent: array de % por edad (debe sumar ~100)
        h_dex: ancho del kernel en dex (ajústalo para parecerse a FADO)
        """
        edades = np.asarray(edades, float)
        x = np.asarray(x_age_percent, float)

        # malla fina en log-edad
        tau_grid = np.linspace(edades.min()-0.25, edades.max()+0.25, ngrid)

        # Kernel gaussiano en log-edad (densidad en dex^-1)
        dtau = tau_grid[:, None] - edades[None, :]
        K = np.exp(-0.5*(dtau/h_dex)**2) / (np.sqrt(2*np.pi)*h_dex)

        # mezcla ponderada (queda como "densidad" en % por dex)
        x_smooth = K @ x

        if renorm:
            # re-normaliza para que el área en tau coincida (≈100%)
            area = np.trapz(x_smooth, tau_grid)
            if area > 0:
                x_smooth *= (x.sum() / area)

        return tau_grid, x_smooth

    

    POPS_TABLE = Table.read(FITS[1])


    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=gs[:,1],
        hspace=0.12   # <-- espacio solo aquí
    )

    metal_conversion = [
        (POPS_TABLE['Z_j']==0.0001),
        (POPS_TABLE['Z_j']==0.0004),
        (POPS_TABLE['Z_j']==0.004),
        (POPS_TABLE['Z_j']==0.008),
        (POPS_TABLE['Z_j']==0.02),
        (POPS_TABLE['Z_j']==0.05)
    ]

    solar_equival = [0.005,0.02,0.2,0.4,1.0,2.5]

    POPS_TABLE['sun_met'] = np.select(metal_conversion, solar_equival, default=0.0)




    edades = np.log10(np.unique(POPS_TABLE['age_j']))
    AGES_axis = [str(round(e,2)) for e in edades]

    metal_library = np.unique(POPS_TABLE['sun_met'])


    metallicities = {}
    for i in range(len(metal_library)):
        metallicities[f"{str(metal_library[i])}"] = np.zeros(len(edades))

    POPS_TABLE['x_j'] = (POPS_TABLE['x_j'] / (np.sum(POPS_TABLE['x_j']))) * 100

    for i in range(len(edades)):
        age_selection = POPS_TABLE[np.log10(POPS_TABLE['age_j']) == edades[i]]
        for x in metal_library:
            tmp = age_selection[age_selection['sun_met']==x]['x_j'].item()
            metallicities[f"{str(x)}"][i] = tmp
            del tmp
        del age_selection

    x_age_total = np.zeros(len(edades))
    for Zkey, weight_count in metallicities.items():
        x_age_total += weight_count
    x_age_total = 8.5 * x_age_total / x_age_total.sum()
    # === 2) Suavizado tipo kernel en log-edad ===
    # Ajusta h_dex hasta que se parezca al gris de FADO:
    # 0.15 = más “picudo”; 0.25–0.35 = más suave.
    tau_grid, x_smooth = smooth_in_logage(edades, x_age_total, h_dex=0.08, ngrid=1200, renorm=True)



    width = 0.07
    bottom = np.zeros(len(AGES_axis))

    ax3 = fig.add_subplot(gs_right[0])

    ax3.plot(tau_grid, x_smooth, color='black', lw=1.5, alpha=0.5, zorder=5)
    ax3.fill_between(tau_grid, x_smooth, 0, color='gray', alpha=0.15, zorder=4)
    for boolean, weight_count in metallicities.items():
        #color = cmap(norm(float(boolean)))
        ax3.bar(edades, 
                weight_count, 
                width,label=fr"${boolean}\,Z_{{\odot}}$", #width,label=fr"${boolean}\,Z_{{\odot}}$", 
                bottom=bottom,#, color=color
                alpha = 0.6,
                edgecolor = 'k')
        ax3.minorticks_on()
        ax3.tick_params(axis='x',which='major',labelbottom='off')
        bottom += weight_count

    ax3.set_xticks(edades)
    ax3.tick_params(axis='x', colors='red',width=1.5,length=5)
    ax3.set_xticklabels([str(round(e,2)) for e in edades], rotation=90, fontsize=7,color='blue')
    tmp_lnorm = [r'$x_{j}$ [%] $L_{\lambda}$=',r'$\AA$',str(int(FITS[1].header['L_NORM']))]
    ax3.set_ylabel(f'{tmp_lnorm[0]}{tmp_lnorm[2]}{tmp_lnorm[1]}',  fontsize=15)
    ax3.legend(loc='upper right', fontsize=12,ncol = int(len(metal_library)/2))
    ax3.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.2)















    edades = np.log10(np.unique(POPS_TABLE['age_j']))
    AGES_axis = [str(round(e,2)) for e in edades]

    metal_library = np.unique(POPS_TABLE['sun_met'])
    metallicities = {}
    for i in range(len(metal_library)):
        metallicities[f"{str(metal_library[i])}"] = np.zeros(len(edades))

    POPS_TABLE['Mcor_j'] = (POPS_TABLE['Mcor_j'] / (POPS_TABLE['Mcor_j'].sum())) * 100


    for i in range(len(edades)):
        age_selection = POPS_TABLE[np.log10(POPS_TABLE['age_j']) == edades[i]]
        for x in metal_library:
            tmp = age_selection[age_selection['sun_met']==x]['Mcor_j'].item()
            metallicities[f"{str(x)}"][i] = tmp
            del tmp
        del age_selection


    x_age_total = np.zeros(len(edades))
    for Zkey, weight_count in metallicities.items():
        x_age_total += weight_count
    x_age_total = 8.5 * x_age_total / x_age_total.sum()
    # === 2) Suavizado tipo kernel en log-edad ===
    # Ajusta h_dex hasta que se parezca al gris de FADO:
    # 0.15 = más “picudo”; 0.25–0.35 = más suave.
    tau_grid, x_smooth = smooth_in_logage(edades, x_age_total, h_dex=0.08, ngrid=1200, renorm=True)


    width = 0.07
    bottom = np.zeros(len(AGES_axis))

    ax4 = fig.add_subplot(gs_right[1])


    ax4.plot(tau_grid, x_smooth, color='black', lw=1.5, alpha=0.5, zorder=5)
    ax4.fill_between(tau_grid, x_smooth, 0, color='gray', alpha=0.15, zorder=4)
    for boolean, weight_count in metallicities.items():
        #color = cmap(norm(float(boolean)))
        ax4.bar(edades, weight_count, width, label=boolean, bottom=bottom,alpha = 0.6,
                edgecolor = 'k') #,color=color
        ax4.minorticks_on()
        ax4.tick_params(axis='x',which='minor',labelbottom='off')
        bottom += weight_count


    #ax4.set_xticks(edades)
    #ax4.set_xticklabels([str(round(e,2)) for e in edades], rotation=90)

    ax4.set_ylabel(r'$\mu_{j}$ [%]', fontsize=15)
    ax4.set_xlabel(r'log $t_{*}$ [yr]', fontsize=15)
    ax4.tick_params(axis='x', labelrotation=0,which='minor')
    ax4.set_yscale('log')
    ax4.set_ylim([1e-2,1e2])
    ax4.grid(True, which="both", ls="--", color = 'gray', linewidth = 0.2)
    ax4.minorticks_on()



    
    x_j_L_norm = POPS_TABLE['x_j'] / 100 #Norm is for already normalized
    mu_j_M_norm = POPS_TABLE['Mcor_j'] / 100 #Norm is for already normalized


    mean_logt_L = np.sum(x_j_L_norm * np.log10(POPS_TABLE['age_j']))
    mean_logt_M = np.sum(mu_j_M_norm * np.log10(POPS_TABLE['age_j']))
    mean_Z_L = np.sum(x_j_L_norm * POPS_TABLE['sun_met'])
    mean_Z_M = np.sum(mu_j_M_norm * POPS_TABLE['sun_met'])




    titulo_starlight  = r'======>  $\mathbfit{STATS \, BEST \, FIT}$'
    chi2 = r'$\mathbfit{\chi^2 / \nu}$ = %.4f' % FITS[1].header['CHI2NL_']
    adev = r'$\,,\,\mathbfit{Adev}$ = %.4f' % FITS[1].header['ADEV']
    red_law = r'$\mathbfit{Reddening}$ $\mathbfit{law}$ = '+ str(FITS[1].header['RED_LAW'])
    base_src = r'$\mathbfit{BASE}$ = '+ str(FITS[1].header['ARQ_BAS'])
    n_base = r'$\mathbfit{N}$ $\mathbfit{Base}$ = %.0f' % FITS[1].header['N_BASE']
    A_V = r'$\mathbfit{A_{V}}$ = %.4f' % FITS[1].header['AV_MIN']
    v_star = r'$\mathbfit{v_{\star}}$ = %.4f' % FITS[1].header['V0_MIN']
    s_star = r'$\,,\, \mathbfit{\sigma_{\star}}$ = %.4f' % FITS[1].header['VD_MIN']
    mean_age_L = r'$\mathbfit{\langle log\,t \rangle_L}$ = %.4f' % mean_logt_L
    mean_age_M = r'$\,,\, \mathbfit{\langle log\,t \rangle_M}$ = %.4f' % mean_logt_M
    mean_Z_L = r'$\mathbfit{\langle Z \rangle_L}$ = %.4f' % mean_Z_L
    mean_Z_M = r'$\,,\, \mathbfit{\langle Z \rangle_M}$ = %.4f' % mean_Z_M

    ltext = [titulo_starlight,
             chi2+adev, 
             red_law,
             base_src,
             n_base,
             A_V,
             v_star+s_star,
             mean_age_L+mean_age_M,
             mean_Z_L+mean_Z_M
             ]
    text = '\n'.join(ltext)

    ax1.annotate(text, xy=(0.65, 0.85), xytext=(15, -15), fontsize=12,
                 xycoords='axes fraction', textcoords='offset points',
                 bbox=dict(facecolor='0.95', alpha=0.9),
                 horizontalalignment='left', verticalalignment='top')
    




    

    FITS.close()
    #return fig


def spec_pop_FADO_mod(id,
    home = os.path.expanduser("~") + '/gdrive/DataHII/HIIGs/FADO/'):
    name = ''
    name = home + id

    mask_path = '0'

    if os.path.exists(name) != True:
        raise FileNotFoundError('File is not in folder. Try relocating the file or changing the name')
    
    _1D = name
    _DE = name.replace('_1D','_DE')
    _EL = name.replace('_1D','_EL')
    _ST = name.replace('_1D','_ST')

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[3, 1.7],
        height_ratios=[2.0, 3.0, 0.5],
        wspace=0.1,
        hspace=0.0
    )
    gs.update(left=0.05, bottom=0.05, right=0.98, hspace=0.0)

    ax1 = fig.add_subplot(gs[0:2, 0])

    spec_1D = fits.open(_1D)
    spec_hdu = spec_1D[0]

    spec_header = spec_hdu.header
    spec_data = spec_hdu.data

    #Defining X range, that is, where the real spectra is located

    no_false = spec_data[0]!=0.0

    Lambda = np.linspace(spec_header['OLSYNINI'],
                         spec_header['OLSYNFIN'],
                         len(spec_data[0][no_false]))

    obs_flux = spec_data[0][no_false]
    syn_flux = spec_data[3][no_false]
    fado_mask = spec_data[2][no_false]
    fado_mask = (fado_mask == 5)

    ax1.plot(Lambda, obs_flux, label='Observed', color='blue', alpha=0.4)

    aux = np.invert(fado_mask)

    Lambda[aux] = np.nan
    obs_flux[aux] = np.nan

    ax1.plot(Lambda, 
             obs_flux, 
             color='green', alpha=0.4, label= r'FADO masks')
    
    Lambda = np.linspace(spec_header['OLSYNINI'],
                        spec_header['OLSYNFIN'],
                        len(spec_data[0][no_false]))
    
    obs_flux = spec_data[0][no_false]
    syn_flux = spec_data[3][no_false]
    
    ax1.plot(Lambda, syn_flux, label='Best model', color='black', alpha=1.0, linewidth=2)

    stellar_flux = spec_data[7][no_false]
    nebular_flux = spec_data[8][no_false]

    Adev = abs(obs_flux-syn_flux) / obs_flux
    Adev = (np.sum(Adev) / len(obs_flux)) * 100

    ax1.plot(Lambda, stellar_flux, label='Stellar', color='red', alpha=0.7, linewidth=1.0,linestyle = "-")

    ax1.plot(Lambda, nebular_flux, color='k', lw=2, path_effects=[pe.Stroke(linewidth=5, foreground='cyan'), pe.Normal()],label = 'Nebular')

    tmp = id[0:25]
    tmp = '[' + (tmp.replace('.','-')).replace('_',']')

    ax1.minorticks_on()
    ax1.set_title(tmp,fontsize=20)
    ax1.legend(loc='upper right',ncol=2, title = 'FADO', fontsize=13,
        title_fontproperties = {'weight':'bold', "size":15})


    ax1.set_ylabel(r'$F_\lambda$', fontsize=15)
    ax1.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.5)
    ax1.tick_params(axis='x',which='major',labelbottom='off')
    ax1.set_ylim(0,max(syn_flux)+0.1)

    ax2 = fig.add_subplot(gs[2, 0])
    residual = (obs_flux - syn_flux)
    ax2.plot(Lambda, residual, color = 'g', alpha = 0.3)

    aux = np.invert(fado_mask)

    Lambda[aux] = np.nan
    residual[aux] = np.nan

    ax2.plot(Lambda, residual, color='green', alpha=0.4)

    ax2.minorticks_on()
    minorLocator = AutoMinorLocator(2)
    ax2.yaxis.set_minor_locator(minorLocator)
    ax2.set_ylim(-0.50,0.5)
    ax2.set_ylabel(r'${O}_{\lambda} \, - \, {M}_{\lambda}$', fontsize=15)
    ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=15)
    ax2.grid(True, which="both", ls="--", color = 'gray', linewidth = 0.5)



    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=gs[:,1],
        hspace=0.12   # <-- espacio solo aquí
    )



    PV_1D = fits.open(_DE)
    PV_hdu = PV_1D[0]

    PV_header = PV_hdu.header
    PV_data = PV_hdu.data

    N_base = int(PV_header['NUM_BASE'])

    light_frac = PV_data[0][0:N_base] * 100
    mass_frac = PV_data[4][0:N_base] / 100
    age = PV_data[37][0:N_base]
    log_age = PV_data[38][0:N_base]
    Zs_metal = PV_data[39][0:N_base] #Metallicities



    def smooth_in_logage(edades, x_age_percent, h_dex=0.20, ngrid=800, renorm=True):
        """
        edades: array de log10(age/yr) (centros)
        x_age_percent: array de % por edad (debe sumar ~100)
        h_dex: ancho del kernel en dex (ajústalo para parecerse a FADO)
        """
        edades = np.asarray(edades, float)
        x = np.asarray(x_age_percent, float)

        # malla fina en log-edad
        tau_grid = np.linspace(edades.min()-0.25, edades.max()+0.25, ngrid)

        # Kernel gaussiano en log-edad (densidad en dex^-1)
        dtau = tau_grid[:, None] - edades[None, :]
        K = np.exp(-0.5*(dtau/h_dex)**2) / (np.sqrt(2*np.pi)*h_dex)

        # mezcla ponderada (queda como "densidad" en % por dex)
        x_smooth = K @ x

        if renorm:
            # re-normaliza para que el área en tau coincida (≈100%)
            area = np.trapz(x_smooth, tau_grid)
            if area > 0:
                x_smooth *= (x.sum() / area)

        return tau_grid, x_smooth






    POPS_TABLE = Table([light_frac,mass_frac,age,log_age,Zs_metal],
                    names = ('x_j','Mcor_j','age_j','logage_j','Z_j'))
    
    metal_conversion = [
        (POPS_TABLE['Z_j']==0.0001),
        (POPS_TABLE['Z_j']==0.0004),
        (POPS_TABLE['Z_j']==0.004),
        (POPS_TABLE['Z_j']==0.008),
        (POPS_TABLE['Z_j']==0.02),
        (POPS_TABLE['Z_j']==0.05)
    ]

    solar_equival = [0.005,0.02,0.2,0.4,1.0,2.5]

    POPS_TABLE['sun_met'] = np.select(metal_conversion, solar_equival, default=0.0)
    
    edades = np.log10(np.unique(POPS_TABLE['age_j']))
    AGES_axis = [str(round(e,2)) for e in edades]

    metal_library = np.unique(POPS_TABLE['sun_met'])
    metallicities = {}
    for i in range(len(metal_library)):
        metallicities[f"{str(metal_library[i])}"] = np.zeros(len(edades))

    POPS_TABLE['x_j'] = (POPS_TABLE['x_j'] / (POPS_TABLE['x_j'].sum())) * 100

    for i in range(len(edades)):
        age_selection = POPS_TABLE[np.log10(POPS_TABLE['age_j']) == edades[i]]
        for x in metal_library:
            tmp = age_selection[age_selection['sun_met']==x]['x_j'].item()
            metallicities[f"{str(x)}"][i] = tmp
            del tmp
        del age_selection


    x_age_total = np.zeros(len(edades))
    for Zkey, weight_count in metallicities.items():
        x_age_total += weight_count
    x_age_total = 8.5 * x_age_total / x_age_total.sum()
    # === 2) Suavizado tipo kernel en log-edad ===
    # Ajusta h_dex hasta que se parezca al gris de FADO:
    # 0.15 = más “picudo”; 0.25–0.35 = más suave.
    tau_grid, x_smooth = smooth_in_logage(edades, x_age_total, h_dex=0.08, ngrid=1200, renorm=True)



    width = 0.07
    bottom = np.zeros(len(AGES_axis))

    ax3 = fig.add_subplot(gs_right[0])

    ax3.plot(tau_grid, x_smooth, color='black', lw=1.5, alpha=0.5, zorder=5)
    ax3.fill_between(tau_grid, x_smooth, 0, color='gray', alpha=0.15, zorder=4)
    for boolean, weight_count in metallicities.items():
        #color = cmap(norm(float(boolean)))
        ax3.bar(edades, 
                weight_count, 
                width,label=fr"${boolean}\,Z_{{\odot}}$", #width,label=fr"${boolean}\,Z_{{\odot}}$", 
                bottom=bottom,#, color=color
                alpha = 0.6,
                edgecolor = 'k')
        ax3.minorticks_on()
        ax3.tick_params(axis='x',which='major',labelbottom='off')
        bottom += weight_count

    ax3.set_xticks(edades)
    ax3.tick_params(axis='x', colors='red',width=1.5,length=5)
    ax3.set_xticklabels([str(round(e,2)) for e in edades], rotation=90, fontsize=7,color='blue')
    tmp_lnorm = [r'$x_{j}$ [%] $L_{\lambda}$=',r'$\AA$',str(int(spec_header['LAMBDA_0']))]
    ax3.set_ylabel(f'{tmp_lnorm[0]}{tmp_lnorm[2]}{tmp_lnorm[1]}',  fontsize=15)
    ax3.legend(loc='upper right', fontsize=12,ncol = int(len(metal_library)/2))
    ax3.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.2)







    edades = np.log10(np.unique(POPS_TABLE['age_j']))
    AGES_axis = [str(round(e,2)) for e in edades]

    metal_library = np.unique(POPS_TABLE['sun_met'])
    metallicities = {}
    for i in range(len(metal_library)):
        metallicities[f"{str(metal_library[i])}"] = np.zeros(len(edades))

    POPS_TABLE['Mcor_j'] = (POPS_TABLE['Mcor_j'] / (POPS_TABLE['Mcor_j'].sum())) * 100


    for i in range(len(edades)):
        age_selection = POPS_TABLE[np.log10(POPS_TABLE['age_j']) == edades[i]]
        for x in metal_library:
            tmp = age_selection[age_selection['sun_met']==x]['Mcor_j'].item()
            metallicities[f"{str(x)}"][i] = tmp
            del tmp
        del age_selection


    x_age_total = np.zeros(len(edades))
    for Zkey, weight_count in metallicities.items():
        x_age_total += weight_count
    x_age_total = 8.5 * x_age_total / x_age_total.sum()
    # === 2) Suavizado tipo kernel en log-edad ===
    # Ajusta h_dex hasta que se parezca al gris de FADO:
    # 0.15 = más “picudo”; 0.25–0.35 = más suave.
    tau_grid, x_smooth = smooth_in_logage(edades, x_age_total, h_dex=0.08, ngrid=1200, renorm=True)


    width = 0.07
    bottom = np.zeros(len(AGES_axis))

    ax4 = fig.add_subplot(gs_right[1])


    ax4.plot(tau_grid, x_smooth, color='black', lw=1.5, alpha=0.5, zorder=5)
    ax4.fill_between(tau_grid, x_smooth, 0, color='gray', alpha=0.15, zorder=4)
    for boolean, weight_count in metallicities.items():
        #color = cmap(norm(float(boolean)))
        ax4.bar(edades, weight_count, width, label=boolean, bottom=bottom,alpha = 0.6,
                edgecolor = 'k') #,color=color
        ax4.minorticks_on()
        ax4.tick_params(axis='x',which='minor',labelbottom='off')
        bottom += weight_count


    #ax4.set_xticks(edades)
    #ax4.set_xticklabels([str(round(e,2)) for e in edades], rotation=90)

    ax4.set_ylabel(r'$\mu_{j}$ [%]', fontsize=15)
    ax4.set_xlabel(r'log $t_{*}$ [yr]', fontsize=15)
    ax4.tick_params(axis='x', labelrotation=0,which='minor')
    ax4.set_yscale('log')
    ax4.set_ylim([1e-2,1e2])
    ax4.grid(True, which="both", ls="--", color = 'gray', linewidth = 0.2)
    ax4.minorticks_on()

    x_j_L_norm = POPS_TABLE['x_j'] / 100 #Norm is for already normalized
    mu_j_M_norm = POPS_TABLE['Mcor_j'] / 100 #Norm is for already normalized

    mean_logt_L = np.sum(x_j_L_norm * np.log10(POPS_TABLE['age_j']))
    mean_logt_M = np.sum(mu_j_M_norm * np.log10(POPS_TABLE['age_j']))
    mean_Z_L = np.sum(x_j_L_norm * POPS_TABLE['sun_met'])
    mean_Z_M = np.sum(mu_j_M_norm * POPS_TABLE['sun_met'])

    titulo_fado = r'======>  $\mathbfit{STATS \, BEST \, FIT}$'
    chi2 = r'$\mathbfit{\chi^2 / \nu}$ = %.4f' % spec_header['CHI2_RED']
    adev = r'$\,,\,\mathbfit{Adev}$ = %.4f' % Adev
    red_law = r'$\mathbfit{Reddening}$ $\mathbfit{law}$ = '+ spec_header['R_LAWOPT'][0:21]
    base_src = r'$\mathbfit{BASE}$ = '+ spec_header['ARQ_BASE']
    n_base = r'$\mathbfit{N}$ $\mathbfit{Base}$ = %.0f'% spec_header['NUM_BASE']
    A_V = r'$\mathbfit{A_{V\,\star}}$ = %.4f' % PV_header['GEXTINCT'] + r'$\,,\, \mathbfit{A_{V\,Neb}}$ = %.4f' % PV_header['GNEBULAR']
    v_star = r'$\mathbfit{v_{\star}}$ = %.4f' % PV_header['V0SYSGAL']
    s_star = r'$\,,\, \mathbfit{\sigma_{\star}}$ = %.4f' % PV_header['VDSYSGAL']
    mean_age_L = r'$\mathbfit{\langle log\,t \rangle_L}$ = %.4f' % mean_logt_L
    mean_age_M = r'$\,,\,\mathbfit{\langle log\,t \rangle_M}$ = %.4f' % mean_logt_M
    mean_Z_L = r'$\mathbfit{\langle Z \rangle_L}$ = %.4f' % mean_Z_L
    mean_Z_M = r'$\,,\,\mathbfit{\langle Z \rangle_M}$ = %.4f' % mean_Z_M

    ltext = [titulo_fado,
             chi2+adev,
             red_law,
             base_src,
             n_base,
             A_V,
             v_star+s_star,
             mean_age_L+mean_age_M,
             mean_Z_L+mean_Z_M
             ]
    text = '\n'.join(ltext)

    ax1.annotate(text, xy=(0.650, 0.84), xytext=(15, -15), fontsize=12,
                 xycoords='axes fraction', textcoords='offset points',
                 bbox=dict(facecolor='0.95', alpha=0.9),
                 horizontalalignment='left', verticalalignment='top')

    spec_1D.close()
    PV_1D.close()
    #return fig



def FADO_results_reader(id,
    home = os.path.expanduser("~") + '/gdrive/DataHII/HIIGs/FADO/'):
    name = ''
    name = home + id
    if os.path.exists(name) != True:
        raise FileNotFoundError('File is not in folder. Try relocating the file or changing the name')
    
    _1D = name
    spec_1D = fits.open(_1D)
    spec_hdu = spec_1D[0]
    spec_header = spec_hdu.header
    spec_data = spec_hdu.data
    no_false = spec_data[0]!=0.0
    Lambda = np.linspace(spec_header['OLSYNINI'],
                         spec_header['OLSYNFIN'],
                         len(spec_data[0][no_false]))
    obs_flux = spec_data[0][no_false]
    syn_flux = spec_data[3][no_false]
    fado_mask = spec_data[2][no_false]
    fado_mask = (fado_mask == 5)
    aux = np.invert(fado_mask)
    Lambda[aux] = np.nan
    obs_flux[aux] = np.nan
    Lambda = np.linspace(spec_header['OLSYNINI'],
                        spec_header['OLSYNFIN'],
                        len(spec_data[0][no_false]))
    obs_flux = spec_data[0][no_false]
    syn_flux = spec_data[3][no_false]
    Adev = abs(obs_flux-syn_flux) / obs_flux
    Adev = (np.sum(Adev) / len(obs_flux)) * 100
    spec_1D.close()

    _DE = name.replace('_1D','_DE')
    PV_1D = fits.open(_DE)
    PV_hdu = PV_1D[0]
    PV_header = PV_hdu.header
    PV_data = PV_hdu.data
    N_base = int(PV_header['NUM_BASE'])
    light_frac = PV_data[0][0:N_base] * 100
    mass_frac = PV_data[4][0:N_base] / 100
    age = PV_data[37][0:N_base]
    log_age = PV_data[38][0:N_base]
    Zs_metal = PV_data[39][0:N_base] #Metallicities


    POPS_TABLE = Table([light_frac,mass_frac,age,log_age,Zs_metal],
                names = ('x_j','Mcor_j','age_j','logage_j','Z_j'))
    
    metal_conversion = [
        (POPS_TABLE['Z_j']==0.0001),
        (POPS_TABLE['Z_j']==0.0004),
        (POPS_TABLE['Z_j']==0.004),
        (POPS_TABLE['Z_j']==0.008),
        (POPS_TABLE['Z_j']==0.02),
        (POPS_TABLE['Z_j']==0.05)
    ]

    solar_equival = [0.005,0.02,0.2,0.4,1.0,2.5]

    POPS_TABLE['sun_met'] = np.select(metal_conversion, solar_equival, default=0.0)

    POPS_TABLE['x_j'] = (POPS_TABLE['x_j'] / (POPS_TABLE['x_j'].sum())) * 100
    POPS_TABLE['Mcor_j'] = (POPS_TABLE['Mcor_j'] / (POPS_TABLE['Mcor_j'].sum())) * 100

    x_j_L_norm = POPS_TABLE['x_j'] / 100 #Norm is for already normalized
    mu_j_M_norm = POPS_TABLE['Mcor_j'] / 100 #Norm is for already normalized

    mean_logt_L = np.sum(x_j_L_norm * np.log10(POPS_TABLE['age_j']))
    mean_logt_M = np.sum(mu_j_M_norm * np.log10(POPS_TABLE['age_j']))
    mean_Z_L = np.sum(x_j_L_norm * POPS_TABLE['sun_met'])
    mean_Z_M = np.sum(mu_j_M_norm * POPS_TABLE['sun_met'])

    PV_1D.close()

    return [spec_header['CHI2_RED'],Adev,PV_header['GEXTINCT'],PV_header['GNEBULAR'],PV_header['V0SYSGAL'],PV_header['VDSYSGAL'],mean_logt_L,mean_logt_M,mean_Z_L,mean_Z_M]







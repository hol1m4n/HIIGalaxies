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

global home
global name
global hdr_params
global hdr_titles
global cat_line

### Names of the varibles given by STARLIGHT output

hdr_params = ["[arq_obs]","[arq_base]","[arq_masks]","[arq_config]","[N_base]","[N_YAV_components = # of components with extra extinction!]",
              "[i_FitPowerLaw (1/0 = Yes/No)]","[alpha_PowerLaw]","[red_law_option]","[q_norm = A(l_norm)/A(V)]", ## Some input info
              
              "[l_ini (A)]","[l_fin (A)]","[dl    (A)]     (OBS: dl_cushion =    50.00 A)", ## (Re)Sampling Parameters

              "[l_norm (A) - for base]","[llow_norm (A) - window for f_obs]","[lupp_norm (A) - window for f_obs]","[fobs_norm (in input units)]", ## Normalization info
              
              "[llow_SN (A) - window for S/N]","[lupp_SN (A) - window for S/N]","[S/N in S/N window]","[S/N in norm. window]","[S/N_err in S/N window]",
              "[S/N_err in norm. window]","[fscale_chi2]", ## S/N

              "[idum_orig]","[NOl_eff]","[Nl_eff]","[Ntot_cliped & clip_method]","[Nglobal_steps]","[N_chains]","[NEX0s_base = N_base in EX0s-fits]",
              "[Clip-Bug, RC-Crash & Burn-In warning-flags, n_censored_weights, wei_nsig_threshold & wei_limit]",
              "[idt_all, wdt_TotTime, wdt_UsrTime & wdt_SysTime (sec)]", ## etc...

              "[chi2/Nl_eff]","[adev (%)]","[sum-of-x (%)]","[Flux_tot (units of input spectrum!)]","[Mini_tot (???)]","[Mcor_tot (???)]",
              "[v0_min  (km/s)]","[vd_min  (km/s)]","[AV_min  (mag)]","[YAV_min (mag)]", ## Synthesis Results - Best model ##
              ]

hdr_titles = ['## OUTPUT of StarlightChains_v05.for    [Cid@UFSC - 10/May/2008] ##',
              "## Some input info", 
              "## (Re)Sampling Parameters", 
              "## Normalization info", 
              "## S/N",
              "## etc...",
              "## Synthesis Results - Best model ##"]

cat_line = '#' *67


### Fuctions defined for string management and reading data ...


def get_first_value(line: str):
    """
    Devuelve el primer valor no vacío de un string.
    - Si solo hay espacios: devuelve 'No data'
    - Si el valor es número entero/float (incluyendo notación científica): devuelve número
    - Si no se puede convertir: devuelve string
    """
    tokens = line.strip().split()
    if not tokens:   # Caso: línea solo con espacios
        return "NoData"
    
    token = tokens[0]
    # Intentar convertir a número
    try:
        val = int(token)
        return val
    except ValueError:
        try:
            val = float(token)
            return val
        except ValueError:
            return token
        
def parse_before_brackets(line: str):
    """
    Devuelve los valores antes de los corchetes.
    Ejemplo: "895    NSIGMA   [..]" -> ["895", "NSIGMA"]
    """
    # separar la parte antes de los corchetes
    before = re.split(r'\[', line.strip(), maxsplit=1)[0]
    # dividir en tokens por espacios
    tokens = before.split()
    return tokens


### Header for the First Table creation ...


def header_firstTABLE(name):
    HDR = fits.Header()
    with open(name, 'r') as file:
        c = 0
        for line in file:
            v = get_first_value(line) 
            if v ==cat_line:
                #HDR.add_comment(cat_line)
                HDR.insert(c, ('COMMENT',cat_line))
            if v == "##":
                for i in hdr_titles:
                    if (i in line):
                        HDR.insert(c, ('COMMENT',i))
                        #HDR.add_comment(i)
            if v == "NoData":
                #HDR.add_comment('  ')
                HDR.insert(c, ('COMMENT',''))
  
            if (v != "##") and (v != "NoData") and (v != cat_line):
                for i in hdr_params:
                    if (i in line):
                        hdr_name = i
                        hdr_name = hdr_name.replace('=','')
                        hdr_name = hdr_name.replace('[','')
                        hdr_name = hdr_name.replace(']','')
                        hdr_name = hdr_name.replace('/','')
                        sep = hdr_name.partition(' ')
                    
                        if (c==40) or (c==45) or (c==51):
                            if c==40:
                                tmp = parse_before_brackets(line)
                                HDR.insert(c,('Ntot_cl', float(tmp[0]),'Ntot_cliped'))
                                c += 1
                                HDR.insert(c,('Clip_me', tmp[1],'Clip_method'))
                            if c==45:
                                tmp = parse_before_brackets(line)
                                HDR.insert(c,('Clip-Bu', float(tmp[0]),'Clip-Bug'))
                                c += 1
                                HDR.insert(c,('RC-Cras', float(tmp[1]),'RC-Crash'))
                                c += 1
                                HDR.insert(c,('Burn-In', float(tmp[2]),'Burn-In warning-flags'))
                                c += 1
                                HDR.insert(c,('n_censo', float(tmp[3]),'n_censored_weights'))
                                c += 1
                                HDR.insert(c,('wei_nsi', float(tmp[4]),'wei_nsig_threshold'))
                                c += 1
                                HDR.insert(c,('wei_lim', tmp[5],'wei_limit'))
                            if c==51:
                                tmp = parse_before_brackets(line)
                                HDR.insert(c,('idt_all', float(tmp[0]),'idt_all'))
                                c += 1
                                HDR.insert(c,('wdt_Tot', float(tmp[1]),'wdt_TotTime'))
                                c += 1
                                HDR.insert(c,('wdt_Usr', float(tmp[2]),'wdt_UsrTime'))
                                c += 1
                                HDR.insert(c,('wdt_Sys', float(tmp[3]),'wdt_SysTime (sec)'))

                            del tmp
                    


                        else:
                            HDR.insert(c,(sep[0][0:7], v,hdr_name))
                    
                        del hdr_name,sep
     
            c += 1
            del v
            if c>=70:
                break
        file.close()
    return HDR


### Reading of STARLIGHT Spectral Synthesis Fit from .out file ...

def read_starlight_ssp_txt(path):
    # 1) Cargar archivo y ubicar la cabecera "# j ..."
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    hdr_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('#') and ('j')and 'x_j(%)' in line and 'age_j(yr)' in line:
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError("No se encontró la línea de cabecera que inicia con '# j'.")

    # 2) Extraer nombres de columna y sanearlos (sin espacios/?,()%)
    raw_header = lines[hdr_idx].lstrip('# ').strip()
    colnames = re.split(r'\s{2,}|\t+', raw_header)  # separa por 2+ espacios o tabs
    colnames = [re.sub(r'\W+', '_', c).strip('_') for c in colnames]  # saneo

    # 3) Recolectar solo las filas de datos (empiezan con índice numérico)
    data_lines = []
    for j in range(hdr_idx + 1, len(lines)):
        row = lines[j].rstrip('\n')
        if not row.strip():
            break
        if re.match(r'^\s*\d+\b', row):     # línea que inicia con un entero -> parte del bloque
            data_lines.append(row)
        else:
            # si aparece otra cabecera/sección, cortamos
            if row.strip().startswith('# ') or row.strip().startswith('##'):
                break
            # si no parece fila válida, también cortamos
            if not re.search(r'\d', row):
                break

    if not data_lines:
        raise ValueError("No se encontraron filas de datos tras la cabecera.")

    # 4) Leer con pandas (separador por espacios)
    df = pd.read_csv(
        StringIO('\n'.join(data_lines)),
        sep=r'\s+',
        engine='python',
        names=colnames
    )

    # 5) Opcional: asegurar tipo string para la columna de componentes (si existe)
    for c in df.columns:
        if 'component' in c.lower():
            df[c] = df[c].astype(str)
            break

    df.rename(columns = {'j':'x_j', 
                         'x_j':'Mini_j',
                         'Mini_j':'Mcor_j',
                         'Mcor_j':'age_j',
                         'age_j_yr':'Z_j',
                         'Z_j':'L/M_j',
                         'L_M__j':'YAV',
                         'YAV':'Mstars',
                         'Mstars':'component_j',
                         'component_j':'a/Fe',
                         'a_Fe':'SSP_chi2r',
                         'SSP_chi2r_SSP_adev':'SSP_adev',
                         'SSP_AV':'SSP_AV',
                         'SSP_x':'SSP_x'}
                         ,inplace=True)           #Data Frame corrected columns


    return df


### Reading of STARLIGHT Best Spectrum Fit from .out file ...


def read_starlight_best_model(path):
    """
    Lee el bloque:
    ## Synthetic spectrum (Best Model) ##l_obs f_obs f_syn wei Best_f_SSP
      <Nl_obs>   <index_Best_SSP>   [Nl_obs & index_Best_SSP]
      <datos...>
    y devuelve (df, meta), donde meta incluye Nl_obs e index_Best_SSP.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 1) Buscar cabecera del bloque
    hdr_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('## Synthetic spectrum (Best Model) ##'):
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError("No se encontró la cabecera de 'Synthetic spectrum (Best Model)'.")

    # 2) Extraer nombres de columnas (aparecen después del segundo '##')
    header_line = lines[hdr_idx].strip()
    # Ejemplo de línea:
    # "## Synthetic spectrum (Best Model) ##l_obs f_obs f_syn wei Best_f_SSP"
    m = re.search(r'##\s*([^#].*)$', header_line)  # toma lo que sigue después del segundo '##'
    if not m:
        raise ValueError("No se pudieron extraer los nombres de columna.")
    raw_cols = m.group(1).strip()
    raw_cols = raw_cols.replace('Synthetic spectrum (Best Model) ##','')

    colnames = re.split(r'\s+', raw_cols)
    # Sanear nombres
    colnames = [re.sub(r'\W+', '_', c).strip('_') for c in colnames]
    #print(colnames)

    # 3) La línea siguiente trae meta: "<Nl_obs> <index_Best_SSP> [Nl_obs & index_Best_SSP]"
    meta_idx = hdr_idx + 1
    meta_line = lines[meta_idx].strip()
    # Parsear primeros dos enteros y lo que hay dentro de corchetes
    ints = re.findall(r'\b\d+\b', meta_line)
    Nl_obs = int(ints[0]) if len(ints) > 0 else None
    index_best = int(ints[1]) if len(ints) > 1 else None
    bracket = re.search(r'\[(.*?)\]', meta_line)
    bracket_label = bracket.group(1).strip() if bracket else None
    meta = {
        "Nl_obs": Nl_obs,
        "index_Best_SSP": index_best,
        "meta_label": bracket_label,
        "meta_raw": meta_line
    }

    # 4) Recolectar filas de datos desde meta_idx+1 hasta que deje de haber numéricas
    data_lines = []
    for j in range(meta_idx + 1, len(lines)):
        row = lines[j].rstrip('\n')
        if not row.strip():
            break
        # filas válidas: comienzan con número (wavelength), separados por espacios
        if re.match(r'^\s*[+-]?\d+(\.\d+)?\b', row):
            data_lines.append(row)
        else:
            # si aparece otra cabecera/sección, cortamos
            if row.strip().startswith(('#', '##')):
                break
            # si no contiene números, cortamos
            if not re.search(r'\d', row):
                break

    if not data_lines:
        raise ValueError("No se encontraron filas de datos del 'Best Model'.")

    # 5) Leer con pandas usando separador de espacios
    df = pd.read_csv(
        StringIO('\n'.join(data_lines)),
        sep=r'\s+',
        engine='python',
        names=colnames
    )

    HDR = fits.Header()
    HDR.insert(0, ('COMMENT',''))
    HDR.insert(1, ('COMMENT','## Synthetic spectrum (Best Model) ##'))
    HDR.insert(2, ('COMMENT',''))
    HDR.insert(3,('Nl_obs', meta['Nl_obs'],'Nl_obs'))
    HDR.insert(4,('index_B', meta['index_Best_SSP'],'index_Best_SSP'))
    HDR.insert(5, ('COMMENT',''))

    return df, HDR



### Creation of STARLIGHT .fits file from .out file (original output)...


def FITS_conversion(id,
    home = os.path.expanduser("~") + '/HIIGs/STARLIGHT'):
    name = ''
    name = home + id
    warnings.filterwarnings('ignore')

    if os.path.exists(name) != True:
        raise FileNotFoundError('.out File is not in folder. Try relocating the file or changing the name')
    
    final_name = name.replace('.out',".fits")

    if os.path.exists(final_name) == True:
        print(f'Fits file already in {final_name}')
    else:  
        rands_x1hdu = np.random.random((1,1))
        primary_hdu = fits.PrimaryHDU(data=rands_x1hdu 
                        )
    
        SPS_hdu = fits.BinTableHDU(data = Table.from_pandas(read_starlight_ssp_txt(name)),
                        header=header_firstTABLE(name),
                        name = 'Stellar Composition')

        Spec_hdu = fits.BinTableHDU(data = Table.from_pandas(read_starlight_best_model(name)[0]),
                        header = read_starlight_best_model(name)[1],
                        name = 'Spectrum Fit')

        HDU = fits.HDUList([primary_hdu,
                        SPS_hdu,
                        Spec_hdu])
        

        HDU.writeto(final_name, overwrite=True)
        print(f'Fits file saved in {final_name}')



### Modeleded spectra vs observed plot from .fits file...



def spectra(id,
    home = os.path.expanduser("~") + '/gdrive/DataHII/HIIGs/STARLIGHT/'):
    name = ''
    name = home + id

    if os.path.exists(name) != True:
        raise FileNotFoundError('File is not in folder. Try relocating the file or changing the name')
    
    FITS = fits.open(name)


    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    gs.update(left=0.05, bottom=0.05, right=0.98, hspace=0.0)
    
    ax1 = plt.subplot(gs[0])

    TABLE = Table.read(FITS[2])

    l_obs = TABLE['l_obs']
    f_obs = TABLE['f_obs']
    f_syn = TABLE['f_syn']

    ax1.plot(l_obs, f_obs, label='Observed')
    ax1.plot(l_obs, f_syn, label='Synthetic')
    ax1.minorticks_on()

    ax1.tick_params(axis='x',which='major',labelbottom='off')
    ax1.set_ylim(0,2.5)

    ax1.set_title(str(FITS[1].header['ARQ_OBS']).replace('.tex',''))
    ax1.legend()

    chi2 = r'$\chi^2$/N_fre = %.4f' % FITS[1].header['CHI2NL_']
    adev = 'adev = %.4f' % FITS[1].header['ADEV']
    red_law = 'Reddening law = '+ str(FITS[1].header['RED_LAW'])
    n_base = r'N Base = %.0f' % FITS[1].header['N_BASE']


    ltext = [chi2, 
             adev,
             red_law,
             n_base]
    text = '\n'.join(ltext)

    ax1.annotate(text, xy=(0, 1), xytext=(15, -15), fontsize=10,
                 xycoords='axes fraction', textcoords='offset points',
                 bbox=dict(facecolor='0.95', alpha=0.9),
                 horizontalalignment='left', verticalalignment='top')

    
    ax2 = plt.subplot(gs[1])
    residual = (f_obs - f_syn)
    ax2.plot(l_obs, residual, color = 'g', alpha = 0.5)

    ax2.minorticks_on()
    minorLocator = AutoMinorLocator(2)
    ax2.yaxis.set_minor_locator(minorLocator)
    ax2.set_ylim(-0.2,0.49)

    ax1.set_ylabel(r'$F_\lambda$', fontsize=15)
    ax2.set_ylabel('Residual', fontsize=15)
    ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=15)

    ax1.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.5)
    ax2.grid(True, which="both", ls="--", color = 'gray', linewidth = 0.5)

    FITS.close()
    return fig


### Light Fraction - Mass Fraction vs Population Age from .fits file...


def stellar_pop(id,
    home = os.path.expanduser("~") + '/gdrive/DataHII/HIIGs/STARLIGHT/',Z_w = False):
    name = ''
    name = home + id

    if os.path.exists(name) != True:
        raise FileNotFoundError('File is not in folder. Try relocating the file or changing the name')
    
    FITS = fits.open(name)

    fig = plt.figure(figsize=(10,8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    gs.update(left=0.05, bottom=0.05, right=0.98, hspace=0.0)

    ax1 = plt.subplot(gs[0])

    TABLE = Table.read(FITS[1])
    COL = TABLE.columns

    x_j,mini_j,mcor_j,age_j,Z_j,LM_j,YAV,mstars,C_j,aFe,SSP_chi2r,SSP_adev,SSP_AV,SSP_x = (TABLE[col].data for col in COL)

    if Z_w == False:
        ax1.bar(age_j,x_j, width=0.1*age_j, color = "goldenrod")
        ax1.set_xscale('log')
        ax1.minorticks_on()
        ax1.tick_params(axis='x',which='major',labelbottom='off')

        ax1.set_title(str(FITS[1].header['ARQ_OBS']).replace('.tex',''))

        chi2 = r'$\chi^2$/N_fre = %.4f' % FITS[1].header['CHI2NL_']
        adev = 'adev = %.4f' % FITS[1].header['ADEV']
        red_law = 'Reddening law = '+ str(FITS[1].header['RED_LAW'])
        n_base = r'N Base = %.0f' % FITS[1].header['N_BASE']


        ltext = [chi2, 
                adev,
                red_law,
                n_base]
        text = '\n'.join(ltext)

        ax1.annotate(text, xy=(0.80, 1), xytext=(15, -15), fontsize=10,
                    xycoords='axes fraction', textcoords='offset points',
                    bbox=dict(facecolor='0.95', alpha=0.9),
                    horizontalalignment='left', verticalalignment='top')

        ax2 = plt.subplot(gs[1])
        ax2.bar(age_j,mcor_j, width=0.1*age_j, color = "goldenrod")
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        ax2.minorticks_on()
        #minorLocator = AutoMinorLocator(2)
        #ax2.yaxis.set_minor_locator(minorLocator)

        ax1.set_ylabel(r'Light Fraction',  fontsize=15)
        ax2.set_ylabel('Mass Fraction', fontsize=15)
        ax2.set_xlabel(r'log(Age)', fontsize=15)

        ax1.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.2)
        ax2.grid(True, which="both", ls="--", color = 'gray', linewidth = 0.2)

    FITS.close()
    return fig



### Open mask files and return values and labels


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






    edades = np.log10(np.unique(POPS_TABLE['age_j']))
    AGES_axis = [str(round(e,2)) for e in edades]

    metal_library = np.unique(POPS_TABLE['Z_j'])
    metallicities = {}
    for i in range(len(metal_library)):
        metallicities[f"{str(metal_library[i])}"] = np.zeros(len(edades))

    POPS_TABLE['x_j'] = (POPS_TABLE['x_j'] / (np.sum(POPS_TABLE['x_j']))) * 100

    for i in range(len(edades)):
        age_selection = POPS_TABLE[np.log10(POPS_TABLE['age_j']) == edades[i]]
        for x in metal_library:
            tmp = age_selection[age_selection['Z_j']==x]['x_j'].item()
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
                width,label=fr"$Z = {boolean}$", #width,label=fr"${boolean}\,Z_{{\odot}}$", 
                bottom=bottom,#, color=color
                alpha = 0.6,
                edgecolor = 'k')
        ax3.minorticks_on()
        ax3.tick_params(axis='x',which='major',labelbottom='off')
        bottom += weight_count

    ax3.set_xticks(edades)
    ax3.tick_params(axis='x', colors='red',width=1.5,length=5)
    ax3.set_xticklabels([str(round(e,2)) for e in edades], rotation=90, fontsize=7,color='blue')
    ax3.set_ylabel(r'$x_{j}$ [%] $L_{\lambda}$=4020$\AA$',  fontsize=15)
    ax3.legend(loc='upper right', fontsize=12,ncol = int(len(metal_library)/2))
    ax3.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.2)




    edades = np.log10(np.unique(POPS_TABLE['age_j']))
    AGES_axis = [str(round(e,2)) for e in edades]

    metal_library = np.unique(POPS_TABLE['Z_j'])
    metallicities = {}
    for i in range(len(metal_library)):
        metallicities[f"{str(metal_library[i])}"] = np.zeros(len(edades))

    POPS_TABLE['Mcor_j'] = (POPS_TABLE['Mcor_j'] / (POPS_TABLE['Mcor_j'].sum())) * 100


    for i in range(len(edades)):
        age_selection = POPS_TABLE[np.log10(POPS_TABLE['age_j']) == edades[i]]
        for x in metal_library:
            tmp = age_selection[age_selection['Z_j']==x]['Mcor_j'].item()
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
    mean_Z_L = np.sum(x_j_L_norm * POPS_TABLE['Z_j'])
    mean_Z_M = np.sum(mu_j_M_norm * POPS_TABLE['Z_j'])




    titulo_starlight = r'$\mathbfit{STATS \, BEST \, FIT}$'
    chi2 = r'$\mathbfit{\chi^2 / \nu}$ = %.4f' % FITS[1].header['CHI2NL_']
    adev = r'$\mathbfit{Adev}$ = %.4f' % FITS[1].header['ADEV']
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
             chi2, 
             adev,
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
    ax2.set_ylim(-0.20,0.2)
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
    
    edades = np.log10(np.unique(POPS_TABLE['age_j']))
    AGES_axis = [str(round(e,2)) for e in edades]

    metal_library = np.unique(POPS_TABLE['Z_j'])
    metallicities = {}
    for i in range(len(metal_library)):
        metallicities[f"{str(metal_library[i])}"] = np.zeros(len(edades))

    POPS_TABLE['x_j'] = (POPS_TABLE['x_j'] / (POPS_TABLE['x_j'].sum())) * 100

    for i in range(len(edades)):
        age_selection = POPS_TABLE[np.log10(POPS_TABLE['age_j']) == edades[i]]
        for x in metal_library:
            tmp = age_selection[age_selection['Z_j']==x]['x_j'].item()
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
                width,label=fr"$Z = {boolean}$", #width,label=fr"${boolean}\,Z_{{\odot}}$", 
                bottom=bottom,#, color=color
                alpha = 0.6,
                edgecolor = 'k')
        ax3.minorticks_on()
        ax3.tick_params(axis='x',which='major',labelbottom='off')
        bottom += weight_count

    ax3.set_xticks(edades)
    ax3.tick_params(axis='x', colors='red',width=1.5,length=5)
    ax3.set_xticklabels([str(round(e,2)) for e in edades], rotation=90, fontsize=7,color='blue')
    ax3.set_ylabel(r'$x_{j}$ [%] $L_{\lambda}$=4020$\AA$',  fontsize=15)
    ax3.legend(loc='upper right', fontsize=12,ncol = int(len(metal_library)/2))
    ax3.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.2)







    edades = np.log10(np.unique(POPS_TABLE['age_j']))
    AGES_axis = [str(round(e,2)) for e in edades]

    metal_library = np.unique(POPS_TABLE['Z_j'])
    metallicities = {}
    for i in range(len(metal_library)):
        metallicities[f"{str(metal_library[i])}"] = np.zeros(len(edades))

    POPS_TABLE['Mcor_j'] = (POPS_TABLE['Mcor_j'] / (POPS_TABLE['Mcor_j'].sum())) * 100


    for i in range(len(edades)):
        age_selection = POPS_TABLE[np.log10(POPS_TABLE['age_j']) == edades[i]]
        for x in metal_library:
            tmp = age_selection[age_selection['Z_j']==x]['Mcor_j'].item()
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
    mean_Z_L = np.sum(x_j_L_norm * POPS_TABLE['Z_j'])
    mean_Z_M = np.sum(mu_j_M_norm * POPS_TABLE['Z_j'])

    titulo_fado = r'$\mathbfit{STATS \, BEST \, FIT}$'
    chi2 = r'$\mathbfit{\chi^2 / \nu}$ = %.4f' % spec_header['CHI2_RED']
    adev = r'$\mathbfit{Adev}$ = %.4f' % Adev
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
             chi2, 
             adev,
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














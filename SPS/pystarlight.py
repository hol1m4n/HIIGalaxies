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

global home
global name
global hdr_params
global hdr_titles
global cat_line

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






def FITS_conversion(id,
    home = os.path.expanduser("~") + '/DataHII/HIIGs/Starlight/'):
    name = ''
    name = home + id
    

    if os.path.exists(name) != True:
        raise FileNotFoundError('.out File is not in folder. Try relocating the file or changing the name')
    
    final_name = (name.replace('.out',"")) + '.fits'

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




def spectra(id,
    home = os.path.expanduser("~") + '/DataHII/HIIGs/Starlight/'):
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

def stellar_pop(id,
    home = os.path.expanduser("~") + '/DataHII/HIIGs/Starlight/',Z_w = False ):
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

def fig_saver(id,spec = True,pop = True):
    name = ''
    name = id
    home = os.path.expanduser("~") + '/DataHII/HIIGs/Starlight/'
    if spec:
        outfile = 'specPLT'+ name.replace('.fits', '') + ".png"
        fig = spectra(name)     # spectra debe devolver un fig
        fig.savefig(home + outfile, dpi=120, bbox_inches='tight', transparent=False)
        plt.close(fig)
    if pop:
        outfile = 'popPLT'+ name.replace('.fits', '') + ".png"
        fig = stellar_pop(name)     # spectra debe devolver un fig
        fig.savefig(home + outfile, dpi=120, bbox_inches='tight', transparent=False)
        plt.close(fig)
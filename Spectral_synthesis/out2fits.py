import re
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
import pandas as pd
from io import StringIO
import warnings

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

def FITS_conversion(id,home):
    name = home + id
    warnings.filterwarnings('ignore')

    if os.path.exists(name) != True:
        raise FileNotFoundError('.out STARLIGHT file is not in folder. Try relocating it or changing the name')
    
    final_name = name.replace('.out',".fits")

    if os.path.exists(final_name) == True:
        print(f'STARLIGHT .fits file transformed before. Location: {final_name}')
        return 'OK'
    else:  
        rands_x1hdu = np.random.random((1,1))
        primary_hdu = fits.PrimaryHDU(data=rands_x1hdu 
                        )
    
        SPS_hdu = fits.BinTableHDU(data = Table.from_pandas(read_starlight_ssp_txt(name)),
                        header=header_firstTABLE(name),
                        name = 'Stellar Population Vector')

        Spec_hdu = fits.BinTableHDU(data = Table.from_pandas(read_starlight_best_model(name)[0]),
                        header = read_starlight_best_model(name)[1],
                        name = 'Spectrum Best Fit')

        HDU = fits.HDUList([primary_hdu,
                        SPS_hdu,
                        Spec_hdu])
        

        HDU.writeto(final_name, overwrite=True)
        print(f'STARLIGHT .out file sucessfuly transformed to .fits format. Location: {final_name}')

        return 'OK'





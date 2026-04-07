from astroquery.sdss import SDSS
from astroquery.ipac.irsa import Irsa
from astroquery.vizier import Vizier
from astropy import coordinates as coord
from astropy import units as u
from astroquery.ipac.ned import Ned
import numpy as np
import pandas as pd

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
import subprocess
from astropy.coordinates import Angle, SkyCoord
import warnings
from astropy.visualization.wcsaxes import SphericalCircle

import matplotlib.pyplot as plt
from astropy.wcs import WCS
import matplotlib.image as mpimg

from astroquery.ukidss import Ukidss
import astropy.coordinates as coord
import astropy.units as u
from astroquery.mast import Catalogs as Mastcat

from astroquery.ipac.ned import Ned
Ned.clear_cache()

import pyfiglet


storage_dir = os.path.join(os.path.expanduser("~"),'Data_storageHII/')
os.makedirs(storage_dir, exist_ok=True)

img_dir = storage_dir + '/HIIG_img/'

sdssIMGspec_dir = os.path.join(img_dir,'SDSS_specIMG/')
os.makedirs(sdssIMGspec_dir, exist_ok=True)

DF = pd.read_csv('/home/holman/HIIGalaxies/Spectral_synthesis/HIIGsample_data.csv')


def weighted_average(value,error):
    value,error = np.array(value),np.array(error)
    nan_remove = ~np.isnan(value) & ~np.isnan(error)
    value,error = value[nan_remove],error[nan_remove]
    inf_remove = [np.isfinite(value) & (value != -999.) & (value != -9999.0), np.isfinite(error) & (error != -999.) & (error != -9999.0)]
    value,error = value[inf_remove[0]],error[inf_remove[0]]
    if len(value) == 0  and len(error) == 0:
        return [np.nan,np.nan]
    else:
        err_sq_inv = 1 / ((error)**2)
        num = np.sum(err_sq_inv * value)
        dem = np.sum(err_sq_inv)
        return [(num/dem),(1 / np.sqrt(dem))]

def magAB_to_flux(mag, mag_err):
    f = 10**(-0.4 * (mag - 8.90))
    ferr = f * np.log(10)/2.5 * mag_err
    return f, ferr

def ukidss_to_mjy(mag, mag_err, band):
    """
    Convierte magnitudes Vega de UKIDSS a flux (mJy) con error.
    """
    UKIDSS_ZP = {
        'Z': 3503.0,
        'Y': 2026.0,
        'J': 1530.0,
        'H': 1019.0,
        'K': 631.0
    }
    f0_mjy = UKIDSS_ZP[band.upper()] * 1000.0
    f = f0_mjy * 10**(-0.4 * mag)
    ferr = f * (np.log(10) / 2.5) * mag_err
    return f, ferr
def vega_to_jy(mag, mag_err, band):
    """Convierte magnitudes Vega de WISE a flux (Jy) con error."""
    WISE_ZP = {
        'W1': 309.540,
        'W2': 171.787,
        'W3': 31.674,
        'W4': 8.363
    }
    f0 = WISE_ZP[band.upper()]
    f = f0 * 10**(-0.4 * mag)
    ferr = f * np.log(10)/2.5 * mag_err
    return f, ferr
def GALEX_data_mJy(ra,dec,tol,filter_band,name):
    warnings.filterwarnings('ignore')
    NED_table_bands = Ned.get_table(name, table='photometry')
    cigale_data = [np.nan for _ in range(len(filter_band*2))]
    if (filter_band[0] or filter_band[1]) in NED_table_bands['Observed Passband']:
        galex_tab = Mastcat.query_region(f"{ra} {dec}", catalog="Galex", radius=tol*u.arcsec)
        if len(galex_tab) == 0:
            return cigale_data
        if len(galex_tab) != 0:
            flx_FUV,flx_err_FUV = galex_tab['fuv_mag'],galex_tab['fuv_magerr']
            flx_NUV,flx_err_NUV = galex_tab['nuv_mag'],galex_tab['nuv_magerr']
            flx_AB_FUV,flx_err_AB_FUV = weighted_average(flx_FUV,flx_err_FUV)
            fx_Jy_FUV, fxerr_Jy_FUV = magAB_to_flux(flx_AB_FUV, flx_err_AB_FUV)
            flx_AB_NUV,flx_err_AB_NUV = weighted_average(flx_NUV,flx_err_NUV)
            fx_Jy_NUV, fxerr_Jy_NUV = magAB_to_flux(flx_AB_NUV, flx_err_AB_NUV)
            cigale_data[0],cigale_data[1] = fx_Jy_FUV * 1e3,fxerr_Jy_FUV * 1e3
            cigale_data[2],cigale_data[3] = fx_Jy_NUV * 1e3,fxerr_Jy_NUV * 1e3
            return cigale_data
    else:
        return cigale_data

def SDSS_data_mJy(ra,dec,tol,filter_band,name):
    warnings.filterwarnings('ignore')
    NED_table_bands = Ned.get_table(name, table='photometry')
    cigale_data = [np.nan for _ in range(len(filter_band*2))]
    if (filter_band[0] or filter_band[1] or filter_band[2] or filter_band[3] or filter_band[4]) in NED_table_bands['Observed Passband']:
        pos = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
        sdss_tab = SDSS.query_region(
            pos,
            radius=tol * u.arcsec,   # <--- este es el argumento que faltaba
            spectro=False,
            photoobj_fields=[
                'modelMag_u','modelMagErr_u',
                'modelMag_g','modelMagErr_g',
                'modelMag_r','modelMagErr_r',
                'modelMag_i','modelMagErr_i',
                'modelMag_z','modelMagErr_z'
            ]
        )
        if len(sdss_tab) == 0:
            return cigale_data
        if len(sdss_tab) != 0:
            flx_u,flx_err_u = sdss_tab['modelMag_u'],sdss_tab['modelMagErr_u']
            flx_g,flx_err_g = sdss_tab['modelMag_g'],sdss_tab['modelMagErr_g']
            flx_r,flx_err_r = sdss_tab['modelMag_r'],sdss_tab['modelMagErr_r']
            flx_i,flx_err_i = sdss_tab['modelMag_i'],sdss_tab['modelMagErr_i']
            flx_z,flx_err_z = sdss_tab['modelMag_z'],sdss_tab['modelMagErr_z']
            flx_AB_u,flx_err_AB_u = weighted_average(flx_u,flx_err_u)
            fx_Jy_u, fxerr_Jy_u = magAB_to_flux(flx_AB_u, flx_err_AB_u)
            flx_AB_u,flx_err_AB_u = weighted_average(flx_u,flx_err_u)
            fx_Jy_u, fxerr_Jy_u = magAB_to_flux(flx_AB_u, flx_err_AB_u)
            flx_AB_g,flx_err_AB_g = weighted_average(flx_g,flx_err_g)
            fx_Jy_g, fxerr_Jy_g = magAB_to_flux(flx_AB_g, flx_err_AB_g)
            flx_AB_r,flx_err_AB_r = weighted_average(flx_r,flx_err_r)
            fx_Jy_r, fxerr_Jy_r = magAB_to_flux(flx_AB_r, flx_err_AB_r)
            flx_AB_i,flx_err_AB_i = weighted_average(flx_i,flx_err_i)
            fx_Jy_i, fxerr_Jy_i = magAB_to_flux(flx_AB_i, flx_err_AB_i)
            flx_AB_z,flx_err_AB_z = weighted_average(flx_z,flx_err_z)
            fx_Jy_z, fxerr_Jy_z = magAB_to_flux(flx_AB_z, flx_err_AB_z)
            cigale_data[0],cigale_data[1] = fx_Jy_u * 1e3,fxerr_Jy_u * 1e3
            cigale_data[2],cigale_data[3] = fx_Jy_g * 1e3,fxerr_Jy_g * 1e3
            cigale_data[4],cigale_data[5] = fx_Jy_r * 1e3,fxerr_Jy_r * 1e3
            cigale_data[6],cigale_data[7] = fx_Jy_i * 1e3,fxerr_Jy_i * 1e3
            cigale_data[8],cigale_data[9] = fx_Jy_z * 1e3,fxerr_Jy_z * 1e3
            return cigale_data
    else:
        return cigale_data

def UKIDSS_data_mJy(ra,dec,tol,filter_band,name=''):
    warnings.filterwarnings('ignore')
    Vizier.ROW_LIMIT = 10000
    cigale_data = [np.nan for _ in range(len(filter_band*2))]
    pos = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    cat_id = "II/319/"
    ukidss_data = Vizier.query_region(pos, radius=tol*u.arcsec, catalog=cat_id)
    if ukidss_data:
        tab = ukidss_data[0]
        ukidss_sel = tab.to_pandas()
        if len(ukidss_sel) == 0:
            return cigale_data
        if len(ukidss_sel) != 0:
            def limpiar_y_promediar_bandas(df, bandas=['Z','Y', 'J', 'H', 'K']):
                """
                Busca columnas como K_mag, K_mag1, K_mag2, las promedia y 
                hace lo mismo con sus errores (e_Kmag, e_Kmag1...).
                """
                df_limpio = pd.DataFrame()
                for b in bandas:
                # 1. Buscar columnas de magnitud: que empiecen por la banda + _mag y opcionalmente un número
                # Ejemplo: J_mag, J_mag1, J_mag2...
                    # 2. Buscar columnas de error: e_ + banda + mag + número
                    # Ejemplo: e_Jmag, e_Jmag1, e_Jmag2...

                    regex_mag = f'^{b}_?mag\d*$'
                    regex_err = f'^e_{b}_?mag\d*$'

                    cols_mag = ukidss_sel.filter(regex=regex_mag, axis=1, items=None).columns
                    cols_err = ukidss_sel.filter(regex=regex_err, axis=1, items=None).columns

                    #print(cols_mag)
                    #print(cols_err)
                    if len(cols_mag) > 0:
                        # Reemplazamos posibles valores de error (como -9999 o strings) por NaN para promediar
                        # Convertimos a numérico por si vienen como strings
                        mags = df[cols_mag].apply(pd.to_numeric, errors='coerce').replace(-9999, np.nan)
                        errs = df[cols_err].apply(pd.to_numeric, errors='coerce').replace(-9999, np.nan)
                        # Promediamos (ignora los NaNs automáticamente)
                        df_limpio[f'{b}_mag'] = mags.mean(axis=1)
                        # Para los errores, usamos propagación del promedio: sqrt(sum(err^2)) / n
                        # O simplemente el promedio si prefieres: errs.mean(axis=1)
                        n = mags.notna().sum(axis=1)
                        df_limpio[f'e_{b}mag'] = np.sqrt((errs**2).sum(axis=1)) / n


                return df_limpio
            clean_sel = limpiar_y_promediar_bandas(ukidss_sel)

            if 'Z_mag' in clean_sel.columns:
                flx_Z,flx_err_Z = clean_sel['Z_mag'],clean_sel['e_Zmag']
                flx_mag_Z,flx_err_mag_Z = weighted_average(flx_Z,flx_err_Z)
                fx_Jy_Z, fxerr_Jy_Z = ukidss_to_mjy(flx_mag_Z, flx_err_mag_Z,'Z')
                cigale_data[0],cigale_data[1] = fx_Jy_Z ,fxerr_Jy_Z
            else:
                cigale_data[0],cigale_data[1] = np.nan,np.nan

            if 'Y_mag' in clean_sel.columns:
                flx_Y,flx_err_Y = clean_sel['Y_mag'],clean_sel['e_Ymag']
                flx_mag_Y,flx_err_mag_Y = weighted_average(flx_Y,flx_err_Y)
                fx_Jy_Y, fxerr_Jy_Y = ukidss_to_mjy(flx_mag_Y, flx_err_mag_Y,'Y')
                cigale_data[2],cigale_data[3] = fx_Jy_Y ,fxerr_Jy_Y
            else:
                cigale_data[2],cigale_data[3] = np.nan,np.nan

            if 'J_mag' in clean_sel.columns:
                flx_J,flx_err_J = clean_sel['J_mag'],clean_sel['e_Jmag']
                flx_mag_J,flx_err_mag_J = weighted_average(flx_J,flx_err_J)
                fx_Jy_J, fxerr_Jy_J = ukidss_to_mjy(flx_mag_J, flx_err_mag_J,'J')
                cigale_data[4],cigale_data[5] = fx_Jy_J ,fxerr_Jy_J
            else:
                cigale_data[4],cigale_data[5] = np.nan,np.nan

            if 'H_mag' in clean_sel.columns:
                flx_H,flx_err_H = clean_sel['H_mag'],clean_sel['e_Hmag']
                flx_mag_H,flx_err_mag_H = weighted_average(flx_H,flx_err_H)
                fx_Jy_H, fxerr_Jy_H = ukidss_to_mjy(flx_mag_H, flx_err_mag_H,'H')
                cigale_data[6],cigale_data[7] = fx_Jy_H ,fxerr_Jy_H
            else:
                cigale_data[6],cigale_data[7] = np.nan,np.nan

            if 'K_mag' in clean_sel.columns:
                flx_K,flx_err_K = clean_sel['K_mag'],clean_sel['e_Kmag']
                flx_mag_K,flx_err_mag_K = weighted_average(flx_K,flx_err_K)
                fx_Jy_K, fxerr_Jy_K = ukidss_to_mjy(flx_mag_K, flx_err_mag_K,'K')
                cigale_data[8],cigale_data[9] = fx_Jy_K ,fxerr_Jy_K
            else:
                cigale_data[8],cigale_data[9] = np.nan,np.nan
 
            return cigale_data
    else:
        return cigale_data

def WISE_data_mJy(ra,dec,tol,filter_band,name):
    warnings.filterwarnings('ignore')
    NED_table_bands = Ned.get_table(name, table='photometry')
    cigale_data = [np.nan for _ in range(len(filter_band*2))]
    if (filter_band[0] or filter_band[1] or filter_band[2] or filter_band[3]) in NED_table_bands['Observed Passband']:
        pos = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
        wise_tab = Irsa.query_region(pos, 
                        catalog='allwise_p3as_psd', 
                        spatial='Cone', 
                        radius=tol*u.arcsec)
        if len(wise_tab) == 0:
            return cigale_data
        if len(wise_tab) != 0:
            flx_W1,flx_err_W1 = wise_tab['w1mpro'],wise_tab['w1sigmpro']
            flx_W2,flx_err_W2 = wise_tab['w2mpro'],wise_tab['w2sigmpro']
            flx_W3,flx_err_W3 = wise_tab['w3mpro'],wise_tab['w3sigmpro']
            flx_W4,flx_err_W4 = wise_tab['w4mpro'],wise_tab['w4sigmpro']
            flx_Vg_W1,flx_err_Vg_W1 = weighted_average(flx_W1,flx_err_W1)
            fx_Jy_W1, fxerr_Jy_W1 = vega_to_jy(flx_Vg_W1, flx_err_Vg_W1,'W1')
            flx_Vg_W2,flx_err_Vg_W2 = weighted_average(flx_W2,flx_err_W2)
            fx_Jy_W2, fxerr_Jy_W2 = vega_to_jy(flx_Vg_W2, flx_err_Vg_W2,'W2')
            flx_Vg_W3,flx_err_Vg_W3 = weighted_average(flx_W3,flx_err_W3)
            fx_Jy_W3, fxerr_Jy_W3 = vega_to_jy(flx_Vg_W3, flx_err_Vg_W3,'W3')
            flx_Vg_W4,flx_err_Vg_W4 = weighted_average(flx_W4,flx_err_W4)
            fx_Jy_W4, fxerr_Jy_W4 = vega_to_jy(flx_Vg_W4, flx_err_Vg_W4,'W4')
            cigale_data[0],cigale_data[1] = fx_Jy_W1 * 1e3,fxerr_Jy_W1 * 1e3
            cigale_data[2],cigale_data[3] = fx_Jy_W2 * 1e3,fxerr_Jy_W2 * 1e3
            cigale_data[4],cigale_data[5] = fx_Jy_W3 * 1e3,fxerr_Jy_W3 * 1e3
            cigale_data[6],cigale_data[7] = fx_Jy_W4 * 1e3,fxerr_Jy_W4 * 1e3
            return cigale_data
    else:        
        return cigale_data

filters = ['FUV (GALEX) AB', 'NUV (GALEX) AB',
           'u (SDSS Model) AB','g (SDSS Model) AB','r (SDSS Model) AB','i (SDSS Model) AB','z (SDSS Model) AB',
           'Z (UKIDSS)','Y (UKIDSS)','J (UKIDSS)','H (UKIDSS)','K (UKIDSS)',
            'W1 (WISE)','W2 (WISE)','W3 (WISE)','W4 (WISE)']

head = "# id redshift galex.FUV galex.FUV_err galex.NUV galex.NUV_err sloan.sdss.u sloan.sdss.u_err sloan.sdss.g sloan.sdss.g_err sloan.sdss.r sloan.sdss.r_err sloan.sdss.i sloan.sdss.i_err sloan.sdss.z sloan.sdss.z_err ukirt.wfcam.Z ukirt.wfcam.Z_err ukirt.wfcam.Y ukirt.wfcam.Y_err ukirt.wfcam.J ukirt.wfcam.J_err ukirt.wfcam.H ukirt.wfcam.H_err ukirt.wfcam.K ukirt.wfcam.K_err wise.W1 wise.W1_err wise.W2 wise.W2_err wise.W3 wise.W3_err wise.W4 wise.W4_err \n"

s_id = []

tol = 1


fig = pyfiglet.Figlet(font='letters')
text = fig.renderText('Comienza loop')
print(f"{text} \n")
print(f"\n")





for T in range(len(DF)):#
    galex = GALEX_data_mJy(DF['RA'].iloc[T], DF['DEC'].iloc[T],tol,filters[0:2],DF['NED_NAME'].iloc[T])
    sdss = SDSS_data_mJy(DF['RA'].iloc[T], DF['DEC'].iloc[T],1,filters[2:7],DF['NED_NAME'].iloc[T])
    ukidss = UKIDSS_data_mJy(DF['RA'].iloc[T], DF['DEC'].iloc[T],tol,filters[7:12],DF['NED_NAME'].iloc[T])
    wise = WISE_data_mJy(DF['RA'].iloc[T], DF['DEC'].iloc[T],tol,filters[12:],DF['NED_NAME'].iloc[T])
    cigale_comp = [f"_{DF['TAB5_INDEX'].iloc[T]}_",DF['Z'].iloc[T]] + galex + sdss + ukidss + wise
    s_id.append(cigale_comp)
    print(f"Data added for object {T} / {DF['TAB5_INDEX'].iloc[T]} / {DF['NED_NAME'].iloc[T]} \n")
    print(f"{cigale_comp} \n")
    print(f"\n")


text = fig.renderText('Termina loop')
print(f"{text} \n")
print(f"\n")


text = fig.renderText('Guardando datos')
print(f"{text} \n")
print(f"\n")

with open("cigale_HIIG_sample.txt", 'w') as file:
    file.write(head)
    for p in range(len(s_id)):
        output = f"{s_id[p][0]} {s_id[p][1]} {s_id[p][2]} {s_id[p][3]} {s_id[p][4]} {s_id[p][5]} {s_id[p][6]} {s_id[p][7]} {s_id[p][8]} {s_id[p][9]} {s_id[p][10]} {s_id[p][11]} {s_id[p][12]} {s_id[p][13]} {s_id[p][14]} {s_id[p][15]} {s_id[p][16]} {s_id[p][17]} {s_id[p][18]} {s_id[p][19]} {s_id[p][20]} {s_id[p][21]} {s_id[p][22]} {s_id[p][23]} {s_id[p][24]} {s_id[p][25]} {s_id[p][26]} {s_id[p][27]} {s_id[p][28]} {s_id[p][29]} {s_id[p][30]} {s_id[p][31]}  {s_id[p][32]}\n"

        file.write(output)

text = fig.renderText('Finalizado con exito')
print(f"{text} \n")
print(f"\n")














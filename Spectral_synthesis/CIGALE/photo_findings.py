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
matplotlib.use("Agg")
plt.ioff()

storage_dir = os.path.join(os.path.expanduser("~"),'Data_storageHII/')
os.makedirs(storage_dir, exist_ok=True)

img_dir = storage_dir + '/HIIG_img/'

sdssIMGspec_dir = os.path.join(img_dir,'SDSS_specIMG/')
os.makedirs(sdssIMGspec_dir, exist_ok=True)

DF = pd.read_csv('/home/holman/HIIGalaxies/Spectral_synthesis/HIIGsample_data.csv')
img_list = os.listdir(img_dir)
img_list.remove('SDSS_specIMG')
img_list = sorted(img_list)

def galex_coord(ra,dec,tol):
    Vizier.ROW_LIMIT = 10000
    warnings.filterwarnings('ignore')
    pos = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    A, B = pos.ra.deg, pos.dec.deg
    galex = Mastcat.query_region(f"{A} {B}", catalog="Galex", radius=tol*u.arcsec)
    match_x,match_y = [],[]
    if len(galex) != 0:
        for e in range(len(galex)):
            coord_p = SkyCoord(galex['ra'][e],galex['dec'][e], unit=(u.deg, u.deg), frame='icrs')
            x, y = coord_p.to_pixel(wcs, origin=0)
            match_x.append(x)
            match_y.append(y)
        return match_x,match_y
    else:
        return match_x,match_y

def sdss_coord(ra,dec,tol):
    warnings.filterwarnings('ignore')
    pos = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    sdss = SDSS.query_region(
        pos,
        radius=tol * u.arcsec,   # <--- este es el argumento que faltaba
        spectro=False,
        photoobj_fields=[
            'ra','dec',])
    match_x,match_y = [],[]
    if len(sdss) != 0:
        for e in range(len(sdss)):
            coord_p = SkyCoord(sdss['ra'][e],sdss['dec'][e], unit=(u.deg, u.deg), frame='icrs')
            x, y = coord_p.to_pixel(wcs, origin=0)
            match_x.append(x)
            match_y.append(y)
        return match_x,match_y
    else:
        return match_x,match_y

def ukidss_coord(ra,dec,tol):
    warnings.filterwarnings('ignore')
    pos = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    # Catálogo UKIDSS LAS en VizieR
    cat_id = "II/319/"  # DR9 LAS (hay DR10/DR11 también)  
    # Consulta por radio
    res = Vizier.query_region(pos, radius=tol*u.arcsec, catalog=cat_id)
    match_x,match_y = [],[]
    if res:
        tab = res[0]
        df_ukidss = tab.to_pandas()
        for e in range(len(df_ukidss)):
            coord_p = SkyCoord(df_ukidss['RAJ2000'][e],df_ukidss['DEJ2000'][e], unit=(u.deg, u.deg), frame='icrs')
            x, y = coord_p.to_pixel(wcs, origin=0)
            match_x.append(x)
            match_y.append(y)
        return match_x,match_y
    else:
        return match_x,match_y
def wise_coord(ra,dec,tol):
    warnings.filterwarnings('ignore')
    pos = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    wise = Irsa.query_region(pos, 
                         catalog='allwise_p3as_psd', 
                         spatial='Cone', 
                         radius=tol*u.arcsec)
    match_x,match_y = [],[]
    if len(wise) != 0:
        for e in range(len(wise)):
            coord_p = SkyCoord(wise['ra'][e],wise['dec'][e], unit=(u.deg, u.deg), frame='icrs')
            x, y = coord_p.to_pixel(wcs, origin=0)
            match_x.append(x)
            match_y.append(y)
        return match_x,match_y
    else:
        return match_x,match_y
def coordinate_search(spec_name,table):
    spec_name = spec_name.replace('.fit.jpeg','.fit')
    table = table[table['SDSS_PMF']==spec_name]
    ra,dec = table['RA'].iloc[0],table['DEC'].iloc[0]
    return [ra,dec]

centroids = [coordinate_search(x,DF) for x in img_list] 
tol = 3

for t in range(121):
    chosen = DF['SDSS_PMF'][t] + '.jpeg'
    Sel = img_list.index(chosen)

    # 1. Load the PNG
    img = mpimg.imread(img_dir + img_list[Sel])

    img = np.flipud(img)

    # 2. Define WCS information (example for 1000x1000 image)
    wcs_dict = {
        'CTYPE1': 'RA---SIN', 'CTYPE2': 'DEC--SIN',
        'CRVAL1': centroids[Sel][0], 'CRVAL2': centroids[Sel][1], # RA/Dec of reference pixel
        'CRPIX1': 100., 'CRPIX2': 100.,       # Reference pixel
        'CDELT1': -0.000057, 'CDELT2': 0.000056,  # Pixel scale
        'CUNIT1': 'deg', 'CUNIT2': 'deg',
        'RADESYS': 'ICRS',
        'LONPOLE' : 180.0,'LATPOLE' : 12.9915,
        'MJDREFI' :0.0,'MJDREFF' :0.0,
    }
    wcs = WCS(wcs_dict)

    ra_o, dec_o = centroids[Sel][0],centroids[Sel][1]
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), 
                        subplot_kw={'projection': wcs})
    
    g_c = [[0,0,1,1],[0,1,0,1]]

    for k in range(4):

        if k == 0:
            obj_x, obj_y = galex_coord(ra_o,dec_o,tol)
            label = f'galex: tol = {tol} arcsec \n N obj = {len(obj_x)}'
        elif k ==1:
            obj_x, obj_y = sdss_coord(ra_o,dec_o,tol)
            label = f'sdss: tol = {tol} arcsec \n N obj = {len(obj_x)}'
        elif k == 2:    
            obj_x, obj_y = ukidss_coord(ra_o,dec_o,tol)
            label = f'ukidss: tol = {tol} arcsec \n N obj = {len(obj_x)}'
        elif k == 3:
            obj_x, obj_y = wise_coord(ra_o,dec_o,tol)
            label = f'wise: tol = {tol} arcsec \n N obj = {len(obj_x)}'

        ax[g_c[0][k],g_c[1][k]].imshow(img,origin='lower')
        ax[g_c[0][k],g_c[1][k]].grid(color='white', ls=':')
        ny, nx = img.shape[0],img.shape[1]
        centro_sky = wcs.pixel_to_world(nx / 2, ny / 2)
        c = SphericalCircle(centro_sky, tol * u.arcsec, 
                            edgecolor='yellow', facecolor='none',
                            transform=ax[g_c[0][k],g_c[1][k]].get_transform('world'),alpha=0.5)
        ax[g_c[0][k],g_c[1][k]].add_patch(c)

        if len(obj_x) != 0 and len(obj_y)!= 0:
            ax[g_c[0][k],g_c[1][k]].plot(obj_x, obj_y,
                markerfacecolor="#f30101ff", marker='o', markersize=3, linewidth = 0, markeredgewidth = 1,markeredgecolor='white',
                label=label,alpha= 0.8)
            ax[g_c[0][k],g_c[1][k]].legend()

    fig.suptitle(f"{DF['TAB5_INDEX'].iloc[t]} / {DF['NED_NAME'].iloc[t]}", fontsize=16, fontweight='bold')
    fig.savefig(f'/home/holman/HIIGalaxies/Spectral_synthesis/CIGALE/Photometry_findings/{DF["TAB5_INDEX"].iloc[t]}.png')
    print(f'Done for {DF["TAB5_INDEX"].iloc[t]} / {DF["NED_NAME"].iloc[t]} \n')


















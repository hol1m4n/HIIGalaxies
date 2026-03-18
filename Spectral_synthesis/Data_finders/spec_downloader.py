import os
import sys
import glob
import numpy as np
#import  scipy.optimize as op
#import matplotlib.pyplot as plt
import urllib.request
#from astropy.io import fits as fits
#from astropy.table import Table
import pandas as pd
import urllib.request
import requests
#import astropy.units as u
#import astropy
#from astropy.coordinates import SkyCoord
from pathlib import Path
from numpy import savetxt
#from PyAstronomy import pyasl
#from scipy.interpolate import interp1d
#from dustmaps.sfd import SFDQuery
#import astropy.cosmology.units as cu
#from astropy.coordinates import match_coordinates_sky
import io
#from astroquery.sdss import SDSS
#from astropy import coordinates as coords
#import matplotlib.image as mpimg
#from astroquery.ipac.ned import Ned as ned
#import seaborn as sns
#import warnings
#from mw_plot import mw_radec # milkyway plane in RA/DEC
#from matplotlib import gridspec
#from matplotlib.patches import FancyArrowPatch
#from collections import namedtuple, OrderedDict
#import gc
#import matplotlib
#matplotlib.use("Agg")
#plt.ioff()
#!pip install gdown
import gdown
import subprocess
#!pip install joblib
from joblib import Parallel, delayed


storage_dir = os.path.join(os.path.expanduser("~"),'Data_storageHII/')
os.makedirs(storage_dir, exist_ok=True)

DATA_URL="https://wwwmpa.mpa-garching.mpg.de/SDSS/DR7/Data/"
file="gal_info_dr7_v5_2.fit.gz"
local_file = os.path.join(storage_dir,file)

if not os.path.exists(local_file):
        print("downloading DR7 quasar dataset from %s to %s"
              % (DATA_URL+file, local_file ))
        tmp = urllib.request.urlretrieve(DATA_URL+local_file, local_file)
        print ("Downloaded file"+local_file)
else:
    print("%s already exist"%(local_file))

file="NED_Chavez2012.csv"
local_file = os.path.join(storage_dir,file)

file_id = '1ABf-_YToMmonC0hZCR-cLUSSHmpKLpRe'
DATA_URL = f'https://drive.google.com/uc?id={file_id}'


if not os.path.exists(local_file):
        print("downloading NED Chavez galaxy list from %s to %s"
              % (DATA_URL, local_file ))
        gdown.download(DATA_URL, local_file, quiet=False)
        print ("Downloaded file"+local_file)
else:
    print("%s already exist"%(local_file))

NED = pd.read_csv(storage_dir + '/NED_Chavez2012.csv', sep=",", header=0)


def g_drivedownloader(folder_link,local_folder):
    gdown.download_folder(folder_link, output=local_folder, use_cookies=True)
    return 'Done parallelized'


dr7_spec_p1 = 'https://drive.google.com/drive/folders/1B8EyJcQ5lPyaG2FYrxTfQAdJnunSYfDf?usp=sharing'
dr7_spec_p2 = 'https://drive.google.com/drive/folders/1ks5clofcC2VLzXVssHIYM5ykbcXqIEVH?usp=sharing'
dr7_spec_p3 = 'https://drive.google.com/drive/folders/1842LTHQrruwBfNW6kH1l9yiJd1i3HGkd?usp=sharing'
DR7Spectra_folder = os.path.join(storage_dir,'DR7SpectraDbox')
spec_folder_parts = [[dr7_spec_p1,dr7_spec_p2,dr7_spec_p3],[DR7Spectra_folder+'/p1/',DR7Spectra_folder+'/p2/',DR7Spectra_folder+'/p3/']]


if not os.path.exists(DR7Spectra_folder):

    print('No folder of DR7Spectra from Dropbox. Downloading data... \n')
    os.mkdir(DR7Spectra_folder)
    resultados = Parallel(n_jobs=-1)(delayed(g_drivedownloader)(i,u) for i,u in zip(spec_folder_parts[0],spec_folder_parts[1]))
    #resultados = Parallel(n_jobs=-1)(delayed(g_drivedownloader)(i,DR7Spectra_folder) for i in spec_folder_parts[0])
    #gdown.download_folder(dr7_spec_p1, output=DR7Spectra_folder, use_cookies=True)
    #gdown.download_folder(dr7_spec_p2, output=DR7Spectra_folder, use_cookies=True)
    #gdown.download_folder(dr7_spec_p3, output=DR7Spectra_folder, use_cookies=True)
    subprocess.run(['bash', 'DR7SpectraDownload.sh'], capture_output=True, text=True)
    #resultado = subprocess.run(['ls', '-l'], capture_output=True, text=True)


else:
    print('La carpeta esta en el directorio local. Revisar si estan los archivos apropiadamente.')



def SDSS_namerecover(name):
    name = str(name)
    name = name.replace('.fit','')
    name = name.replace('spSpec-','')
    name = name.replace('-',',')
    return name

Dbox_ls = os.listdir(DR7Spectra_folder)
Dbox_ls

SDSS_string = "MJD,PLATEID,FIBERID\n"

for i in Dbox_ls:
    A = SDSS_namerecover(i)
    SDSS_string = SDSS_string + A +"\n"
    #print(A)
    del A

SDSS_df_Dbox = pd.read_csv(io.StringIO(SDSS_string), sep=',')



def check_link(path): # Function that takes care of seeing if the file exists on the web or not
    r = requests.head(path)
    status = r.status_code == requests.codes.ok
    if (status == True):
        return True
    else:
        return False
    

FOLDER_dr7link = ['1d_23','1d_25','1d_26']
URL_DR7 = "https://data.sdss.org/sas/dr7/das2/spectro/"
URL_DR17 =  "https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/"


PREdwn_listDR7 =[]
dwn_listDR7 =[]
PREdwn_listDR17 =[]
dwn_listDR17 =[]

for n in range(len(SDSS_df_Dbox)):

    plate_key,mjd_key,fiber_key = str(SDSS_df_Dbox.iloc[n]['PLATEID']),str(SDSS_df_Dbox.iloc[n]['MJD']),str(SDSS_df_Dbox.iloc[n]['FIBERID'])

    if (len(plate_key)<4) or (len(mjd_key)<5) or (len(fiber_key)<3):
        if (len(plate_key)<4):
            for x in range(4-len(plate_key)):
                plate_key = '0' + plate_key
        if (len(mjd_key)<5):
            for y in range(5-len(mjd_key)):
                mjd_key = '0' + mjd_key
        if (len(fiber_key)<3):
            for z in range(3-len(fiber_key)):
                fiber_key = '0' + fiber_key
            
    pre_link = URL_DR7 + FOLDER_dr7link[0] + '/' + plate_key +"/1d/spSpec" + '-' + mjd_key + '-' + plate_key+ '-' + fiber_key + '.fit'

    PREdwn_listDR7.append(pre_link)

    plate_key,mjd_key,fiber_key = str(SDSS_df_Dbox.iloc[n]['PLATEID']),str(SDSS_df_Dbox.iloc[n]['MJD']),str(SDSS_df_Dbox.iloc[n]['FIBERID'])

    if (len(plate_key)<4) or (len(mjd_key)<5) or (len(fiber_key)<4):
        if (len(plate_key)<4):
            for x in range(4-len(plate_key)):
                plate_key = '0' + plate_key
        if (len(mjd_key)<5):
            for y in range(5-len(mjd_key)):
                mjd_key = '0' + mjd_key
        if (len(fiber_key)<4):
            for z in range(4-len(fiber_key)):
                fiber_key = '0' + fiber_key

    pre_link = URL_DR17 + plate_key +"/spec-" + plate_key + '-' + mjd_key+ '-' + fiber_key + '.fits'

    PREdwn_listDR17.append(pre_link)


def list_checker_forDR7(obj_link):
    plate_key,mjd_key,fiber_key = obj_link[70:74],obj_link[64:69],obj_link[75:78]
    if check_link(obj_link)==False:
        obj_link = obj_link.replace(FOLDER_dr7link[0],FOLDER_dr7link[1])
        if check_link(obj_link)==False:
            obj_link = obj_link.replace(FOLDER_dr7link[1],FOLDER_dr7link[2])
            if check_link(obj_link)==False:
                rta = 'Not Found'
            else:
                rta = obj_link
        else:
            rta = obj_link
    else:
        rta = obj_link
    print("Finished for: ",plate_key,mjd_key,fiber_key,'\n')

    return rta

dwn_listDR7 = Parallel(n_jobs=-1)(delayed(list_checker_forDR7)(i) for i in PREdwn_listDR7)


def list_checker_forDR17(obj_link):
    plate_key,mjd_key,fiber_key = obj_link[71:75],obj_link[76:81],obj_link[82:86]
    if check_link(obj_link)==False:
        rta = 'Not Found'
    else:
        rta = obj_link
    print("Finished for: ",plate_key,mjd_key,fiber_key,'\n')

    return rta

dwn_listDR17 = Parallel(n_jobs=-1)(delayed(list_checker_forDR17)(i) for i in PREdwn_listDR17)


dwn_listDR7 = [item for item in dwn_listDR7 if item != 'Not Found']
fits_folder = '/HIIG_specDR7/'

if (os.path.exists(storage_dir + fits_folder) == True):
    print('Folder for spec set saving already created \n')
else:
    print(f'Folder for spec set saving created at ({storage_dir+fits_folder}) \n')
    os.mkdir(storage_dir+fits_folder)

def dr7_downloader(LINK,storage_dir,fits_folder):
    print(f"Downloading {LINK[-25:]} ...")
    direc = storage_dir + fits_folder + LINK[-25:]
    DOWNLOAD = urllib.request.urlretrieve(LINK, direc)
    return f"Downloading {LINK[-25:]} ..."

tmp = Parallel(n_jobs=-1)(delayed(dr7_downloader)(i,storage_dir,fits_folder) for i in dwn_listDR7)


dwn_listDR17 = [item for item in dwn_listDR17 if item != 'Not Found']
fits_folder = '/HIIG_specDR17/'

if (os.path.exists(storage_dir + fits_folder) == True):
    print('Folder for spec set saving already created \n')
else:
    print(f'Folder for spec set saving created at ({storage_dir+fits_folder}) \n')
    os.mkdir(storage_dir+fits_folder)

def dr17_downloader(LINK,storage_dir,fits_folder):
    print(f"Downloading {LINK[-25:]} ...")
    direc = storage_dir + fits_folder + LINK[-25:]
    DOWNLOAD = urllib.request.urlretrieve(LINK, direc)
    return f"Downloading {LINK[-25:]} ..."

tmp = Parallel(n_jobs=-1)(delayed(dr17_downloader)(i,storage_dir,fits_folder) for i in dwn_listDR17)

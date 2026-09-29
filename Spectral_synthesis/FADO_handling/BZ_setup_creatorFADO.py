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

import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import LambdaCDM, FlatLambdaCDM
from astropy.cosmology import Planck18



def z_get_dist(name):
    DF = pd.read_csv('~/HIIGalaxies/Spectral_synthesis/HIIGsample_main.csv')
    PLATE,MJD,FIBER = int(name[14:18]),int(name[8:13]),int(name[19:22])
    DF_n = DF[ (DF['MJD']==MJD) & (DF['PLATE']==PLATE) & (DF['FIBERID']==FIBER) ]
    if len(DF_n)==1:
        z = DF_n['Z'].iloc[0]
    else:
        return f'Not found'
    Z = z * cu.redshift
    d = Z.to(u.Mpc, cu.redshift_distance(Planck18, kind="luminosity"))
    return round(d.value,3)


storage_dir = os.path.join(os.path.expanduser("~"),'Data_storageHII/')
os.makedirs(storage_dir, exist_ok=True)

tex_folder = storage_dir + 'specTx_Tab5ID/'
tex_list = os.listdir(tex_folder)
tex_list.sort()


BZ_libs = ['BZ19Stndr','BZ19Step1','BZ19Step5','BZ19Step5Z10','BZ19Stp5Z10.M300','BZ19Stp5Z10.M600']


for LIB_CASE in BZ_libs:
    n_spec = '1'
    base_dir = f'/home/holman/BASES/Base_{LIB_CASE}/' # Cambiar
    obs_dir = '/home/holman/Data_storageHII/specTx_Tab5ID/'
    out_dir = f'/home/holman/FADOv1b/o_{LIB_CASE}/' # Cambiar
    plots_dir = f'/home/holman/FADOv1b/p_{LIB_CASE}/' # Cambiar
    input_folder = f'/home/holman/FADOv1b/i_{LIB_CASE}/' # Cambiar
    config = '/home/holman/FADOv1b/FR_4020.config'

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(input_folder, exist_ok=True)


    units  = 1.e-17
    d = 20
    res = 2.3
    Olsyn_ini = '3400.0'
    Olsyn_fin = '8500.0'
    Odlsyn = '1.0'
    ext_laws = ['GOR1']
    BASE = f'Base_bruz2019_{LIB_CASE}' # Cambiar

    for i in range(len(tex_list)):
        with open(f"{input_folder}/{LIB_CASE}{tex_list[i][0:3]}.txt", 'w') as file: # Cambiar
            n_spec = int(n_spec)
            Distancia = z_get_dist(tex_list[i])   
            for aux in range(n_spec):
                file.write(f"./FADO -i {obs_dir + tex_list[i]} -b {base_dir +  BASE} -s {Olsyn_ini} {Olsyn_fin} {Odlsyn} -r {res} -d {Distancia} -e {ext_laws[aux]} -o {out_dir+ tex_list[i][0:3]+ext_laws[aux]}.{LIB_CASE} -p {plots_dir + tex_list[i][0:3]+ ext_laws[aux]}.{LIB_CASE} -u {units} -c {config}\n") # Cambiar








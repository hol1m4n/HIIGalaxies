from __future__ import annotations
import os
import json
import numpy as np
from astropy.table import Table
import pandas as pd

from astropy.cosmology import FlatLambdaCDM
import pymultinest
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import scipy.optimize as op
import astropy.units as u
from scipy import stats
import scipy.optimize as op
from astropy import constants as const
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import math
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp
from matplotlib import gridspec
import argparse
import shutil


def READ_AND_CLEAN(SET,sub_id,anchor_group):

    home = os.path.expanduser('~/HIIGalaxies/Bayesian_samplers/Multinest/MOCKS')
    anchor_group = anchor_group.replace('_recal.csv','')
    os.makedirs(f'{home}/Results/{SET}/{anchor_group}/Plots', exist_ok = True)

    ruta_carpeta = f"{home}/Results/{SET}/{anchor_group}"
    ruta_archivo = f"{ruta_carpeta}/Results.csv"

    os.makedirs(ruta_carpeta, exist_ok=True)

    complete_run = False
    if os.path.exists(ruta_archivo) == True:
        results_file = pd.read_csv(ruta_archivo,comment="#",index_col=False)
        results_check = results_file[(results_file['code_id'] == f"{SET}.{args.sub_id}") & (results_file['subgrupo'] == f"{anchor_group}")]
        if len(results_check) == 1:
            complete_run = True


    if complete_run == False:

        shutil.copy(f"{home}/{SET}/{SET}.{sub_id}_{anchor_group}/{SET}.{sub_id}_run.png", f'{home}/Results/{SET}/{anchor_group}/Plots/{SET}.{sub_id}_run.png')

        results_path = f"{home}/{SET}/{SET}.{sub_id}_{anchor_group}/{SET}.{sub_id}_"
        analyzer = pymultinest.analyse.Analyzer(
            n_params=3,
            outputfiles_basename=results_path
        )
        stats = analyzer.get_stats()
        alpha_mean,beta_mean,h_mean = stats['modes'][0]['mean']
        alpha_err,beta_err,h_err = stats['modes'][0]['sigma']



        if not os.path.exists(ruta_archivo):
            with open(ruta_archivo, "w") as archivo:
                archivo.write("carpeta,code_id,subgrupo,variable,valor,error,H0,e_H0 \n")
                archivo.write(f"{SET}.{sub_id}_{anchor_group},{SET}.{sub_id},{anchor_group},h,{h_mean},{h_err},{h_mean*100},{h_err*100} \n")
        else:
            with open(ruta_archivo, "a") as archivo:
                archivo.write(f"{SET}.{sub_id}_{anchor_group},{SET}.{sub_id},{anchor_group},h,{h_mean},{h_err},{h_mean*100},{h_err*100} \n")

        shutil.rmtree(f"{home}/{SET}/{SET}.{sub_id}_{anchor_group}/")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--SET', type=str, help='Set according to python code')
    parser.add_argument('--sub_id', type=str, help='Code subid for identification')
    parser.add_argument('--anchor_group', type=str, help='Anchor group')
    args = parser.parse_args()

    READ_AND_CLEAN(args.SET,args.sub_id,args.anchor_group)
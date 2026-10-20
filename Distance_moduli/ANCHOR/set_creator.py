import os
import json
from astropy.cosmology import FlatwCDM
import os, sys
#import pymultinest
import re
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
from astropy.table import Column
from scipy.stats import bootstrap
from getdist import plots, MCSamples
import scipy.optimize as op
from scipy import stats
from scipy.stats import norm
from astropy.time import Time
import matplotlib.dates as mdates
from joblib import Parallel, delayed
import time

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "text.latex.preamble": r"\usepackage{amssymb}"
})


class subset_creation:
    def __init__(self,
                 input_df,
                 condition,
                 subset_name):

        self.input_df = input_df
        self.condition = condition #Debe ser una lista de Tuplas
        self.subset_name = subset_name
        self.subset = None
        self.set_selection()

    def set_selection(self):
        tmp_dataFrame = pd.read_csv(self.input_df)
        for s in range(len(self.condition)):
            if self.condition[s][1] == '==':
                tmp_dataFrame = tmp_dataFrame[tmp_dataFrame[self.condition[s][0]]==self.condition[s][2]]
            elif self.condition[s][1] == '!=':
                tmp_dataFrame = tmp_dataFrame[tmp_dataFrame[self.condition[s][0]]!=self.condition[s][2]]
            elif self.condition[s][1] == '>=':
                tmp_dataFrame = tmp_dataFrame[tmp_dataFrame[self.condition[s][0]]>=self.condition[s][2]]
            elif self.condition[s][1] == '<=':
                tmp_dataFrame = tmp_dataFrame[tmp_dataFrame[self.condition[s][0]]<=self.condition[s][2]]
            elif self.condition[s][1] == '>':
                tmp_dataFrame = tmp_dataFrame[tmp_dataFrame[self.condition[s][0]]>self.condition[s][2]]
            elif self.condition[s][1] == '<':
                tmp_dataFrame = tmp_dataFrame[tmp_dataFrame[self.condition[s][0]]<self.condition[s][2]]
        self.subset = tmp_dataFrame

        if len(self.subset) != 0:
            self.subset.to_csv(self.subset_name,index=False)

            galaxies = list(tmp_dataFrame['galaxy'].unique())


            main_folder_name = (self.subset_name).replace('.csv','')
            if not os.path.exists(main_folder_name):
                os.makedirs(main_folder_name, exist_ok=True)
                os.makedirs(f"{main_folder_name}/Bootstrap/", exist_ok=True)
                os.makedirs(f"{main_folder_name}/Nested_sampling/", exist_ok=True)
                os.makedirs(f"{main_folder_name}/Subplots/", exist_ok=True)
                os.makedirs(f"{main_folder_name}/Data/", exist_ok=True)

                for G in range(len(galaxies)):
                    os.makedirs(f"{main_folder_name}/Bootstrap/{galaxies[G]}/", exist_ok=True)
                    os.makedirs(f"{main_folder_name}/Nested_sampling/{galaxies[G]}/", exist_ok=True)
                    os.makedirs(f"{main_folder_name}/Subplots/{galaxies[G]}", exist_ok=True)


# Aqui si decidi sacar a Macri por que representa un outlier bastante marcado y aun no se ha realizado el sigma clip. Habra que ver si se integra el analisis de sigma clip en ulteriores corridas.


ANCHORS = ['LMC', 'GC', 'LMC+GC', 'N4258', 'LMC+GC+N4258', 'TM']

for Anchor in ANCHORS:
    subset_creation(input_df = 'PLRC_general_v1.2.csv',
                condition = [('metal_correction','==',False),('random_error','>=',0),('citation','!=','Macri et al.2001')
                             ,('anchor','==',Anchor)],
                subset_name = f'nZ_eR_{Anchor}.csv')
    subset_creation(input_df = 'PLRC_general_v1.2.csv',
                    condition = [('metal_correction','==',False),('total_error','>=',0),('citation','!=','Macri et al.2001')
                                ,('anchor','==',Anchor)],
                    subset_name = f'nZ_eT_{Anchor}.csv')
    subset_creation(input_df = 'PLRC_general_v1.2.csv',
                    condition = [('metal_correction','==',True),('random_error','>=',0),('citation','!=','Macri et al.2001')
                                ,('anchor','==',Anchor)],
                    subset_name = f'Z_eR_{Anchor}.csv')
    subset_creation(input_df = 'PLRC_general_v1.2.csv',
                    condition = [('metal_correction','==',True),('total_error','>=',0),('citation','!=','Macri et al.2001')
                                ,('anchor','==',Anchor)],
                    subset_name = f'Z_eT_{Anchor}.csv')


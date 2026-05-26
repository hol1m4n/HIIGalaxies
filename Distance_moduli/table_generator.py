from __future__ import annotations
import os
import json
from astropy.cosmology import FlatwCDM


import os, sys
print(sys.executable)
print(os.environ.get("LD_LIBRARY_PATH","<empty>"))
import pymultinest
print("OK")

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
from astropy.table import Column
from scipy.stats import bootstrap



import pymultinest
import corner
from getdist import plots, MCSamples
import scipy.optimize as op
from scipy import stats
from scipy.stats import norm
from astropy.time import Time
import matplotlib.dates as mdates
from joblib import Parallel, delayed

###### Funciones


def weighted_average(modulus,error):
    modulus,error = np.array(modulus),np.array(error)
    err_sq_inv = 1 / ((error)**2)
    num = np.sum(err_sq_inv * modulus)
    dem = np.sum(err_sq_inv)

    return num/dem

def weighted_error(error):
    error = np.array(error)
    err_sq_inv = 1 / ((error)**2)
    dem = np.sum(err_sq_inv)

    return 1 / np.sqrt(dem)

def bootstrap_error2(modulus,error,N=200000,group = '',Galaxy=''):

    PATH = f'Bootstrap_resampling/{group}/{Galaxy}.txt'

    if os.path.exists(PATH) == True:
        bootstrap_statistics = np.loadtxt(PATH)
        print(f'Reading bootstrap for {Galaxy}\n')
    if os.path.exists(PATH) == False:
        os.makedirs(PATH.replace(f'{Galaxy}.txt',''), exist_ok=True)
        print(f'Running bootstrap for {Galaxy}\n')
        modulus,error = np.array(modulus),np.array(error)
        size = len(modulus)

        n_bootstraps = N  # Number of bootstrap iterations
        bootstrap_statistics = []

        for i in range(n_bootstraps):
            # Create a resample with replacement, same size as original data
            bootstrap_sample = np.random.choice(size, size=size, replace=True)

            modulus_sample = modulus[bootstrap_sample]
            error_sample = error[bootstrap_sample]

            sample_mean = weighted_average(modulus_sample,error_sample)
            bootstrap_statistics.append(sample_mean)

        bootstrap_statistics = np.array(bootstrap_statistics)

        np.savetxt(fname = PATH, X = bootstrap_statistics)

    N_dist = len(bootstrap_statistics)

    br_mean = np.mean(bootstrap_statistics)

    quad_br = (bootstrap_statistics-br_mean)**2
    
    lower_bound = np.percentile(bootstrap_statistics, 15.87)
    upper_bound = np.percentile(bootstrap_statistics, 84.13)

    return [np.sqrt(((1)/(N_dist-1)) * np.sum(quad_br)),
           [br_mean - lower_bound, upper_bound - br_mean],
            bootstrap_statistics]

    
def cochran_error(modulus,error):
    Mu_i,Er_i = np.array(modulus),np.array(error)
    n = len(Mu_i)
    w_i = 1 / ((Er_i)**2)

    Mu_w = weighted_average(Mu_i,Er_i)
    w_mean = np.mean(w_i)

    s1 = n / ((n-1) * (np.sum(w_i))**2)
    s2 = (w_i * Mu_i  - (w_mean * Mu_w))**2
    s3 = (w_i - w_mean) * ((w_i*Mu_i) - (w_mean*Mu_w)) 
    s4 = (w_i - w_mean)**2

    return np.sqrt(s1 * (np.sum(s2) - (2 * Mu_w * np.sum(s3)) +    (Mu_w**2 * np.sum(s4)) ) )


def nestedsampling_error(modulus,error,steps=10000, Name='', group = ''):

    def prior_transform(x):
        mu = 20 * x + 20
        return mu
    
    def lnlike(theta, m, merr):
        mu_pi = theta
        R = (mu_pi - m)
        W = 1.0/(merr**2)
        xsq = np.sum(R**2 * W)
        L = -0.5*xsq
        return L
    
    def loglike_for_mnest(theta):
        return lnlike(theta, modulus, error)
    
    outdir = os.path.join('Nested_sampling', group)
    os.makedirs(outdir, exist_ok=True)
    prefix = os.path.join(outdir, Name)

    n_dims = 1
    result = pymultinest.solve(
        LogLikelihood=loglike_for_mnest,
        Prior=prior_transform,
        n_dims=n_dims,
        outputfiles_basename=prefix,
        evidence_tolerance=0.5,
        n_live_points=steps,
        multimodal=True,
        verbose=False,
    )

    samples = result["samples"] 
    T_sample = samples.T


    # calcula loglike para cada muestra
    logL = np.array([loglike_for_mnest(s) for s in samples])

    # punto MAP (aprox): el de mayor logL
    #theta_map = samples[np.argmax(logL)]

    # chi^2 en MAP
    chi2_map = -2.0*np.max(logL)

    N = len(modulus)   # número de datos
    k = n_dims         # parámetros del modelo
    nu = N - k
    chi2_red = chi2_map / nu

    sigma2_corr = T_sample.std() * chi2_red

    return [T_sample.std(),sigma2_corr]



# Bootstrap mayor a 3 DMs <-

# Revisar si el archivo existe (o no) en carpeta. Es mejor si no existe y quitar el IF

# Se abre el fits
# Se crea el encabezado de la tabla

# Se aplican las funciones a cada una de las galaxias (paralelizar)

# Definir variables del Metodo (CEFEIDAS o TRGB)

# Aplicar condicionales (Utilizar la estructura que use para lo de FADO)

# Crear los arrays y variables necesarias



class DM_set_select:
    def __init__(self,data,selection):
        self.data = data
        self.select = selection #Debe ser una lista de Tuplas
        self.set = None
        self.set_selection()

    def set_selection(self):
        tmp_dataFrame = self.data
        for s in range(len(self.select)):
            if self.select[s][1] == '==':
                tmp_dataFrame = tmp_dataFrame[tmp_dataFrame[self.select[s][0]]==self.select[s][2]]
            elif self.select[s][1] == '!=':
                tmp_dataFrame = tmp_dataFrame[tmp_dataFrame[self.select[s][0]]!=self.select[s][2]]
            elif self.select[s][1] == '>=':
                tmp_dataFrame = tmp_dataFrame[tmp_dataFrame[self.select[s][0]]>=self.select[s][2]]
            elif self.select[s][1] == '<=':
                tmp_dataFrame = tmp_dataFrame[tmp_dataFrame[self.select[s][0]]<=self.select[s][2]]
            elif self.select[s][1] == '>':
                tmp_dataFrame = tmp_dataFrame[tmp_dataFrame[self.select[s][0]]>self.select[s][2]]
            elif self.select[s][1] == '<':
                tmp_dataFrame = tmp_dataFrame[tmp_dataFrame[self.select[s][0]]<self.select[s][2]]
        self.set = tmp_dataFrame



def MASTER_STATS(csv_name,Metodo,condition_x_group):
    csv_PATH = os.path.expanduser(f'~/HIIGalaxies/Distance_moduli/Results_tables/{csv_name}.csv')
    if not os.path.exists(csv_PATH):
        master_file = os.path.expanduser('~/HIIGalaxies/Distance_moduli/modulus_tracker_TABLES.fits')
        stats_comp = [] # Encabezado, valido para cualquier grupo
        stats_comp =['Galaxia','N_dat','mu_w','sigma_w','sigma_br','sigma_C','sigma_cl+','sigma_cl-','sigma_L','sigma_Lcorr']

        def parallel_statistics(i,FITS_name):
                FITS_file = fits.open(FITS_name)
                gal_host = i
                T = Table.read(FITS_file[gal_host])
                if (FITS_file[gal_host].header['METHOD'] == Metodo) and (len(T)>0):

                    SeT = DM_set_select(T,condition_x_group).set
                    Group_modulus = np.array(SeT['mu_0'])
                    if any('e_R' in tupla for tupla in condition_x_group):
                        Group_error = np.array(SeT['e_R'])
                    if any('e_T' in tupla for tupla in condition_x_group):
                        Group_error = np.array(SeT['e_T'])                    
                    Galaxia = FITS_file[gal_host].header['EXTNAME']
                    N_dat = len(SeT)

                    print(f"\n")
                    print(f"Working for for {Galaxia} -> {FITS_file[gal_host].header['METHOD']} \n")
                    print(f"Counter: {i}\n")
                    print(f"\n")


                    if N_dat == 0:
                        print(f"No data for {Galaxia}\n")
                    else:
                        mu_w = weighted_average(Group_modulus,Group_error)
                        sigma_w = weighted_error(Group_error)
                        sigma_C = cochran_error(Group_modulus,Group_error)
                        if N_dat > 1:
                            sigma_br,sigma_cl,Br_stats = bootstrap_error2(modulus = Group_modulus
                                                                        ,error = Group_error,N=200000,group=csv_name,Galaxy=Galaxia)
                            
                            sigma_br = sigma_br
                            sigma_cl[0] = sigma_cl[0] # Ya no me acuerdo por que redefini estas variables
                            sigma_cl[1] = sigma_cl[1]
                            Br_stats = Br_stats


                            sigma_L, sigma_Lcorr = nestedsampling_error(Group_modulus,
                                                                        Group_error,Name=Galaxia,group = csv_name, steps=100)
                            stats_comp.append([Galaxia,N_dat,mu_w,sigma_w,sigma_br,sigma_C,sigma_cl[0],sigma_cl[1],sigma_L,sigma_Lcorr])
                        else:
                            stats_comp.append([Galaxia,N_dat,mu_w,sigma_w,'--','--','--','--','--','--'])
                    
                FITS_file.close()

                #return 0
        # Aplicar el bucle paralelizado

        #tmp = Parallel(n_jobs=1)(delayed(parallel_statistics)(parent_galaxy,master_file) for parent_galaxy in range(1,45))                   

        for parent_galaxy in range(1,46):
            parallel_statistics(parent_galaxy,master_file)


        

        # Salvando el DF final

        DF = pd.DataFrame(data = stats_comp[10:],columns = stats_comp[0:10])
        DF.to_csv(csv_PATH)#,index=False)



MASTER_STATS('sm_r',
             'CEPHEIDS',
             [('Zcorr','==',False),
                 ('e_R','>=',0),('Author','==','Valencia et al.2024')])


MASTER_STATS('cm_r',
             'CEPHEIDS',
             [('Zcorr','==',True),
                 ('e_R','>=',0),('Author','==','Valencia et al.2024')])


MASTER_STATS('sm_t',
             'CEPHEIDS',
             [('Zcorr','==',False),
                 ('e_T','>=',0),('Author','==','Valencia et al.2024')])


MASTER_STATS('cm_t',
             'CEPHEIDS',
             [('Zcorr','==',True),
                 ('e_T','>=',0),('Author','==','Valencia et al.2024')])


MASTER_STATS('t_r',
             'TRGB',
             [('e_R','>=',0),('Author','==','Valencia et al.2024')])












'''










if os.path.exists('sm_r.csv'):
    DF = pd.read_csv('sm_r.csv')
else:
    S = fits.open('modulus_tracker_TABLES.fits')
    sm_r = []
    sm_r =['Galaxia','N_dat','mu_w','sigma_w','sigma_br','sigma_C','sigma_cl+','sigma_cl-','sigma_L','sigma_Lcorr']

    for i in range(1,len(S)):

        gal_host = i

        T = Table.read(S[gal_host])

        if (S[gal_host].header['METHOD'] == 'CEPHEIDS') and (len(T)>0):

            T = T[(T['Zcorr']==False) & (T['e_R']>= 0)]

            G1_m = np.array(T['mu_0'])
            G1_e = np.array(T['e_R'])
            Galaxia = S[gal_host].header['EXTNAME']
            N_dat = len(T)
            
            
            if N_dat == 0:
                print(f"No data for {Galaxia}")
            if N_dat == 1:
                #print(N_dat)
                mu_w = weighted_average(G1_m,G1_e)
                sigma_w = weighted_error(G1_e)
                sm_r.append([Galaxia,N_dat,mu_w,sigma_w,'--','--','--','--','--','--'])        
            if N_dat > 1:
                #print(N_dat)
                mu_w = weighted_average(G1_m,G1_e)
                sigma_w = weighted_error(G1_e)
                sigma_br,sigma_cl,Br_stats = bootstrap_error2(modulus = G1_m,error = G1_e,N=200000,group='sm_r',Galaxy=Galaxia)
                sigma_br = sigma_br 
                sigma_cl[0] = sigma_cl[0]
                sigma_cl[1] = sigma_cl[1]
                sigma_C = cochran_error(G1_m,G1_e)
                sigma_L, sigma_Lcorr = nestedsampling_error(G1_m,G1_e,Name=Galaxia,group = 'sm_r')
                sm_r.append([Galaxia,N_dat,mu_w,sigma_w,sigma_br,sigma_C,sigma_cl[0],sigma_cl[1],sigma_L,sigma_Lcorr])

    S.close()

    DF = pd.DataFrame(data = sm_r[10:],columns = sm_r[0:10])
    DF.to_csv('sm_r.csv')


if os.path.exists('cm_r.csv'):
    DF = pd.read_csv('cm_r.csv')
else:
    S = fits.open('modulus_tracker_TABLES.fits')
    cm_r = []
    cm_r =['Galaxia','N_dat','mu_w','sigma_w','sigma_br','sigma_C','sigma_cl+','sigma_cl-','sigma_L','sigma_Lcorr']

    for i in range(1,len(S)):

        gal_host = i

        T = Table.read(S[gal_host])

        if (S[gal_host].header['METHOD'] == 'CEPHEIDS') and (len(T)>0):

            T = T[(T['Zcorr']==True) & (T['e_R']>= 0)]

            G1_m = np.array(T['mu_0'])
            G1_e = np.array(T['e_R'])
            Galaxia = S[gal_host].header['EXTNAME']
            N_dat = len(T)
            
            
            if N_dat == 0:
                print(f"No data for {Galaxia}")
            if N_dat == 1:
                #print(N_dat)
                mu_w = weighted_average(G1_m,G1_e)
                sigma_w = weighted_error(G1_e)
                cm_r.append([Galaxia,N_dat,mu_w,sigma_w,'--','--','--','--','--','--'])        
            if N_dat > 1:
                #print(N_dat)
                mu_w = weighted_average(G1_m,G1_e)
                sigma_w = weighted_error(G1_e)
                sigma_br,sigma_cl,Br_stats = bootstrap_error2(modulus = G1_m,error = G1_e,N=200000,group='cm_r',Galaxy=Galaxia)
                sigma_br = sigma_br 
                sigma_cl[0] = sigma_cl[0]
                sigma_cl[1] = sigma_cl[1]
                sigma_C = cochran_error(G1_m,G1_e)
                sigma_L, sigma_Lcorr = nestedsampling_error(G1_m,G1_e,Name=Galaxia,group = 'cm_r')
                cm_r.append([Galaxia,N_dat,mu_w,sigma_w,sigma_br,sigma_C,sigma_cl[0],sigma_cl[1],sigma_L,sigma_Lcorr])

    S.close()

    DF = pd.DataFrame(data = cm_r[10:],columns = cm_r[0:10])
    DF.to_csv('cm_r.csv')


if os.path.exists('sm_t.csv'):
    DF = pd.read_csv('sm_t.csv')
else:
    S = fits.open('modulus_tracker_TABLES.fits')
    sm_t = []
    sm_t =['Galaxia','N_dat','mu_w','sigma_w','sigma_br','sigma_C','sigma_cl+','sigma_cl-','sigma_L','sigma_Lcorr']

    for i in range(1,len(S)):

        gal_host = i

        T = Table.read(S[gal_host])

        if (S[gal_host].header['METHOD'] == 'CEPHEIDS') and (len(T)>0):

            T = T[(T['Zcorr']==False)  &  (T['e_T']>= 0)]

            G1_m = np.array(T['mu_0'])
            G1_e = np.array(T['e_T'])
            Galaxia = S[gal_host].header['EXTNAME']
            N_dat = len(T)
            
            
            if N_dat == 0:
                print(f"No data for {Galaxia}")
            if N_dat == 1:
                #print(N_dat)
                mu_w = weighted_average(G1_m,G1_e)
                sigma_w = weighted_error(G1_e)
                sm_t.append([Galaxia,N_dat,mu_w,sigma_w,'--','--','--','--','--','--'])        
            if N_dat > 1:
                #print(N_dat)
                mu_w = weighted_average(G1_m,G1_e)
                sigma_w = weighted_error(G1_e)
                sigma_br,sigma_cl,Br_stats = bootstrap_error2(modulus = G1_m,error = G1_e,N=200000,group='sm_t',Galaxy=Galaxia)
                sigma_br = sigma_br 
                sigma_cl[0] = sigma_cl[0]
                sigma_cl[1] = sigma_cl[1]
                sigma_C = cochran_error(G1_m,G1_e)
                sigma_L, sigma_Lcorr = nestedsampling_error(G1_m,G1_e,Name=Galaxia,group = 'sm_t')
                sm_t.append([Galaxia,N_dat,mu_w,sigma_w,sigma_br,sigma_C,sigma_cl[0],sigma_cl[1],sigma_L,sigma_Lcorr])

    S.close()

    DF = pd.DataFrame(data = sm_t[10:],columns = sm_t[0:10])
    DF.to_csv('sm_t.csv')


if os.path.exists('cm_t.csv'):
    DF = pd.read_csv('cm_t.csv')
else:
    S = fits.open('modulus_tracker_TABLES.fits')
    cm_t = []
    cm_t =['Galaxia','N_dat','mu_w','sigma_w','sigma_br','sigma_C','sigma_cl+','sigma_cl-','sigma_L','sigma_Lcorr']

    for i in range(1,len(S)):

        gal_host = i

        T = Table.read(S[gal_host])

        if (S[gal_host].header['METHOD'] == 'CEPHEIDS') and (len(T)>0):

            T = T[(T['Zcorr']==True)  &  (T['e_T']>= 0)]

            G1_m = np.array(T['mu_0'])
            G1_e = np.array(T['e_T'])
            Galaxia = S[gal_host].header['EXTNAME']
            N_dat = len(T)
            
            
            if N_dat == 0:
                print(f"No data for {Galaxia}")
            if N_dat == 1:
                mu_w = weighted_average(G1_m,G1_e)
                sigma_w = weighted_error(G1_e)
                cm_t.append([Galaxia,N_dat,mu_w,sigma_w,'--','--','--','--','--','--'])        
            if N_dat > 1:
                mu_w = weighted_average(G1_m,G1_e)
                sigma_w = weighted_error(G1_e)
                sigma_br,sigma_cl,Br_stats = bootstrap_error2(modulus = G1_m,error = G1_e,N=200000,group='cm_t',Galaxy=Galaxia)
                sigma_br = sigma_br 
                sigma_cl[0] = sigma_cl[0]
                sigma_cl[1] = sigma_cl[1]
                sigma_C = cochran_error(G1_m,G1_e)
                sigma_L, sigma_Lcorr = nestedsampling_error(G1_m,G1_e,Name=Galaxia,group = 'cm_t')
                cm_t.append([Galaxia,N_dat,mu_w,sigma_w,sigma_br,sigma_C,sigma_cl[0],sigma_cl[1],sigma_L,sigma_Lcorr])

    S.close()

    DF = pd.DataFrame(data = cm_t[10:],columns = cm_t[0:10])
    DF.to_csv('cm_t.csv')


if os.path.exists('t_r.csv'):
    DF = pd.read_csv('t_r.csv')
else:
    S = fits.open('modulus_tracker_TABLES.fits')
    t_r = []
    t_r =['Galaxia','N_dat','mu_w','sigma_w','sigma_br','sigma_C','sigma_cl+','sigma_cl-','sigma_L','sigma_Lcorr']

    for i in range(1,len(S)):

        gal_host = i

        T = Table.read(S[gal_host])

        if (S[gal_host].header['METHOD'] == 'TRGB') and (len(T)>0):
            G1_m = np.array(T['mu_0'])
            G1_e = np.array(T['e_R'])
            Galaxia = S[gal_host].header['EXTNAME']
            N_dat = len(T)

            if N_dat == 0:
                print(f"No data for {Galaxia}")
            if N_dat == 1:
                mu_w = weighted_average(G1_m,G1_e)
                sigma_w = weighted_error(G1_e)
                t_r.append([Galaxia,N_dat,mu_w,sigma_w,'--','--','--','--','--','--'])
                print("\n")
                print(f"Done for {Galaxia}, {N_dat}\n")      
            if N_dat > 1:
                mu_w = weighted_average(G1_m,G1_e)
                sigma_w = weighted_error(G1_e)
                sigma_br,sigma_cl,Br_stats = bootstrap_error2(modulus = G1_m,error = G1_e,N=200000,group='t_r',Galaxy=Galaxia)
                sigma_br = sigma_br 
                sigma_cl[0] = sigma_cl[0]
                sigma_cl[1] = sigma_cl[1]
                sigma_C = cochran_error(G1_m,G1_e)
                sigma_L, sigma_Lcorr = nestedsampling_error(G1_m,G1_e,Name=Galaxia,group = 't_r')
                t_r.append([Galaxia,N_dat,mu_w,sigma_w,sigma_br,sigma_C,sigma_cl[0],sigma_cl[1],sigma_L,sigma_Lcorr])
                print("\n")
                print(f"Done for {Galaxia} , {N_dat}\n")

    S.close()

    DF = pd.DataFrame(data = t_r[10:],columns = t_r[0:10])
    DF.to_csv('t_r.csv')

    

    '''
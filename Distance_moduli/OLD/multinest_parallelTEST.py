from __future__ import annotations
import os, sys
print(sys.executable)
print(os.environ.get("LD_LIBRARY_PATH","<empty>"))
import pymultinest
print("OK")

import re
import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
import pymultinest
import time
from mpi4py import MPI


start_time = time.perf_counter()


# Configuramos MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


for x in range(5):
    print('SES')

'''

# Solo el proceso 0 imprime el estado inicial
if rank == 0:
    print(f"Python: {sys.executable}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH','<empty>')}")
    print("Iniciando ejecución paralela... OK")

def nestedsampling_error(modulus, error, steps=10000, Name='', group=''):
    def prior_transform(x):
        return 20 * x + 20
    
    def loglike_for_mnest(theta):
        mu_pi = theta[0]
        R = (mu_pi - modulus)
        W = 1.0/(error**2)
        return -0.5 * np.sum(R**2 * W)
    
    outdir = os.path.join('NS', group)
    if rank == 0: # Solo el proceso principal crea carpetas
        os.makedirs(outdir, exist_ok=True)
    
    # IMPORTANTE: Esperar a que el rank 0 cree la carpeta antes de que otros escriban
    comm.Barrier() 
    
    prefix = os.path.join(outdir, Name)

    # PyMultiNest.solve detecta MPI automáticamente si mpi4py está importado
    result = pymultinest.solve(
        LogLikelihood=loglike_for_mnest,
        Prior=prior_transform,
        n_dims=1,
        outputfiles_basename=prefix,
        evidence_tolerance=0.5,
        n_live_points=steps,
        multimodal=True,
        verbose=False, # Evita colisiones de impresión en paralelo
    )
    
    # Procesamiento de resultados solo en el rank 0
    if rank == 0:
        samples = result["samples"]
        T_sample = samples.T
        logL = np.array([loglike_for_mnest(s) for s in samples])
        chi2_map = -2.0 * np.max(logL)
        nu = len(modulus) - 1
        chi2_red = chi2_map / nu
        sigma2_corr = T_sample.std() * chi2_red
        return [T_sample.std(), sigma2_corr]
    else:
        return None

# --- INICIO DEL FLUJO ---
if rank == 0:
    start_time = time.perf_counter()
    S = fits.open('modulus_tracker_TABLES.fits')
    gal_host = 38
    T = Table.read(S[gal_host])
    
    if (S[gal_host].header['METHOD'] == 'TRGB') and (len(T) > 0):
        G1_m = np.array(T['mu_0'])
        G1_e = np.array(T['e_R'])
        Galaxia = S[gal_host].header['EXTNAME']
        
        if len(T) > 1:
            res = nestedsampling_error(G1_m, G1_e, Name=Galaxia, group='P_T')
            print(f"\nDone for {Galaxia}")
            print(f"Resultados: {res}")

    S.close()
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
else:
    # Los procesos esclavos (rank > 0) entran aquí y esperan órdenes de solve()
    # Deben recibir los mismos argumentos para la función de likelihood
    # Pero como 'modulus' y 'error' son necesarios, deben estar disponibles.
    
    # Para scripts simples, puedes leer el FITS en todos los ranks para que 
    # todos tengan las variables G1_m y G1_e, o usar comm.bcast.
    
    S = fits.open('modulus_tracker_TABLES.fits')
    T = Table.read(S[38])
    G1_m = np.array(T['mu_0'])
    G1_e = np.array(T['e_R'])
    Galaxia = S[38].header['EXTNAME']
    
    nestedsampling_error(G1_m, G1_e, Name=Galaxia, group='P_T')
    S.close()


'''




























'''
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
    
    outdir = os.path.join('NS', group)
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



S = fits.open('modulus_tracker_TABLES.fits')

gal_host = 38

T = Table.read(S[gal_host])

t_r = []

if (S[gal_host].header['METHOD'] == 'TRGB') and (len(T)>0):
    G1_m = np.array(T['mu_0'])
    G1_e = np.array(T['e_R'])
    Galaxia = S[gal_host].header['EXTNAME']
    N_dat = len(T)
      
    if N_dat > 1:
        sigma_L, sigma_Lcorr = nestedsampling_error(G1_m,G1_e,Name=Galaxia,group = 'P_T') # Parallel test
        t_r.append([Galaxia,N_dat,sigma_L,sigma_Lcorr])
        print("\n")
        print(f"Done for {Galaxia}\n")

S.close()


'''

'''
if rank == 0:
    start_time = time.perf_counter()
    # Solo el proceso 0 lee el archivo FITS
    S = fits.open('modulus_tracker_TABLES.fits')
    # ... (resto de tu lógica de preparación de datos) ...
    gal_host = 38

    T = Table.read(S[gal_host])

    t_r = []

    if (S[gal_host].header['METHOD'] == 'TRGB') and (len(T)>0):
        G1_m = np.array(T['mu_0'])
        G1_e = np.array(T['e_R'])
        Galaxia = S[gal_host].header['EXTNAME']
        N_dat = len(T)
        
        if N_dat > 1:
            sigma_L, sigma_Lcorr = nestedsampling_error(G1_m,G1_e,Name=Galaxia,group = 'P_T') # Parallel test
            t_r.append([Galaxia,N_dat,sigma_L,sigma_Lcorr])
            print("\n")
            print(f"Done for {Galaxia}\n")

    S.close()

else:
    G1_m = None
    G1_e = None
    Galaxia = None


if rank == 0:
    if N_dat > 1:
        sigma_L, sigma_Lcorr = nestedsampling_error(G1_m, G1_e, Name=Galaxia, group='P_T')
        print(f"Done for {Galaxia}")
'''


end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Code execution time: {elapsed_time:.6f} seconds, {elapsed_time/60:.6f} minutes")

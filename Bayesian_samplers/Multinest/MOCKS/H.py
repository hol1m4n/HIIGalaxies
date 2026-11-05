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

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

ENV_MOCK_VAR = 'H'


LENS_GH_ORDER = 20
LENS_GH_X, LENS_GH_W = np.polynomial.hermite.hermgauss(LENS_GH_ORDER)
LENS_DM_COEFF = 2.5/np.log(10.0)







class Lsig_Ho_sampler:
    def __init__(self,
                 data_frame = None,
                 distance_estimator_set = None,
                 estimator_error_kind = None,
                 main_title = None,
                 folder_name = None,
                 analysis_mode = None,
                 id_prefix = None,
                 sub_id_code = None):
        
        self.data_frame = data_frame
        self.distance_estimator_set = distance_estimator_set
        self.estimator_error_kind = estimator_error_kind
        self.main_title = main_title
        self.folder_name = folder_name
        self.analysis_mode = analysis_mode
        self.id_prefix = id_prefix
        self.outdir = None
        self.prefix = None


        self.sub_id_code = sub_id_code
        self.code_id = f'{ENV_MOCK_VAR}.{self.sub_id_code}'

        self.dir_results_creation()
        self.main_sampler()
        self.plots_maker()

    def dir_results_creation(self):
        tmp_set_code_ids = ENV_MOCK_VAR
        os.makedirs(tmp_set_code_ids, exist_ok=True)
        self.outdir = os.path.join(os.path.expanduser(f'~/HIIGalaxies/Bayesian_samplers/Multinest/MOCKS/{tmp_set_code_ids}'), f"{self.folder_name}")
        os.makedirs(self.outdir, exist_ok=True)
        self.prefix = os.path.join(self.outdir, f"{self.id_prefix}_")



    def group_reader(self, DF=None, group=None, error_kind = None, MOCK = ''):

        def simulate_GEHR_host(
            mu,
            alpha,
            beta,
            sigma_int,
            logsigma_reference,
            logsigma_error_real,
            logflux_error_real,
            N=500000,
            sigma_add_error=0.06,
        ):

            # -------------------------------
            # 1. Sigma latente
            # -------------------------------

            logsigma_true = np.random.choice(
                logsigma_reference,
                size=N,
                replace=True
            ) + np.random.uniform(0,sigma_add_error,N)

            sigma_true = 10**logsigma_true 


            # -------------------------------
            # 2. Scatter intrínseco
            # -------------------------------

            eps_int = np.random.normal(
                0.0,
                sigma_int,
                size=N
            )


            # -------------------------------
            # 3. L(Hbeta)
            # -------------------------------

            logL_true = (
                alpha
                + beta * logsigma_true
                + eps_int
            )


            # -------------------------------
            # 4. F(Hbeta)
            # log F = log L - 0.4 mu - 40.079
            # -------------------------------

            logF_true = (
                logL_true
                - 0.4 * mu
                - 40.079
            )

            flux_true = 10**logF_true

            # -------------------------------
            # 5. Log Sigma error
            # e~N(mean,stdev)
            # -------------------------------

            sigma_errors = np.asarray(logsigma_error_real, dtype=float)

            # Eliminar errores nulos, negativos, NaN e infinitos
            sigma_errors = sigma_errors[
                np.isfinite(sigma_errors) & (sigma_errors > 0)
            ]

            if len(sigma_errors) == 0:
                raise ValueError("No hay errores válidos de log_sigma")

            # Eliminar valores extremos (percentiles 1 y 99)
            if len(sigma_errors) >= 20:
                p1, p99 = np.percentile(sigma_errors, [1, 99])
                sigma_errors = sigma_errors[
                    (sigma_errors >= p1) & (sigma_errors <= p99)
                ]

            logsigma_error = np.random.choice(
                sigma_errors, size=N, replace=True
            )




            # -------------------------------
            # 6. Log Flux error
            # e~N(mean,stdev)
            # -------------------------------

            flux_errors = np.asarray(logflux_error_real, dtype=float)

            flux_errors = flux_errors[
                np.isfinite(flux_errors) & (flux_errors > 0)
            ]

            if len(flux_errors) == 0:
                raise ValueError("No hay errores válidos de logflux")

            if len(flux_errors) >= 20:
                p1, p99 = np.percentile(flux_errors, [1, 99])
                flux_errors = flux_errors[
                    (flux_errors >= p1) & (flux_errors <= p99)
                ]

            logflux_error = np.random.choice(
                flux_errors, size=N, replace=True
            )


            Mock_data = pd.DataFrame({
                "mu": mu,
                "sigma_kms": sigma_true,
                "LogSigma": logsigma_true,
                "LogSigma_e": logsigma_error,
                "LogL_Hb": logL_true,
                "LogFlux_Hb": logF_true,
                "LogFlux_e": logflux_error,
                "Flux_Hb": flux_true
            })

            return Mock_data.sample(n=1)

        if DF is not None and group is not None and error_kind is not None:
            DF_pd = DF.to_pandas()
            df_group = pd.read_csv(group)
            galaxies_in_group = df_group['galaxy'].unique()
            filtro = DF_pd[DF_pd['GEHR_id'].isin(galaxies_in_group) | (DF_pd['origin_id'] > 0.0)].copy()
            filtro = filtro.merge(
                df_group[['galaxy', 'mu_W', f'{error_kind}']], 
                left_on='GEHR_id', 
                right_on='galaxy', 
                how='left'
            )
            coincide = filtro['mu_W'].notna()
            filtro.loc[coincide, 'z_or_mu'] = filtro.loc[coincide, 'mu_W']
            filtro.loc[coincide, 'e_z_or_e_mu'] = filtro.loc[coincide, f'{error_kind}']
            filtro = filtro.drop(columns=['galaxy', 'mu_W', f'{error_kind}'])
            
            filtro['Mock'] = False



            galaxies_in_DF = filtro[filtro['origin_id']==0.0]['GEHR_id'].unique()
            match = np.isin(galaxies_in_group,galaxies_in_DF)
            print('Galaxias en DF original: ', galaxies_in_DF)
            print('Galaxias en recal file: ', galaxies_in_group)
            print('Match: ', galaxies_in_group[match])
            print('Dismatch: ', galaxies_in_group[~match])

            reference = DF_pd[
                DF_pd['origin_id'] == 0.0
            ][[
                'log_sigma',
                'e_log_sigma',
                'e_log_f_Hbeta'
            ]].copy()


            newGEHR_per_galaxy = {
                "M66": 3,
                "M96": 1,
                "NGC1073": 4,
                "NGC2500": 4,
                "NGC2541": 1,
                "NGC3198": 2,
                "NGC3370": 3,
                "NGC4214": 2,
                "NGC4258": 1,
                "NGC4414": 3,
                "NGC4496A": 3,
                "NGC4535": 2,
                "NGC4536": 2,
                "NGC4725": 1,
                "NGC5204": 4,
                "NGC5584": 3,
                "NGC7331": 4,
                "UGC08091": 2,
                "UGC09128": 1

                # Agregar más galaxias...
            }

            selected_hosts = [
                "IC0010",
                "M33",
                "M81",
                "M101",
                "MRK116",
                "NGC2366",
                "NGC2403",
                "NGC4258",
                "NGC4395",
                "NGC6822",
                "M96",
                "NGC3370",
                "M66",
                "NGC5584",
                # Agregar más galaxias...
            ]


            if MOCK == 'A':
                print('10 nuevas GEHR en galaxias presentes')
                newGEHR_oldGAL = np.random.choice(galaxies_in_group[match], size=10, replace=True)

                for G in newGEHR_oldGAL:
                    new_ix = filtro['index'].iloc[-1] + 1
                    mu = df_group[df_group['galaxy']== G]['mu_W'].iloc[0]
                    mu_e = df_group[df_group['galaxy']== G]['sigma_W'].iloc[0]

                    
                    mock_GEHR = simulate_GEHR_host(
                        mu=mu,
                        alpha=33.25,
                        beta=5.02,
                        sigma_int=0.26,
                        logsigma_reference = reference["log_sigma"],
                        logsigma_error_real = reference["e_log_sigma"],
                        logflux_error_real = reference["e_log_f_Hbeta"],
                        )       

                    new_entry = pd.DataFrame([{'index': new_ix, 
                                                'z_or_mu': mu,
                                                'e_z_or_e_mu': mu_e,
                                                'log_sigma': mock_GEHR['LogSigma'].iloc[0],
                                                'e_log_sigma': mock_GEHR['LogSigma_e'].iloc[0],
                                                'log_f_Hbeta': mock_GEHR['LogFlux_Hb'].iloc[0],
                                                'e_log_f_Hbeta': mock_GEHR['LogFlux_e'].iloc[0],
                                                'origin_id': 0.0,
                                                'GEHR_id': G,
                                                'Mock': True,}])

                    filtro = pd.concat([filtro, new_entry], ignore_index=True)

            if MOCK == 'B':
                print('10 nuevas GEHR en galaxias nuevas')
                newGEHR_newGAL = np.random.choice(galaxies_in_group[~match], size=10, replace=True)

                for G in newGEHR_newGAL:
                    new_ix = filtro['index'].iloc[-1] + 1
                    mu = df_group[df_group['galaxy']== G]['mu_W'].iloc[0]
                    mu_e = df_group[df_group['galaxy']== G]['sigma_W'].iloc[0]

                    
                    mock_GEHR = simulate_GEHR_host(
                        mu=mu,
                        alpha=33.25,
                        beta=5.02,
                        sigma_int=0.26,
                        logsigma_reference = reference["log_sigma"],
                        logsigma_error_real = reference["e_log_sigma"],
                        logflux_error_real = reference["e_log_f_Hbeta"],
                        )       

                    new_entry = pd.DataFrame([{'index': new_ix, 
                                                'z_or_mu': mu,
                                                'e_z_or_e_mu': mu_e,
                                                'log_sigma': mock_GEHR['LogSigma'].iloc[0],
                                                'e_log_sigma': mock_GEHR['LogSigma_e'].iloc[0],
                                                'log_f_Hbeta': mock_GEHR['LogFlux_Hb'].iloc[0],
                                                'e_log_f_Hbeta': mock_GEHR['LogFlux_e'].iloc[0],
                                                'origin_id': 0.0,
                                                'GEHR_id': G,
                                                'Mock': True,}])

                    filtro = pd.concat([filtro, new_entry], ignore_index=True)


            if MOCK == 'C':
                print('30 nuevas GEHR en galaxias presentes')
                newGEHR_oldGAL = np.random.choice(galaxies_in_group[match], size=30, replace=True)

                for G in newGEHR_oldGAL:
                    new_ix = filtro['index'].iloc[-1] + 1
                    mu = df_group[df_group['galaxy']== G]['mu_W'].iloc[0]
                    mu_e = df_group[df_group['galaxy']== G]['sigma_W'].iloc[0]

                    
                    mock_GEHR = simulate_GEHR_host(
                        mu=mu,
                        alpha=33.25,
                        beta=5.02,
                        sigma_int=0.26,
                        logsigma_reference = reference["log_sigma"],
                        logsigma_error_real = reference["e_log_sigma"],
                        logflux_error_real = reference["e_log_f_Hbeta"],
                        )       

                    new_entry = pd.DataFrame([{'index': new_ix, 
                                                'z_or_mu': mu,
                                                'e_z_or_e_mu': mu_e,
                                                'log_sigma': mock_GEHR['LogSigma'].iloc[0],
                                                'e_log_sigma': mock_GEHR['LogSigma_e'].iloc[0],
                                                'log_f_Hbeta': mock_GEHR['LogFlux_Hb'].iloc[0],
                                                'e_log_f_Hbeta': mock_GEHR['LogFlux_e'].iloc[0],
                                                'origin_id': 0.0,
                                                'GEHR_id': G,
                                                'Mock': True,}])

                    filtro = pd.concat([filtro, new_entry], ignore_index=True)

            if MOCK == 'D':
                print('30 nuevas GEHR en galaxias nuevas')
                newGEHR_newGAL = np.random.choice(galaxies_in_group[~match], size=30, replace=True)

                for G in newGEHR_newGAL:
                    new_ix = filtro['index'].iloc[-1] + 1
                    mu = df_group[df_group['galaxy']== G]['mu_W'].iloc[0]
                    mu_e = df_group[df_group['galaxy']== G]['sigma_W'].iloc[0]

                    
                    mock_GEHR = simulate_GEHR_host(
                        mu=mu,
                        alpha=33.25,
                        beta=5.02,
                        sigma_int=0.26,
                        logsigma_reference = reference["log_sigma"],
                        logsigma_error_real = reference["e_log_sigma"],
                        logflux_error_real = reference["e_log_f_Hbeta"],
                        )       

                    new_entry = pd.DataFrame([{'index': new_ix, 
                                                'z_or_mu': mu,
                                                'e_z_or_e_mu': mu_e,
                                                'log_sigma': mock_GEHR['LogSigma'].iloc[0],
                                                'e_log_sigma': mock_GEHR['LogSigma_e'].iloc[0],
                                                'log_f_Hbeta': mock_GEHR['LogFlux_Hb'].iloc[0],
                                                'e_log_f_Hbeta': mock_GEHR['LogFlux_e'].iloc[0],
                                                'origin_id': 0.0,
                                                'GEHR_id': G,
                                                'Mock': True,}])

                    filtro = pd.concat([filtro, new_entry], ignore_index=True)

            # ============================================================
            # MOCK E, F, G, H, I, J
            # ============================================================

            if MOCK in ('E', 'F', 'G', 'H', 'I', 'J'):

                # Referencia: únicamente GEHR originalmente observadas.
                # Evitamos utilizar mocks recién generados como referencia.
                #reference = filtro[filtro['origin_id'] == 0.0].copy()

                # Índice único para cada nueva GEHR
                next_ix = int(filtro['index'].max()) + 1

                # Tabla de distancias por galaxia
                distance_table = (
                    df_group.drop_duplicates(subset='galaxy')
                    .set_index('galaxy')
                )

                # Función auxiliar para generar una GEHR
                def create_mock_GEHR(G, new_ix):

                    if G not in distance_table.index:
                        raise ValueError(
                            f"La galaxia {G} no está en el archivo de anclas."
                        )

                    mu = distance_table.at[G, 'mu_W']
                    mu_e = distance_table.at[G, error_kind]

                    mock_GEHR = simulate_GEHR_host(
                        mu=mu,
                        alpha=33.25,
                        beta=5.02,
                        sigma_int=0.26,
                        logsigma_reference=reference["log_sigma"],
                        logsigma_error_real=reference["e_log_sigma"],
                        logflux_error_real=reference["e_log_f_Hbeta"],
                    )

                    new_entry = pd.DataFrame([{
                        'index': new_ix,
                        'z_or_mu': mu,
                        'e_z_or_e_mu': mu_e,
                        'log_sigma': mock_GEHR['LogSigma'].iloc[0],
                        'e_log_sigma': mock_GEHR['LogSigma_e'].iloc[0],
                        'log_f_Hbeta': mock_GEHR['LogFlux_Hb'].iloc[0],
                        'e_log_f_Hbeta': mock_GEHR['LogFlux_e'].iloc[0],
                        'origin_id': 0.0,
                        'GEHR_id': G,
                        'Mock': True,
                    }])

                    return new_entry

                # ========================================================
                # MOCK E: Número fijo de nuevas GEHR por galaxia
                # ========================================================

                if MOCK == 'E':

                    print('MOCK E: GEHR adicionales por galaxia')

                    for G, n_new in newGEHR_per_galaxy.items():

                        if G not in distance_table.index:
                            print(f"[SKIP] {G}: no está en el archivo de anclas.")
                            continue                    

                        print(f"{G}: agregando {n_new} GEHR")

                        if not isinstance(n_new, (int, np.integer)) or n_new < 0:
                            raise ValueError(
                                f"Número inválido de GEHR para {G}: {n_new}"
                            )

                        for _ in range(n_new):

                            new_entry = create_mock_GEHR(G, next_ix)

                            filtro = pd.concat(
                                [filtro, new_entry],
                                ignore_index=True
                            )

                            next_ix += 1


                # ========================================================
                # MOCK F: Exactamente 2 GEHR por galaxia
                # MOCK G: Exactamente 5 GEHR por galaxia
                # Incluye galaxias sin GEHR originalmente observadas
                # ========================================================

                if MOCK in ('F', 'G'):

                    target = 2 if MOCK == 'F' else 5

                    print(
                        f'MOCK {MOCK}: {target} GEHR por galaxia anfitriona'
                    )

                    rng = np.random.default_rng(42)

                    # Recorrer TODAS las galaxias del archivo de anclas,
                    # incluso aquellas sin GEHR observadas.
                    for G in galaxies_in_group:

                        # Identificar GEHR originales de esta galaxia
                        idx = filtro.index[
                            (filtro['origin_id'] == 0.0) &
                            (filtro['GEHR_id'] == G)
                        ].to_numpy()

                        n_obs = len(idx)

                        # CASO 1: Hay más GEHR observadas que el objetivo
                        if n_obs > target:

                            n_remove = n_obs - target

                            drop_idx = rng.choice(
                                idx,
                                size=n_remove,
                                replace=False
                            )

                            filtro = filtro.drop(index=drop_idx)

                            print(
                                f"{G}: {n_obs} observadas -> "
                                f"{target} conservadas"
                            )

                        # CASO 2: Hay menos GEHR que el objetivo
                        # Incluye galaxias con 0 GEHR observadas
                        elif n_obs < target:

                            n_new = target - n_obs

                            for _ in range(n_new):

                                new_entry = create_mock_GEHR(G, next_ix)

                                filtro = pd.concat(
                                    [filtro, new_entry],
                                    ignore_index=True
                                )

                                next_ix += 1

                            print(
                                f"{G}: {n_obs} observadas + "
                                f"{n_new} simuladas = {target}"
                            )

                        # CASO 3: Ya tiene exactamente el número deseado
                        else:

                            print(
                                f"{G}: {target} observadas, sin cambios"
                            )

                    filtro = filtro.reset_index(drop=True)





                if MOCK in ('H', 'I', 'J'):
                    gehr_mask = (
                        (DF_pd['origin_id'] == 0.0) &
                        (DF_pd['GEHR_id'].isin(selected_hosts))
                    )

                    filtro = DF_pd[gehr_mask | (DF_pd['origin_id'] > 0.0)].copy()
                    filtro = filtro.merge(
                        df_group[['galaxy', 'mu_W', f'{error_kind}']], 
                        left_on='GEHR_id', 
                        right_on='galaxy', 
                        how='left'
                    )
                    coincide = filtro['mu_W'].notna()
                    filtro.loc[coincide, 'z_or_mu'] = filtro.loc[coincide, 'mu_W']
                    filtro.loc[coincide, 'e_z_or_e_mu'] = filtro.loc[coincide, f'{error_kind}']
                    filtro = filtro.drop(columns=['galaxy', 'mu_W', f'{error_kind}'])
                    
                    filtro['Mock'] = False



                    # ========================================================
                    # MOCK H: Solo GEHR observadas de galaxias seleccionadas
                    # ========================================================

                    if MOCK == 'H':

                        print('MOCK H: GEHR observadas en galaxias seleccionadas')

                        for G in selected_hosts:

                            n_obs = len(filtro[
                                (filtro['origin_id'] == 0.0) &
                                (filtro['GEHR_id'] == G)
                            ])

                            if n_obs > 0:
                                print(f"{G}: {n_obs} GEHR observadas")
                            else:
                                print(f"[SKIP] {G}: sin GEHR observadas")


                    # ========================================================
                    # MOCK I: Galaxias seleccionadas + GEHR simuladas
                    # en galaxias sin observaciones
                    # ========================================================

                    if MOCK == 'I':

                        print('MOCK I: Galaxias seleccionadas con simulaciones adicionales')

                        for G in selected_hosts:

                            n_obs = len(filtro[
                                (filtro['origin_id'] == 0.0) &
                                (filtro['GEHR_id'] == G)
                            ])

                            # Si tiene observaciones, conservarlas sin añadir mocks
                            if n_obs > 0:
                                print(f"{G}: {n_obs} GEHR observadas")
                                continue

                            # Solo simular si existe distancia de calibración
                            if G not in distance_table.index:
                                print(f"[SKIP] {G}: sin datos en df_group")
                                continue

                            # Número de GEHR especificado en el diccionario
                            n_new = newGEHR_per_galaxy.get(G, 0)

                            if not isinstance(n_new, (int, np.integer)) or n_new < 0:
                                raise ValueError(
                                    f"Número de GEHR inválido para {G}: {n_new}"
                                )

                            for _ in range(n_new):

                                new_entry = create_mock_GEHR(G, next_ix)

                                filtro = pd.concat(
                                    [filtro, new_entry],
                                    ignore_index=True
                                )

                                next_ix += 1

                            print(f"{G}: 0 observadas + {n_new} simuladas")


                    # ========================================================
                    # MOCK J: Exactamente 4 GEHR por galaxia seleccionada
                    # con distancia disponible en df_group
                    # ========================================================

                    if MOCK == 'J':

                        target = 4

                        print('MOCK J: 4 GEHR por galaxia seleccionada')

                        rng = np.random.default_rng(42)

                        for G in selected_hosts:

                            # Sin distancia de calibración no se pueden
                            # generar mocks. Se conservan sus observaciones.
                            if G not in distance_table.index:
                                print(f"[SKIP] {G}: sin distancia en df_group")
                                continue

                            idx = filtro.index[
                                (filtro['origin_id'] == 0.0) &
                                (filtro['GEHR_id'] == G)
                            ].to_numpy()

                            n_obs = len(idx)

                            # CASO 1: Más de 4 GEHR observadas
                            if n_obs > target:

                                n_remove = n_obs - target

                                drop_idx = rng.choice(
                                    idx,
                                    size=n_remove,
                                    replace=False
                                )

                                filtro = filtro.drop(index=drop_idx)

                                print(
                                    f"{G}: {n_obs} observadas -> "
                                    f"{target} conservadas"
                                )

                            # CASO 2: Menos de 4 GEHR observadas
                            elif n_obs < target:

                                n_new = target - n_obs

                                for _ in range(n_new):

                                    new_entry = create_mock_GEHR(G, next_ix)

                                    filtro = pd.concat(
                                        [filtro, new_entry],
                                        ignore_index=True
                                    )

                                    next_ix += 1

                                print(
                                    f"{G}: {n_obs} observadas + "
                                    f"{n_new} simuladas = {target}"
                                )

                            # CASO 3: Exactamente 4 GEHR observadas
                            else:

                                print(f"{G}: {target} observadas, sin cambios")

                        filtro = filtro.reset_index(drop=True)


            new_DF = Table.from_pandas(filtro)
            return new_DF
        else:
            return None


    '''
    Hasta aqui esta todo igual, salvo por la adicion de las nuevas librerias. De aqui en adelante voy a probar las funciones para probar lensing.
    '''

    def lens_sigma_eff(self,z):
        """
        Fiducial lensing prescription used in the paper:
        sigma_eff(z) = 0.088 z
        """
        z = np.asarray(z, dtype=float)
        return np.clip(0.088 * z, 0.0, None)

    def lens_lognormal_params(self,z):
        """
        Return U, S for:
            t = ln(M) ~ Normal(U,S^2)
        with E[M] = 1
        """
        sigma_eff = self.lens_sigma_eff(z)
        S = np.sqrt(np.log1p(sigma_eff**2))
        U = -0.5 * S**2
        return U, S

    def prepare_lensing_cache(self, DF=None, zmax = 20.0):
        """
        Precompute the lensing prior once because galaxy redshifts do not change during nested sampling.
        Anchors (origin_id == 0) must never receive this correction.
        """
        if DF is None:
            return None

        origin = np.asarray(DF["origin_id"], dtype=float)
        z_or_mu = np.asarray(DF["z_or_mu"], dtype=float)
        is_hiig = origin != 0.0
        mask = (
            is_hiig
            & np.isfinite(z_or_mu)
            & (z_or_mu > 0.0)
            & (z_or_mu < zmax)
        )
        U = np.zeros(len(DF), dtype=float)
        S = np.zeros(len(DF),dtype=float)
        if np.any(mask):
            U[mask], S[mask] = self.lens_lognormal_params(
                z_or_mu[mask]
            )
        return {
            "mask": mask,
            "U": U,
            "S": S
        }


    # Marginalizacion

    def lensing_loglike_gh(
        self,
        residual,
        sigma_dm,
        lens_cache,
        normalized = False,):
        """
        Per-object likelihood marginalized over t = ln(M), using 20-point Gauss-Hermite quadrature.

        residual:
            mu_obs - mu_cosmo

        sigma_dm:
            Gaussian non-lensing uncertainty in distance modulus.
        """
        R = np.asarray(residual, dtype=float)
        sigma = np.asarray(sigma_dm, dtype=float)
        if R.shape != sigma.shape:
            raise ValueError("residual and sigma_dm musta have same shape")
        sigma = np.clip(sigma, 1.0e-12, None)
        # Ordinary Gaussian likelihood for all objects.
        logp = -0.5 * (R/sigma)**2
        if normalized:
            # -0.5 log(2*pi) is omitted because it is a true constant.
            logp -= np.log(sigma)
        mask = np.asarray(lens_cache["mask"], dtype = bool)
        if not np.any(mask):
            return np.sum(logp)
        U = np.asarray(lens_cache["U"])[mask]
        S = np.asarray(lens_cache["S"])[mask]
        Rm = R[mask]
        sm = sigma[mask]
        # Eq. 25 of the paper:
        # t_ik = U_i + sqrt(2) S_i x_k
        t = (
            U[:, None]
            + np.sqrt(2.0)
            * S[:, None]
            * LENS_GH_X[None,:] 
        )
        # Eq. 26:
        arg = (
            Rm[:,None]
            + LENS_DM_COEFF * t
        ) / sm[:, None]

        log_terms = (
            np.log(LENS_GH_W)[None, :]
            -0.5 * arg**2
        )
        # Eq. 27, evaluated stably.
        log_expectation = (
            -0.5 * np.log(np.pi)
            + logsumexp(log_terms, axis = 1 )
        )
        if normalized:
            log_expectation -= np.log(sm)

        logp[mask] = log_expectation
        return np.sum(logp)



    # Reconfiguracion del lnlike
    def lnlike(
        self,
        theta = None,
        DF = None,
        lens_cache =None,):
        if theta is None or DF is None:
            return None
        alpha, beta, h = theta
        # Incluir radiacion como se menciona en el paper
        try:
            cosmo = FlatLambdaCDM(
                H0=h * 100.0,
                Om0=0.3,
                Tcmb0=2.7255 * u.K,
            )
        except Exception:
            return -np.inf
        origin = np.asarray(DF["origin_id"], dtype=float)
        z_or_mu = np.asarray(
            DF["z_or_mu"],
            dtype=float,
        )
        e_z_or_mu = np.asarray(
            DF["e_z_or_e_mu"],
            dtype=float,
        )
        log_sigma = np.asarray(
            DF["log_sigma"],
            dtype=float,
        )
        e_log_sigma = np.asarray(
            DF["e_log_sigma"],
            dtype=float,
        )
        log_f = np.asarray(
            DF["log_f_Hbeta"],
            dtype=float,
        )
        e_log_f = np.asarray(
            DF["e_log_f_Hbeta"],
            dtype=float,
        )   
        anchors = origin == 0.0
        hiig = ~anchors
        n = len(DF)
        mu_model = np.empty(n, dtype=float)
        mu_model_err = np.empty(n, dtype=float)
        # Primary-distance anchors
        mu_model[anchors] = z_or_mu[anchors]
        mu_model_err[anchors] = e_z_or_mu[anchors]
        # HII galaxies: exact luminosity distance at every z
        z = z_or_mu[hiig]
        ez = e_z_or_mu[hiig]
        if np.any(
            (~np.isfinite(z))
            | (z <= 0.0)
        ):
            return -np.inf
        try:
            d_l = cosmo.luminosity_distance(z).value
        except Exception:
            return -np.inf            
        if np.any(
            (~np.isfinite(d_l))
            | (d_l <= 0.0)
        ):
            return -np.inf
        mu_model[hiig] = (
            5.0 * np.log10(d_l)
            + 25.0
        )
        # d(mu)/dz exacto para una cosmologia plana FLRW
        # Revisar esto despues, derivar para posibles nuevas cosmologias o la integracion de epsilon (dispersion)
        Ez = cosmo.efunc(z)
        Iz = (
            cosmo.comoving_distance(z)
            * cosmo.H0
            / const.c.to(u.km/u.s)
        ).to_value(u.dimensionless_unscaled)
        dmu_dz = (
            5.0 / np.log(10.0)
        ) * (
            1.0 / (1.0 + z)
            + 1.0 / (Ez * Iz)
        )
        mu_model_err[hiig] = (
            np.abs(dmu_dz) * ez
        )
        # L-sigma distance estimator.
        mu_obs = (
            2.5
            * (
                beta * log_sigma
                + alpha
                - log_f
            )
            - 100.19477738511641
        )
        mu_obs_err = (
            2.5
            * np.sqrt(
                e_log_f**2
                + beta**2 * e_log_sigma**2
            )
        )
        # Aqui se podria agregar el valor para los errores sistematicos por objeto, pero por ahora lo voy a dejar en cero por que no estoy seguro de si esto se usa antes o despues.
        sigma_sys = np.zeros(n, dtype=float)
        sigma_dm = np.sqrt(
            mu_obs_err**2
            + mu_model_err**2
            + sigma_sys**2
        )
        residual = mu_obs - mu_model
        if np.any(~np.isfinite(residual)):
            return -np.inf
        if np.any(~np.isfinite(sigma_dm)):
            return -np.inf
        return self.lensing_loglike_gh(
            residual=residual,
            sigma_dm=sigma_dm,
            lens_cache=lens_cache,
            normalized=False,
        )


    def main_sampler(self):

        concat_filter_DF = self.group_reader(
            DF = self.data_frame,
            group = self.distance_estimator_set,
            error_kind = self.estimator_error_kind,
            MOCK=ENV_MOCK_VAR
        )

        concat_filter_DF.write(f'{self.outdir}/dataset_run.csv', format='ascii.csv', overwrite=True)

        lens_cache = self.prepare_lensing_cache(
            concat_filter_DF,
            zmax = 20.0,
        )

        # Test de reproduccion de datos

        origin = np.asarray(
            concat_filter_DF["origin_id"],
              dtype=float,
        )

        is_anchor = origin == 0.0
        is_hiig = ~is_anchor

        z_hiig = np.asarray(
            concat_filter_DF["z_or_mu"][is_hiig],
            dtype=float,
        )


        def prior_transform(u):
            alpha = 32.5 + 2.0 * u[0]     # 32.5 -> 34.5
            beta  =  4.5 + 1.0 * u[1]     # 4.5 -> 5.5
            h    =  0.5 +  0.5 * u[2]     # 0.5 -> 1.0
            return np.array([alpha, beta, h])

        def loglike(theta):
            return self.lnlike(
                theta = theta,
                DF = concat_filter_DF,
                lens_cache = lens_cache,
                )

        n_dims = 3
        result = pymultinest.solve(
            LogLikelihood=loglike,
            Prior=prior_transform,
            n_dims=n_dims,
            outputfiles_basename=self.prefix,
            evidence_tolerance=0.5,
            n_live_points=2500,
            multimodal=True,
            verbose=False
        )


        print("\nEvidence lnZ = %.3f ± %.3f" % (result["logZ"], result["logZerr"]))


        # Creacion del Triangle Plot

        samples = result["samples"]   # shape = (Nsamples, 3)
        parameters = [r"\alpha", r"\beta", r"h"]
        print("Posterior means/stdev:")
        for name, col in zip(parameters, samples.T):
            print(f"{name:>6s}: {col.mean():.3f} ± {col.std():.3f}")

        with open(self.prefix + "params.json", "w") as f:
            json.dump(parameters, f, indent=2)

        gds = MCSamples(
            samples=samples,
            names=["alpha", "beta", "h"],
            labels=[r"\alpha", r"\beta", r"h"] 
        )
        g = plots.getSubplotPlotter()
        g.settings.num_plot_contours = 4
        g.triangle_plot([gds], filled=True, title_limit=1, colors=['red'])

        #titulo = g.fig.suptitle(f"{self.main_title}", fontsize=16, y=1.03)

        plt.savefig(
            self.prefix + "triangle_getdist.png", 
            bbox_inches='tight', 
            #bbox_extra_artists=[titulo],
            dpi=150,
            transparent=True
        )
        plt.close(g.fig)
        #print("GetDist triangle en:", self.prefix + "triangle_getdist.png")
        #print("\nListo. Archivos en:", self.outdir)







    def luminosity_sigma_plane(self,ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            dataset = pd.read_csv(f'{self.outdir}/dataset_run.csv')

            # Reading results from Multinest
            results_path = self.prefix
            analyzer = pymultinest.analyse.Analyzer(
                n_params=3,
                outputfiles_basename=results_path
            )
            stats = analyzer.get_stats()
            alpha_mean,beta_mean,h_mean = stats['modes'][0]['mean']
            alpha_err,beta_err,h_err = stats['modes'][0]['sigma']




            ####
            # Defining cosmology
            ####

            cosmo = FlatLambdaCDM(
                H0=h_mean * 100.0,
                Om0=0.3,
                Tcmb0=2.7255 * u.K,
            )



            N_total   = len(dataset)
            N_anchors = len(dataset[dataset["origin_id"]==0.0])
            N_HIIG    = len(dataset[dataset["origin_id"]!=0.0])
            z_min     = np.min(dataset[dataset["origin_id"]!=0.0]["z_or_mu"])
            z_max     = np.max(dataset[dataset["origin_id"]!=0.0]["z_or_mu"])

            markers = {
                0: "s", 1: "o", 2: "D",
                3: "^", 4: "v", 5: "p",
                6: "*", 7: "P", 8: "h",
                9: "8"
            }
            # square circle diamond triangle up triangle down x star filled plus hexagon
            colors = {
                0: "#C46A46",   # Terracota
                1: "#4477AA",   # Azul
                2: "#228833",
                3: "#AA3377",
                4: "#CCBB44",
                5: "#777777",
                6: "#8855AA",
                7: "#44AA99",
                8: "#333333",
                9: "#36DB4F",
            }
            samples = {
                0: "GEHR",
                1: "Local HIIG",
                2: "Literature (Mid-z)",
                3: "VLT/X-shooter",
                4: "Keck/MOSFIRE",
                5: "VLT/KMOS",
                6: "JWST/NIRSpec",
                7: "VUDS/VANDELS",
                8: "ALMA+JWST/MIRI",
                9: "Mock GEHR"
            }

            def draw_points(x, y, ex, ey, group_id):
                color = colors[group_id]
                ax.errorbar(
                    x, y,
                    xerr=ex,
                    yerr=ey,
                    fmt=markers[group_id],
                    linestyle="none",
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.45,
                    markersize=5.5 if group_id == 0 else 4.5,
                    ecolor=to_rgba(color, 0.35),
                    elinewidth=0.7,
                    capsize=0,
                    barsabove=False,
                    zorder=3 if group_id == 0 else 2,
                    label=f"{samples[group_id]} ({len(x)})",
                )


            GEHR = dataset[dataset["origin_id"]==0.0]
            HIIG = dataset[dataset["origin_id"]!=0.0]

            GEHR_obs = GEHR[GEHR['Mock']==False]
            GEHR_mock = GEHR[GEHR['Mock']==True]

            # GEHR section

            pc_to_cm = 3.08567758e18
            zeta = pc_to_cm 
            logL = np.log10(4*np.pi*(zeta)**2) + ((2*GEHR_obs['z_or_mu'] +10)/5) + GEHR_obs['log_f_Hbeta']
            e_logL = np.sqrt(((2/5)*GEHR_obs['e_z_or_e_mu'])**2 + (GEHR_obs['e_log_f_Hbeta'])**2)
            logSigma = GEHR_obs['log_sigma']
            e_logSigma = GEHR_obs['e_log_sigma']
            draw_points(logSigma, logL, e_logSigma, e_logL, 0)

            pc_to_cm = 3.08567758e18
            zeta = pc_to_cm 
            logL = np.log10(4*np.pi*(zeta)**2) + ((2*GEHR_mock['z_or_mu'] +10)/5) + GEHR_mock['log_f_Hbeta']
            e_logL = np.sqrt(((2/5)*GEHR_mock['e_z_or_e_mu'])**2 + (GEHR_mock['e_log_f_Hbeta'])**2)
            logSigma = GEHR_mock['log_sigma']
            e_logSigma = GEHR_mock['e_log_sigma']
            draw_points(logSigma, logL, e_logSigma, e_logL, 9)

            # HIIG section

            for e, group in HIIG.groupby("origin_id", sort=True):
                e = int(e)
                z = group["z_or_mu"].to_numpy()
                ez = group["e_z_or_e_mu"].to_numpy()
                Mpc_to_cm = 3.08567758e24
                c = 299792.458
                Ez = cosmo.efunc(z)
                Iz = (
                    cosmo.comoving_distance(z)
                    * cosmo.H0
                    / const.c.to(u.km / u.s)
                ).to_value(u.dimensionless_unscaled)
                logL = (
                    np.log10(4 * np.pi * Mpc_to_cm**2 * c**2)
                    - 2 * np.log10(h_mean * 100)
                    + group["log_f_Hbeta"].to_numpy()
                    + 2 * np.log10(1 + z)
                    + 2 * np.log10(Iz)
                )
                e_logL = np.sqrt(
                    (
                        (2 / np.log(10))
                        * (1 / (1 + z) + 1 / (Ez * Iz))
                        * ez
                    )**2
                    + (2 * h_err / (h_mean * np.log(10)))**2
                    + group["e_log_f_Hbeta"].to_numpy()**2
                )
                draw_points(
                    group["log_sigma"].to_numpy(),
                    logL,
                    group["e_log_sigma"].to_numpy(),
                    e_logL,
                    e,
                )

            xmin, xmax = ax.get_xlim()
            x_line = np.linspace(xmin, xmax, 200)

            ax.plot(
                x_line,
                beta_mean * x_line + alpha_mean,
                color="#333333",
                linestyle="--",
                linewidth=1.3,
                zorder=1,
                label=r"$\log L(\mathrm{H}\beta)=\alpha+\beta\log\sigma$",
            )

            ax.set_xlabel(
                r"$\log_{10}\,[\sigma/(\mathrm{km\,s^{-1}})]$",
                fontsize=16,
                labelpad=8,
            )
            ax.set_ylabel(
                r"$\log_{10}\,[L(\mathrm{H}\beta)/(\mathrm{erg\,s^{-1}})]$",
                fontsize=16,
                labelpad=8,
            )

            ax.set_box_aspect(1)
            ax.set_axisbelow(True)
            ax.grid(True, color="#DDDDDD", linewidth=0.6, alpha=0.6)

            ax.tick_params(
                axis="both",
                which="major",
                labelsize=11,
                direction="in",
                length=4,
            )

            for spine in ax.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(0.8)

            ax.legend(
                loc="upper left",
                ncol=1,
                fontsize=10,
                frameon=True,
                facecolor="white",
                edgecolor="#DDDDDD",
                framealpha=0.95,
                handlelength=2,
                borderpad=0.7,
                labelspacing=0.5,
            )

            # Creation of the second legend

            code_reader = pd.read_csv('Id_codes.csv',comment="#",index_col=False)

            id_code_data = code_reader[code_reader['Code']==self.code_id].iloc[0]

            labels_x_seconLeg = [f'Code id: {id_code_data['Code']}',
                                 f'Nickname: {id_code_data['Nickname']}',
                                 f'Redshift range: {id_code_data['Redshift_range']}',
                                 f'Cosmology: {id_code_data['Cosmology']}',
                                 f'Sample: {id_code_data['Sample']}',
                                 f'Number of objects: {N_total}',
                                 f'Number of GEHR: {N_anchors}',
                                 f'Number of HIIG: {N_HIIG}',
                                 r'$z_{min}$'+f' {z_min}',
                                 r'$z_{max}$'+f' {z_max}',]

            for G in np.unique(GEHR['GEHR_id']):
                selec = GEHR[GEHR['GEHR_id']==G]
                N_obs = len(GEHR[(GEHR['GEHR_id'] == G) & ~GEHR['Mock']])
                N_mock = len(GEHR[(GEHR['GEHR_id'] == G) & GEHR['Mock']])
                mu = selec['z_or_mu'].iloc[0]
                e_mu = selec['e_z_or_e_mu'].iloc[0]

                labels_x_seconLeg.append(f'{G} No:({N_obs},{N_mock}) ' + r'$\mu$ = '+ f'{mu:.2f}'+ r'$\pm$' + f'{e_mu:.2f}')

        return ax, labels_x_seconLeg



    def plots_maker(self):

        # Crear la figura sin ejes iniciales superpuestos.
        fig = plt.figure(figsize=(16, 9), dpi=150)

        gs = fig.add_gridspec(
            1, 2,
            width_ratios=[1, 1],
            left=0.075,
            right=0.975,
            bottom=0.27,
            top=0.96,
            wspace=0.17,
        )

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Panel izquierdo.
        _, info_labels = self.luminosity_sigma_plane(ax=ax1)

        # Panel derecho: adaptar la imagen al espacio disponible.
        img = mpimg.imread(self.prefix + "triangle_getdist.png")

        ax2.imshow(
            img,
            aspect="equal",
            interpolation="antialiased",
        )
        ax2.set_anchor("C")
        ax2.set_axis_off()

        # Leyenda informativa común, sin símbolos.
        info_handles = [
            Line2D([], [], linestyle="none", marker="")
            for _ in info_labels
        ]

        # Intentar hasta cuatro columnas y reducirlas si no caben.
        max_columns = min(4, len(info_labels))

        for ncols in range(max_columns, 0, -1):
            info_legend = fig.legend(
                handles=info_handles,
                labels=info_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.16),
                bbox_transform=fig.transFigure,
                ncol=ncols,
                fontsize=9,
                frameon=True,
                facecolor="#FAFAFA",
                edgecolor="#DDDDDD",
                handlelength=0,
                handletextpad=0,
                columnspacing=2.0,
                labelspacing=0.65,
                borderpad=0.9,
                borderaxespad=0,
            )

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            legend_width = info_legend.get_window_extent(renderer).width
            available_width = 0.94 * fig.bbox.width

            if legend_width <= available_width or ncols == 1:
                break

            info_legend.remove()

        fig.savefig(
            self.prefix + "run.png",
            dpi=150,
            bbox_inches="tight",
            bbox_extra_artists=(info_legend,),
            pad_inches=0.15,
            facecolor="white",
        )

        plt.close(fig)





def select_redshift_cut(tab, zmax):
    """
    Mantiene siempre la muestra ancla origin_id == 0.
    Para objetos no-ancla, aplica z_or_mu <= zmax.
    Si zmax is None, usa todos los objetos.
    """

    origin = np.asarray(tab["origin_id"], dtype=float)

    if zmax is None:
        mask = np.ones(len(tab), dtype=bool)
    else:
        z_or_mu = np.asarray(tab["z_or_mu"], dtype=float)

        mask = (
            (origin == 0.0)
            |
            ((origin != 0.0) & (z_or_mu <= zmax))
        )

    return tab[mask]






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='Sample dataset of the L-sigma relation (fluxes and sigmas)')
    parser.add_argument('--z_cut', type=float, help='Cut in redshift')
    parser.add_argument('--anchor_data', type=str, help='CSV with the data of the anchors')
    parser.add_argument('--sub_id', type=str, help='Code subid for identification')


    args = parser.parse_args()

    # Lee datos L-sigma
    LSdata_df = pd.read_csv(
        args.dataset,
        comment="#",
        index_col=False,
        dtype={"GEHR_id": str},
    )

    z_cut = args.z_cut

    id_inS = args.anchor_data.replace('_recal','')
    id_inS = id_inS.replace('.csv','')

    # Does the Results file exists?
    home = os.path.expanduser('~/HIIGalaxies/Bayesian_samplers/Multinest/MOCKS')
    anchor_group = args.anchor_data.replace('_recal.csv','')
    ruta_carpeta = f"{home}/Results/{ENV_MOCK_VAR}/{anchor_group}"
    ruta_archivo = f"{ruta_carpeta}/Results.csv"


    complete_run = False
    if os.path.exists(ruta_archivo) == True:
        results_file = pd.read_csv(ruta_archivo,comment="#",index_col=False)
        results_check = results_file[(results_file['code_id'] == f"{ENV_MOCK_VAR}.{args.sub_id}") & (results_file['subgrupo'] == f"{anchor_group}")]
        if len(results_check) == 1:
            complete_run = True
    

    if not os.path.exists(ruta_archivo) or complete_run == False:
        Lsig_Ho_sampler(
            data_frame=select_redshift_cut(Table.from_pandas(LSdata_df), z_cut),
            distance_estimator_set= args.anchor_data,
            estimator_error_kind='sigma_W',
            main_title=f"{ENV_MOCK_VAR}.{args.sub_id} {id_inS} anchor",
            folder_name=f"{ENV_MOCK_VAR}.{args.sub_id}_{id_inS}",
            analysis_mode='None',
            id_prefix=f'{ENV_MOCK_VAR}.{args.sub_id}',
            sub_id_code = f'{args.sub_id}'
        )



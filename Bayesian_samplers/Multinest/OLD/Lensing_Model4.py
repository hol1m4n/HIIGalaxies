from __future__ import annotations
import os
import json
import numpy as np
from astropy.table import Table
import pandas as pd

# Cosmologias
from astropy.cosmology import FlatwCDM
from astropy.cosmology import Flatw0waCDM
from astropy.cosmology import wCDM
from astropy.cosmology import w0waCDM
from astropy.cosmology import LambdaCDM
from astropy.cosmology import FlatLambdaCDM

import pymultinest
import matplotlib.pyplot as plt
#import corner
from getdist import plots, MCSamples
import scipy.optimize as op
import astropy.units as u
from scipy import stats
import scipy.optimize as op
from astropy import constants as const
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp



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
                 id_prefix = None):
        
        self.data_frame = data_frame
        self.distance_estimator_set = distance_estimator_set
        self.estimator_error_kind = estimator_error_kind
        self.main_title = main_title
        self.folder_name = folder_name
        self.analysis_mode = analysis_mode
        self.id_prefix = id_prefix
        self.outdir = None
        self.prefix = None

        self.dir_results_creation()
        self.main_sampler()

    def dir_results_creation(self):
        self.outdir = os.path.join(os.path.expanduser('~/HIIGalaxies/Bayesian_samplers/Multinest'), f"{self.folder_name}")
        os.makedirs(self.outdir, exist_ok=True)
        self.prefix = os.path.join(self.outdir, f"{self.id_prefix}_")



    def group_reader(self, DF=None, group=None, error_kind = None):

        if DF != None and group != None and error_kind != None:
            DF_pd = DF.to_pandas()
            df_group = pd.read_csv(group)
            galaxies_in_group = df_group['Galaxia'].unique()
            filtro = DF_pd[DF_pd['GEHR_id'].isin(galaxies_in_group) | (DF_pd['origin_id'] > 0.0)].copy()
            filtro = filtro.merge(
                df_group[['Galaxia', 'mu_w', f'{error_kind}']], 
                left_on='GEHR_id', 
                right_on='Galaxia', 
                how='left'
            )
            coincide = filtro['mu_w'].notna()
            filtro.loc[coincide, 'z_or_mu'] = filtro.loc[coincide, 'mu_w']
            filtro.loc[coincide, 'e_z_or_e_mu'] = filtro.loc[coincide, f'{error_kind}']
            filtro = filtro.drop(columns=['Galaxia', 'mu_w', f'{error_kind}'])
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

        alpha, beta, h, Om, wDE = theta


        # Incluir radiacion como se menciona en el paper

        try:
            cosmo = FlatwCDM(
                H0=h * 100.0,
                Om0=Om,
                w0=wDE,
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
        )

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

        print("N total   =",len(concat_filter_DF))
        print("N anchors =",np.sum(is_anchor))
        print("N HIIG    =",np.sum(is_hiig))
        print("z min     =",np.min(z_hiig))
        print("z max     =",np.max(z_hiig))





        def prior_transform(u):
            alpha = 32.5 + 2.0 * u[0]     # 32.5 -> 34.5
            beta  =  4.5 + 1.0 * u[1]     # 4.5 -> 5.5
            h    =  0.5 +  0.5 * u[2]     # 0.5 -> 1.0
            Om    =  0.0 +  1.0 * u[3]    # 0.0 -> 0.1
            wDE    =  -2.0 +  2.0 * u[4]  # -2.0 -> 0.0
            return np.array([alpha, beta, h, Om, wDE])

        def loglike(theta):
            return self.lnlike(
                theta = theta,
                DF = concat_filter_DF,
                lens_cache = lens_cache,
                )

        n_dims = 5
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


        samples = result["samples"]   # shape = (Nsamples, 3)
        parameters = [r"\alpha", r"\beta", r"h",r"\Omega_m",r"{\omega}_{DE}"]
        print("Posterior means/stdev:")
        for name, col in zip(parameters, samples.T):
            print(f"{name:>6s}: {col.mean():.3f} ± {col.std():.3f}")


        with open(self.prefix + "params.json", "w") as f:
            json.dump(parameters, f, indent=2)


        gds = MCSamples(
            samples=samples,
            names=["alpha", "beta", "h","Om","wDE"],
            labels=[r"\alpha", r"\beta", r"h",r"\Omega_m",r"{\omega}_{DE}"] 
        )
        g = plots.getSubplotPlotter()
        g.settings.num_plot_contours = 4
        g.triangle_plot([gds], filled=True, title_limit=1, colors=['red'])

        #titulo = g.fig.suptitle(f"{self.main_title}", fontsize=16, y=1.03)

        plt.savefig(
            self.prefix + "triangle_getdist.png", 
            bbox_inches='tight', 
            #bbox_extra_artists=[titulo],
            dpi=800,
            transparent=True
        )
        plt.close(g.fig)

        print("GetDist triangle en:", self.prefix + "triangle_getdist.png")

        print("\nListo. Archivos en:", self.outdir)



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



# Lee datos L-sigma
LSdata_df = pd.read_csv(
    "Compilation2026_main.csv",
    comment="#",
    index_col=False,
    dtype={"GEHR_id": str},
)

LS_tab = Table.from_pandas(LSdata_df)

data_cut = select_redshift_cut(LS_tab, None)


Lsig_Ho_sampler(
    data_frame=data_cut,
    distance_estimator_set='GEHR_base.csv',
    estimator_error_kind='sigma_w',
    main_title="Lensing test",
    folder_name="Lens_FlatwCDM",
    analysis_mode='High',
    id_prefix='Lens_FlatwCDM',
)

Lsig_Ho_sampler(
    data_frame=data_cut,
    distance_estimator_set='sm_r.csv',
    estimator_error_kind='sigma_w',
    main_title="Lensing test",
    folder_name="Lens_FlatwCDM_CepnZeR",
    analysis_mode='High',
    id_prefix='Lens_FlatwCDM_CepnZeR',
)

Lsig_Ho_sampler(
    data_frame=data_cut,
    distance_estimator_set='sm_t.csv',
    estimator_error_kind='sigma_w',
    main_title="Lensing test",
    folder_name="Lens_FlatwCDM_CepnZeT",
    analysis_mode='High',
    id_prefix='Lens_FlatwCDM_CepnZeT',
)

Lsig_Ho_sampler(
    data_frame=data_cut,
    distance_estimator_set='cm_r.csv',
    estimator_error_kind='sigma_w',
    main_title="Lensing test",
    folder_name="Lens_FlatwCDM_CepcZeR",
    analysis_mode='High',
    id_prefix='Lens_FlatwCDM_CepcZeR',
)

Lsig_Ho_sampler(
    data_frame=data_cut,
    distance_estimator_set='cm_t.csv',
    estimator_error_kind='sigma_w',
    main_title="Lensing test",
    folder_name="Lens_FlatwCDM_CepcZeT",
    analysis_mode='High',
    id_prefix='Lens_FlatwCDM_CepcZeT',
)

Lsig_Ho_sampler(
    data_frame=data_cut,
    distance_estimator_set='t_r.csv',
    estimator_error_kind='sigma_w',
    main_title="Lensing test",
    folder_name="Lens_FlatwCDM_TRGB",
    analysis_mode='High',
    id_prefix='Lens_FlatwCDM_TRGB',
)

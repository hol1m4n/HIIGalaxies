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

        self.code_id = 'A.1'

        self.dir_results_creation()
        self.main_sampler()
        self.plots_maker()

    def dir_results_creation(self):
        self.outdir = os.path.join(os.path.expanduser('~/HIIGalaxies/Bayesian_samplers/Multinest/MAIN'), f"{self.folder_name}")
        os.makedirs(self.outdir, exist_ok=True)
        self.prefix = os.path.join(self.outdir, f"{self.id_prefix}_")



    def group_reader(self, DF=None, group=None, error_kind = None):

        if DF != None and group != None and error_kind != None:
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
            dpi=800,
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
                8: "ALMA+JWST/MIRI"
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

            # GEHR section

            pc_to_cm = 3.08567758e18
            zeta = pc_to_cm 
            logL = np.log10(4*np.pi*(zeta)**2) + ((2*GEHR['z_or_mu'] +10)/5) + GEHR['log_f_Hbeta']
            e_logL = np.sqrt(((2/5)*GEHR['e_z_or_e_mu'])**2 + (GEHR['e_log_f_Hbeta'])**2)
            logSigma = GEHR['log_sigma']
            e_logSigma = GEHR['e_log_sigma']
            draw_points(logSigma, logL, e_logSigma, e_logL, 0)

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
                N = len(selec)
                mu = selec['z_or_mu'].iloc[0]
                e_mu = selec['e_z_or_e_mu'].iloc[0]

                labels_x_seconLeg.append(f'{G} No:({N}) ' + r'$\mu$ = '+ f'{mu:.2f}'+ r'$\pm$' + f'{e_mu:.2f}')

        return ax, labels_x_seconLeg



    def plots_maker(self):

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
            dpi=300,
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







# Lee datos L-sigma
LSdata_df = pd.read_csv(
    "Compilation2026_main.csv",
    comment="#",
    index_col=False,
    dtype={"GEHR_id": str},
)

z_cut = 0.16



subgroups = ['Joint_eR_recal.csv',
             'PLRC_all_eR_recal.csv',
             'PLRC_nZ_eR_recal.csv',
             'PLRC_nZ_eT_recal.csv',
             'PLRC_Z_eR_recal.csv',
             'PLRC_Z_eT_recal.csv',
             'TRGB_eR_recal.csv',] 

for S in subgroups:
    id_inS = S.replace('_recal.csv','')
    Lsig_Ho_sampler(
        data_frame=select_redshift_cut(Table.from_pandas(LSdata_df), z_cut),
        distance_estimator_set= S,
        estimator_error_kind='sigma_W',
        main_title=f"A.1 {id_inS} anchor",
        folder_name=f"A.1_{id_inS}",
        analysis_mode='Local',
        id_prefix=f'A.1',
    )

# Recordar cambiar siempre el self.code_id  arriba



from __future__ import annotations
import os
import json
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatwCDM
import pandas as pd

import pymultinest
import matplotlib.pyplot as plt
import corner
from getdist import plots, MCSamples
import scipy.optimize as op
import astropy.units as u
from scipy import stats
import scipy.optimize as op
from astropy import constants as const


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



    def lnlike(self,theta=None,DF=None,z_range = None):

        if theta is not None and DF is not None and z_range is not None:
            alpha, beta, h0 = theta
            cosmo = FlatwCDM(H0=h0*100.0, Om0=0.3, w0=-1.0)

            
            def dmu_dz(cosmo,z,universe_antiquity):

                if universe_antiquity == 'Low':
                    return (5.0/np.log(10.0))*(1/z)

                if universe_antiquity == 'Moderate':
                    Om_z = cosmo.Om(z)
                    Ode_z = cosmo.Ode(z)
                    q = (Om_z / 2.0) - Ode_z
                    return (10* ((q-1)*z-1) ) / (z * np.log(10) * ((q-1)*z-2))
                
                if universe_antiquity == 'High':
                    z = np.atleast_1d(np.asarray(z, dtype=float))
                    Ez = cosmo.efunc(z)  # E(z) = H(z)/H0
                    # I(z) = integral_0^z dz'/E(z') = (H0/c) * D_C(z)
                    Iz = (cosmo.comoving_distance(z) * cosmo.H0 / const.c).to_value(u.dimensionless_unscaled)
                    return np.abs((5.0/np.log(10.0)) * (1.0/(1.0+z) + 1.0/(Ez*Iz)))

            G = DF['origin_id'] == 0.0
            H = DF['origin_id'] != 0.0

            Mum = DF['z_or_mu'] * 0.0
            MumErr = DF['e_z_or_e_mu'] * 0.0

            Mum[G] = DF[G]['z_or_mu']
            MumErr[G] = DF[G]['e_z_or_e_mu']
            Mum[H] = 5.0*np.log10(cosmo.luminosity_distance(DF[H]['z_or_mu']).value) + 25.0
            MumErr[H] =  dmu_dz(cosmo, DF[H]['z_or_mu'],z_range) * DF[H]['e_z_or_e_mu']

            Mu = 2.5*(beta*DF['log_sigma'] + alpha) - 2.5*DF['log_f_Hbeta'] - 100.19477738511641 
            MuErr = 2.5*np.sqrt((DF['e_log_f_Hbeta'])**2 + beta**2*(DF['e_log_sigma'])**2)

            R = (Mu - Mum)
            W = 1.0/(MuErr**2 + MumErr**2)

            xsq = np.sum(R**2 * W)
            llq = -0.5*xsq
            return llq
        else:
            return None




    def main_sampler(self):

        concat_filter_DF = self.group_reader(DF = self.data_frame,
                                   group = self.distance_estimator_set,
                                   error_kind = self.estimator_error_kind)

        def prior_transform(u):
            alpha = 20.0 + 20.0 * u[0]     # 20->40
            beta  =  0.0 + 10.0 * u[1]     # 0->10
            h0    =  0.5 +  0.5 * u[2]     # 0.5->1.0
            return np.array([alpha, beta, h0])

        def loglike(theta):
            return self.lnlike(theta = theta,
                               DF = concat_filter_DF,
                               z_range = self.analysis_mode)

        n_dims = 3
        result = pymultinest.solve(
            LogLikelihood=loglike,
            Prior=prior_transform,
            n_dims=n_dims,
            outputfiles_basename=self.prefix,
            evidence_tolerance=0.5,
            n_live_points=100,
            multimodal=True,
            verbose=False
        )


        print("\nEvidence lnZ = %.3f ± %.3f" % (result["logZ"], result["logZerr"]))


        samples = result["samples"]   # shape = (Nsamples, 3)
        parameters = [r"\alpha", r"\beta", r"h"]
        print("Posterior means/stdev:")
        for name, col in zip(parameters, samples.T):
            print(f"{name:>6s}: {col.mean():.3f} ± {col.std():.3f}")


        with open(self.prefix + "params.json", "w") as f:
            json.dump(parameters, f, indent=2)

        fig = corner.corner(
            samples,
            labels=[r"$\alpha$", r"$\beta$", r"$h$"],
            show_titles=True,
            title_fmt=".3f",
            quantiles=[0.16, 0.5, 0.84]
        )
        fig.savefig(self.prefix + "corner.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Corner guardado en:", self.prefix + "corner.png")



        gds = MCSamples(
            samples=samples,
            names=["alpha", "beta", "h"],
            labels=[r"\alpha", r"\beta", r"h"]  
        )
        g = plots.getSubplotPlotter()
        g.settings.num_plot_contours = 2
        g.triangle_plot([gds], filled=True, title_limit=1)

        titulo = g.fig.suptitle(f"{self.main_title}", fontsize=16, y=1.03)

        plt.savefig(
            self.prefix + "triangle_getdist.png", 
            bbox_inches='tight', 
            bbox_extra_artists=[titulo],
            dpi=300
        )
        plt.close(g.fig)

        print("GetDist triangle en:", self.prefix + "triangle_getdist.png")

        print("\nListo. Archivos en:", self.outdir)        
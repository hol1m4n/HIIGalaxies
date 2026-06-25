from __future__ import annotations
import os
import json
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatwCDM
import pandas as pd

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
            alpha, beta, h0, Om, wDE = theta
            cosmo = FlatwCDM(H0=h0*100.0, Om0=Om, w0=wDE)


            def D_L(cosmo,z,universe_antiquity):
                if universe_antiquity == 'Low':
                    d_l = (const.c.to(u.km/u.s) / cosmo.H0) * z
                    return d_l.value

                if universe_antiquity == 'Moderate':
                    q0 = cosmo.Om0 / 2.0 - cosmo.Ode0
                    d_l = (const.c.to(u.km/u.s) / cosmo.H0) * (z + ((1/2)*(1-q0)*(z**2)) )    
                    return d_l.value
                
                if universe_antiquity == 'High':
                    z = np.atleast_1d(np.asarray(z, dtype=float))
                    d_l = cosmo.luminosity_distance(z).value
                    return d_l

            def dmu_dz(cosmo,z,universe_antiquity):

                if universe_antiquity == 'Low':
                    return (5.0/np.log(10.0))*(1/z)

                if universe_antiquity == 'Moderate':
                    #Om_z = cosmo.Om(z)
                    #Ode_z = cosmo.Ode(z)
                    #q = (Om_z / 2.0) - Ode_z
                    #return (10* ((q-1)*z-1) ) / (z * np.log(10) * ((q-1)*z-2))

                    q0 = cosmo.Om0 / 2.0 - cosmo.Ode0
                    return (10* ((q0-1)*z-1) ) / (z * np.log(10) * ((q0-1)*z-2))


                
                if universe_antiquity == 'High':
                    z = np.atleast_1d(np.asarray(z, dtype=float))
                    Ez = cosmo.efunc(z)  # E(z) = H(z)/H0
                    # I(z) = integral_0^z dz'/E(z') = (H0/c) * D_C(z)
                    Iz = (cosmo.comoving_distance(z) * cosmo.H0 / const.c.to(u.km/u.s)).to_value(u.dimensionless_unscaled)
                    return np.abs((5.0/np.log(10.0)) * (1.0/(1.0+z) + 1.0/(Ez*Iz)))


            G = DF['origin_id'] == 0.0
            H = DF['origin_id'] != 0.0

            Mum = DF['z_or_mu'] * 0.0
            MumErr = DF['e_z_or_e_mu'] * 0.0

            Mum[G] = DF[G]['z_or_mu']
            MumErr[G] = DF[G]['e_z_or_e_mu']
            
            Mum[H] = 5.0*np.log10(D_L(cosmo, DF[H]['z_or_mu'],z_range)) + 25.0
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
            Om    =  0.0 +  1.0 * u[3]     # 0.2->0.7
            wDE   = -2.0 +  3.0 * u[4]     # -1.0->1.0
            return np.array([alpha, beta, h0, Om, wDE])

        def loglike(theta):
            return self.lnlike(theta = theta,
                               DF = concat_filter_DF,
                               z_range = self.analysis_mode)

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
        parameters = [r"\alpha", r"\beta", r"h",r"\Omega_m",r"w_{DE}"]
        print("Posterior means/stdev:")
        for name, col in zip(parameters, samples.T):
            print(f"{name:>6s}: {col.mean():.3f} ± {col.std():.3f}")


        with open(self.prefix + "params.json", "w") as f:
            json.dump(parameters, f, indent=2)

        #fig = corner.corner(
        #    samples,
        #    labels=[r"$\alpha$", r"$\beta$", r"$h$"],
        #    show_titles=True,
        #    title_fmt=".3f",
        #    quantiles=[0.16, 0.5, 0.84]
        #)
        #fig.savefig(self.prefix + "corner.png", dpi=150, bbox_inches="tight")
        #plt.close(fig)
        #print("Corner guardado en:", self.prefix + "corner.png")



        gds = MCSamples(
            samples=samples,
            names=["alpha", "beta", "h","Om","wDE"],
            labels=[r"\alpha", r"\beta", r"h",r"\Omega_m",r"w_{DE}"] 
        )
        g = plots.getSubplotPlotter()
        g.settings.num_plot_contours = 4
        g.triangle_plot([gds], filled=True, title_limit=1)

        #titulo = g.fig.suptitle(f"{self.main_title}", fontsize=16, y=1.03)

        plt.savefig(
            self.prefix + "triangle_getdist.png", 
            bbox_inches='tight', 
            #bbox_extra_artists=[titulo],
            dpi=1500,
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
    "Compilation2026.csv",
    comment="#",
    index_col=False,
    dtype={"GEHR_id": str},
)

LS_tab = Table.from_pandas(LSdata_df)

data_cut = select_redshift_cut(LS_tab, 7)


Lsig_Ho_sampler(
    data_frame=data_cut,
    distance_estimator_set='GEHR_base.csv',
    estimator_error_kind='sigma_w',
    main_title="L-sigma test: HM 5 parameters",
    folder_name="Five_par_Full_cosmo_Fullsample",
    analysis_mode='High',
    id_prefix='Five_Fc_Fs',
)




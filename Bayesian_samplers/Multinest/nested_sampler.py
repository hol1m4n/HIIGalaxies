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
        self.run_plotter()

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

                if universe_antiquity == 'Moderate':  # Ojo aqui es q0 o q(z)?? revisar despues
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
            n_live_points=2000,
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
        g.settings.num_plot_contours = 4
        g.triangle_plot([gds], filled=True, title_limit=1)

        #titulo = g.fig.suptitle(f"{self.main_title}", fontsize=16, y=1.03)

        plt.savefig(
            self.prefix + "triangle_getdist.png", 
            bbox_inches='tight', 
            #bbox_extra_artists=[titulo],
            dpi=300,
            transparent=True
        )
        plt.close(g.fig)

        print("GetDist triangle en:", self.prefix + "triangle_getdist.png")

        print("\nListo. Archivos en:", self.outdir)


    def L_sigma_plotter(self,DF=None,ax=None,codeX=None,h=None,e_h=None):
        if DF is not None and ax is not None and h is not None and e_h is not None and codeX is not None:
            cosmo = FlatwCDM(H0=h*100.0, Om0=0.3, w0=-1.0)
            set_ = DF[(DF['origin_id'] == codeX)]
            markers = {
                0: "s",   # square
                1: "o",   # circle
                2: "D",   # diamond
                3: "^",   # triangle up
                4: "v",   # triangle down
                5: "x",   # x
                6: "*",   # star
                7: "P",   # filled plus
                8: "h",   # hexagon
            }

            colors = {
                0: "brown",
                1: "blue",
                2: "green",
                3: "red",
                4: "orange",
                5: "gray",
                6: "purple",
                7: "teal",
                8: "black",
            }
            samples = {
                0: "GEHR",
                1: "Local HIIG",
                2: "Liter.Mid-z",
                3: "VLT/X-shooter",
                4: "Keck/MOSFIRE",
                5: "VLT/KMOS",
                6: "JWST/NIRSpec",
                7: "VUDS/VANDELS",
                8: "ALMA+JWST/MIRI"
            }

            if len(set_) != 0 and codeX == 0.0:
                pc_to_cm = 3.08567758e18
                zeta = pc_to_cm 
                logL = np.log10(4*np.pi*(zeta)**2) + ((2*set_['z_or_mu'] +10)/5) + set_['log_f_Hbeta']
                e_logL = np.sqrt(((2/5)*set_['e_z_or_e_mu'])**2 + (set_['e_log_f_Hbeta'])**2)
                logSigma = set_['log_sigma']
                e_logSigma = set_['e_log_sigma']
                ax.errorbar(
                    logSigma,
                    logL,
                    color = colors[codeX],
                    xerr=e_logSigma,
                    yerr=e_logL,
                    fmt=markers[codeX],
                    linestyle="none",
                    label=samples[codeX],
                    alpha=0.8,
                    capsize=5,
                    markersize=10
                )
                return ax            

            if len(set_) != 0 and codeX != 0.0:
                #if mode == 'Local':
                Mpc_to_cm = 3.08567758e24
                #km_to_cm = 100000
                eta = Mpc_to_cm #* km_to_cm
                c = 299792.458 #km/s
                Ez = cosmo.efunc(set_['z_or_mu'])  # E(z) = H(z)/H0
                # I(z) = integral_0^z dz'/E(z') = (H0/c) * D_C(z)
                Iz = (cosmo.comoving_distance(set_['z_or_mu']) * cosmo.H0 / const.c).to_value(u.dimensionless_unscaled)
                logL = np.log10(4*np.pi*(eta)**2*c**2) - (2*np.log10(h*100)) + set_['log_f_Hbeta'] + (2*np.log10(1 + set_['z_or_mu']))  + (2*np.log10(Iz)) 
                e_logL = np.sqrt(((2.0/np.log(10.0)) * (1.0/(1.0+set_['z_or_mu']) + 1.0/(Ez*Iz)) * set_['e_z_or_e_mu'])**2 + (e_h * (-2/(h*np.log(10))))**2   + (set_['e_log_f_Hbeta'])**2)
                logSigma = set_['log_sigma']
                e_logSigma = set_['e_log_sigma']
                ax.errorbar(
                    logSigma,
                    logL,
                    color = colors[codeX],
                    xerr=e_logSigma,
                    yerr=e_logL,
                    fmt=markers[codeX],
                    linestyle="none",
                    label=samples[codeX],
                    alpha=0.8,
                    capsize=5,
                    markersize=10
                )
                #ax.scatter(logSigma, logL, s=set_['z_or_mu'], alpha=0.2, c=set_['z_or_mu'], cmap='viridis', edgecolors='black')
                return ax
            
            else:
                return 0

    def run_plotter(self):
        
        results_path = self.prefix

        analyzer = pymultinest.analyse.Analyzer(
            n_params=3,
            outputfiles_basename=results_path
        )

        stats = analyzer.get_stats()
        alpha_mean,beta_mean,h_mean = stats['modes'][0]['mean']
        alpha_err,beta_err,h_err = stats['modes'][0]['sigma']



        fig, ax = plt.subplots(figsize=(28, 22), dpi = 100)

        for obj in range(9):
            self.L_sigma_plotter(DF=self.data_frame,
                                 ax=ax,
                                 codeX=obj,
                                 h=h_mean,
                                 e_h=h_err)

        # recta ajustada
        xmin, xmax = ax.get_xlim()
        x_line = np.linspace(xmin-0.05, xmax+0.05, 100)
        y_line = beta_mean * x_line + alpha_mean
        ax.plot(x_line, y_line, label=r"Ajuste", color = 'k', linestyle = '--')


        img_path = self.prefix + "triangle_getdist.png"
        img = mpimg.imread(img_path)
        imagebox = OffsetImage(img, zoom=0.35)  # ajusta zoom según tamaño deseado
        ab = AnnotationBbox(
            imagebox,
            (0.97, 0.03),              # posición en coordenadas relativas del eje
            xycoords="axes fraction",  # 0-1 respecto al área del plot
            box_alignment=(1, 0),      # ancla: derecha-abajo
            frameon=False
        )
        ax.add_artist(ab)













        ax.set_xlabel(r"$\log_{10}\, \sigma\ \mathrm{(km\ s^{-1})}$", fontsize = 50)
        ax.set_ylabel(r"$\log_{10}\, L\ \mathrm{(erg\ s^{-1})}$", fontsize = 50)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=20) # Changes x-axis tick label font size
        ax.tick_params(axis='y', labelsize=20) 
        ax.set_title(self.main_title,fontsize = 50,  fontweight='bold')

        ax.legend(ncol= 3,
                loc="upper left",
                title=r"$\alpha$=" + f'{alpha_mean:.3f}' + r"$\pm$" + f"{alpha_err:.3f}" + "\n" +   
                r"$\beta$=" + f'{beta_mean:.3f}' + r"$\pm$" + f"{beta_err:.3f}" + "\n" +  
                r"$h$=" + f'{h_mean:.3f}' + r"$\pm$" + f"{h_err:.3f}",
                title_fontsize=40,
                fontsize = 30)

        #plt.tight_layout()
        fig.savefig(self.prefix + "L-sigmaPlot.png", dpi=150, bbox_inches="tight")
        plt.close()
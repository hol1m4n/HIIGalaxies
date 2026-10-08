import os
import pymultinest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from scipy.stats import norm
from astropy.time import Time
import matplotlib.dates as mdates
import time
import argparse
from astropy.stats import sigma_clip

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





class moduli_stats_calculator:
    def __init__(self,
                 dataset=None,
                 error_selection='random_error',
                 galaxy_name=None,
                 multinest_steps = 10000,
                 bootstrap_steps = 200000):

        self.dataset = dataset
        self.error_selection = error_selection
        self.galaxy_name = galaxy_name
        self.multinest_steps = multinest_steps
        self.bootstrap_steps = bootstrap_steps

        self.input_data_frame = self.check_data()

        self.sigma_clipping()

        if self.input_data_frame is not None:
            self.moduli = np.array(self.input_data_frame['modulus'])
            self.moduli_error = np.array(self.input_data_frame[error_selection])
            self.N_records = len(self.input_data_frame)

            if self.N_records >= 2:
                BOOTSTRAP = self.bootstrap_error()
                NESTED_SAMPLING = self.nestedsampling_error() 

                self.mu_W = self.weighted_average()
                self.mu_B = BOOTSTRAP[0]
                self.mu_L = NESTED_SAMPLING[0]

                self.sigma_W = self.weighted_error()
                self.sigma_B = BOOTSTRAP[1]
                self.sigma_C = self.cochran_error()
                self.sigma_cl_plus = BOOTSTRAP[2][0]
                self.sigma_cl_minus = BOOTSTRAP[2][1]
                self.sigma_L = NESTED_SAMPLING[1]
                self.sigma_Lcorr = NESTED_SAMPLING[2]

                self.bootstrap_dist = BOOTSTRAP[3]
                self.nested_sampling_dist = NESTED_SAMPLING[3]

                self.summary_stats()
                self.summary_plot()
                self.forest_plot()
                self.main_dist_plot()
                self.pdf_arts_plot()

            else:
                self.mu_W = self.weighted_average()
                self.mu_B = np.nan
                self.mu_L = np.nan

                self.sigma_W = self.weighted_error()
                self.sigma_B = np.nan
                self.sigma_C = self.cochran_error()
                self.sigma_cl_plus = np.nan
                self.sigma_cl_minus = np.nan
                self.sigma_L = np.nan
                self.sigma_Lcorr = np.nan            

                self.summary_stats()




    def check_data(self):
        """
        Funcion para checkear que toda la informacion esta bien y separar informacion por galaxia.
        """
        if self.dataset != None:
            data_frame = pd.read_csv(self.dataset)
            column_list = list(data_frame.columns)
            if not self.error_selection in column_list:
                print('Error selection not found in data frame')
                return None
            else:
                galaxy_subframe = data_frame[data_frame['galaxy']==self.galaxy_name]
                if len(galaxy_subframe) == 0:
                    print('Galaxy not found')
                    return None
                else:
                    galaxy_subframe.to_csv(f'{self.dataset.replace('.csv','')}/Data/{self.galaxy_name}_data.csv',index=False)
                    return galaxy_subframe



    def sigma_clipping(self):

        if self.input_data_frame is not None and len(self.input_data_frame) > 3:
            print(f"Running Sigma Clipping for {self.galaxy_name}")

            df = self.input_data_frame.copy()
            # Convertir explícitamente a un arreglo numérico.
            distance_moduli = df["modulus"].to_numpy(dtype=float)

            filtered_data = sigma_clip(
                distance_moduli,
                sigma=3,
                maxiters=5,
                cenfunc="median",
                stdfunc="mad_std",
            )

            # Garantiza una máscara booleana con la forma del arreglo.
            mask_valid = ~np.ma.getmaskarray(filtered_data)
            # Conservar las filas descartadas para revisarlas posteriormente.
            self.sigma_clipping_rejected = df.loc[~mask_valid].copy()
            # Filtrar TODAS las columnas simultáneamente.
            self.input_data_frame = df.loc[mask_valid].copy()
            print(
                f"Originales: {len(df)} | "
                f"Conservados: {mask_valid.sum()} | "
                f"Descartados: {(~mask_valid).sum()}"
            )


    def weighted_average(self):
        """
        Computo del promedio ponderado
        """
        modulus,error = self.moduli,self.moduli_error
        err_sq_inv = 1 / ((error)**2)
        num = np.sum(err_sq_inv * modulus)
        dem = np.sum(err_sq_inv)
        return num/dem

    def weighted_error(self):
        """
        Computo del error ponderado
        """
        error = self.moduli_error
        err_sq_inv = 1 / ((error)**2)
        dem = np.sum(err_sq_inv)
        return 1 / np.sqrt(dem)

    def bootstrap_error(self):
        """
        Computo del promedio, error, distribucion y percentiles mediante bootstrap
        """
        def weighted_average_x_bootstrap(modulus,error):
            modulus,error = np.array(modulus),np.array(error)
            err_sq_inv = 1 / ((error)**2)
            num = np.sum(err_sq_inv * modulus)
            dem = np.sum(err_sq_inv)
            return num/dem
        PATH = f'{self.dataset.replace('.csv','')}/Bootstrap/{self.galaxy_name}/{self.galaxy_name}.txt'
        if os.path.exists(PATH) == True:
            bootstrap_statistics = np.loadtxt(PATH)
            print(f'Reading bootstrap for {self.galaxy_name}\n')
        if os.path.exists(PATH) == False:
            print(f'Running bootstrap for {self.galaxy_name}\n')
            modulus,error = self.moduli,self.moduli_error
            size = len(modulus)
            n_bootstraps = self.bootstrap_steps  # Number of bootstrap iterations
            bootstrap_statistics = []
            for i in range(n_bootstraps):
                # Create a resample with replacement, same size as original data
                bootstrap_sample = np.random.choice(size, size=size, replace=True)
                modulus_sample = modulus[bootstrap_sample]
                error_sample = error[bootstrap_sample]
                sample_mean = weighted_average_x_bootstrap(modulus_sample,error_sample)
                bootstrap_statistics.append(sample_mean)
            bootstrap_statistics = np.array(bootstrap_statistics)
            np.savetxt(fname = PATH, X = bootstrap_statistics)

        N_dist = len(bootstrap_statistics)
        br_mean = np.mean(bootstrap_statistics)
        quad_br = (bootstrap_statistics-br_mean)**2
        lower_bound = np.percentile(bootstrap_statistics, 15.87)
        upper_bound = np.percentile(bootstrap_statistics, 84.13)

        return [br_mean,
            np.sqrt(((1)/(N_dist-1)) * np.sum(quad_br)),
            [br_mean - lower_bound, upper_bound - br_mean],
            bootstrap_statistics]




    def cochran_error(self):
        """
        Computo del error mediante el metodo de Cochran
        """
        def weighted_average_x_cochran(modulus,error):
            err_sq_inv = 1 / ((error)**2)
            num = np.sum(err_sq_inv * modulus)
            dem = np.sum(err_sq_inv)
            return num/dem
        Mu_i,Er_i = self.moduli,self.moduli_error
        n = len(Mu_i)
        w_i = 1 / ((Er_i)**2)
        Mu_w = weighted_average_x_cochran(Mu_i,Er_i)
        w_mean = np.mean(w_i)

        s1 = n / ((n-1) * (np.sum(w_i))**2)
        s2 = (w_i * Mu_i  - (w_mean * Mu_w))**2
        s3 = (w_i - w_mean) * ((w_i*Mu_i) - (w_mean*Mu_w)) 
        s4 = (w_i - w_mean)**2

        return np.sqrt(s1 * (np.sum(s2) - (2 * Mu_w * np.sum(s3)) +    (Mu_w**2 * np.sum(s4)) ) )




    def nestedsampling_error(self):
        """
        Computo del modulo de distancia y error mediante Nested Sampling
        """
        modulus,error = self.moduli,self.moduli_error

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
        
        outdir = os.path.join(self.dataset.replace('.csv',''), f'Nested_sampling/{self.galaxy_name}/')
        os.makedirs(outdir, exist_ok=True)
        prefix = os.path.join(outdir, self.galaxy_name)

        n_dims = 1
        result = pymultinest.solve(
            LogLikelihood=loglike_for_mnest,
            Prior=prior_transform,
            n_dims=n_dims,
            outputfiles_basename=prefix,
            evidence_tolerance=0.5,
            n_live_points=self.multinest_steps,
            multimodal=True,
            verbose=False,
        )

        samples = result["samples"] 
        T_sample = samples.T
        logL = np.array([loglike_for_mnest(s) for s in samples])
        chi2_map = -2.0*np.max(logL)
        N = len(modulus)   # número de datos
        k = n_dims         # parámetros del modelo
        nu = N - k
        chi2_red = chi2_map / nu
        sigma2_corr = T_sample.std() * chi2_red

        samples_x_weight  = pymultinest.analyse.Analyzer(n_params = 1, outputfiles_basename=prefix)
        mu_L = float(samples_x_weight.get_stats()['modes'][0]['mean'][0])

        mu_pi_samples = samples_x_weight.get_equal_weighted_posterior()[:,0]

        return [mu_L,T_sample.std(),sigma2_corr,mu_pi_samples]


    def summary_stats(self):
        """
        Calculadora de estadisticas y resumen de resultados.
        """

        PATH = os.path.join(self.dataset.replace('.csv',''), f'{self.dataset.replace('.csv','')}_recal.csv')

        if not os.path.exists(PATH):
                column_x_emptyDF = ['galaxy','records','mu_W','mu_B','mu_L','sigma_W','sigma_B','sigma_C','sigma_cl+','sigma_cl-','sigma_L','sigma_Lcorr']
                Empty_DF = pd.DataFrame(columns=column_x_emptyDF)
                Empty_DF.to_csv(PATH,index=False)

        DF_comp_recal = pd.read_csv(PATH)
        if not self.galaxy_name in list(DF_comp_recal['galaxy']):
            new_row = pd.DataFrame({
                'galaxy': [self.galaxy_name],
                'records': [self.N_records],
                'mu_W': [self.mu_W],
                'mu_B': [self.mu_B],
                'mu_L': [self.mu_L],
                'sigma_W': [self.sigma_W],
                'sigma_B': [self.sigma_B],
                'sigma_C': [self.sigma_C],
                'sigma_cl+': [self.sigma_cl_plus],
                'sigma_cl-': [self.sigma_cl_minus],
                'sigma_L': [self.sigma_L],
                'sigma_Lcorr': [self.sigma_Lcorr]
            })
            DF_comp_recal = pd.concat([DF_comp_recal, new_row], ignore_index=True)
            DF_comp_recal.to_csv(PATH,index=False)




    def forest_plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        if self.input_data_frame  is not None:
            ax.minorticks_on()


            for a,b,c,d,e in zip([self.mu_W, self.mu_B, self.mu_L], 
                               [self.sigma_W, self.sigma_B, self.sigma_L], 
                               ['black', 'gray', 'brown'], 
                               ['--','-',':'],
                               [r'$\mu_{w}$ = ',r'$\mu_{B}$ = ',r'$\mu_{\mathcal{L}}$ = ']):
                ax.axvline(x=a, 
                           color=c, 
                           linestyle=d, 
                           linewidth=0.8, 
                           alpha=0.7,
                           label=e+f"{a:.4f}"+r' $\pm$ '+f"{b:.4f}")


            def jitter_same_time(t, base_buffer=25):
                """
                Si hay repetidos en t, los separa con pequeños offsets en el eje x.
                base_buffer en unidades de t (si t es JD, esto son días).
                """
                t = np.asarray(t, dtype=float)
                t_out = t.copy()
                # buffer automático: ~0.5% de la mediana del espaciado entre fechas únicas
                if base_buffer is None:
                    tu = np.unique(t)
                    if len(tu) > 1:
                        dt = np.diff(np.sort(tu))
                        base_buffer = 0.005 * np.median(dt)  # ajusta (0.005–0.02) según densidad
                    else:
                        base_buffer = 1.0  # 1 día si todo cae en la misma fecha
                # para cada fecha repetida, asigna offsets simétricos: -k,...,0,...,+k
                for val in np.unique(t):
                    idx = np.where(t == val)[0]
                    if len(idx) > 1:
                        k = len(idx)
                        offsets = (np.arange(k) - (k - 1)/2.0) * base_buffer
                        t_out[idx] = val + offsets
                return t_out

            t = np.array(self.input_data_frame['ads_jd'])  # JD
            x = np.array(self.input_data_frame['modulus'])  # modulus
            s = np.array(self.input_data_frame[self.error_selection])

    
            ordr = np.argsort(t)
            t_sorted = np.asarray(t)[ordr]
            x_sorted = np.asarray(x)[ordr]
            s_sorted = np.asarray(s)[ordr]
    
            t_jit = jitter_same_time(t_sorted)
            t_datetime = Time(t_jit, format='jd').to_datetime()
    
            cmap = plt.cm.tab10   # buen colormap categórico
            author_color = {}     # autor -> color
            plotted_authors = set()


            for u in range(self.N_records):
                author = self.input_data_frame['citation'].iloc[u].replace('&', r'$\&$')
                # Asignar color si es la primera vez que aparece el autor
                if author not in author_color:
                    author_color[author] = cmap(len(author_color) % cmap.N)
                color = author_color[author]
                label = author if author not in plotted_authors else None

                ax.errorbar(
                    x = x_sorted[u],
                    y = t_datetime[u],
                    xerr = s_sorted[u],
                    fmt='o', 
                    markerfacecolor=color, # Puedes cambiar a 'white' si prefieres marcadores huecos
                    markeredgecolor=color,
                    ecolor=color, 
                    elinewidth=1.0,         # Líneas de error más delgadas y elegantes
                    capsize=2,              # Remates de error más sutiles
                    markersize=5,
                    color=color,
                    label=label
                )
                plotted_authors.add(author)


            ax.set_xlabel(r'Distance Modulus, $\mu$', fontsize=16)
            ax.set_ylabel(r'Publication Year', fontsize=16)
            ax.legend(loc='best', ncol=1)
            #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)


            PATH_img = os.path.join(self.dataset.replace('.csv',''), f'Subplots/{self.galaxy_name}/{self.galaxy_name}_forest_plot.png')
            fig = ax.get_figure() 
            fig.savefig(PATH_img, bbox_inches='tight')
        return ax


    def main_dist_plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        if self.input_data_frame  is not None:
            binS = np.histogram_bin_edges(self.bootstrap_dist, bins='doane')
            ax.hist(self.nested_sampling_dist, bins=binS, density=True, 
                    color='#1f77b4', alpha=0.7, label=r'$\mathcal{P} \,( \mu |D)$',histtype='step',linewidth=3)
            ax.hist(self.bootstrap_dist, bins=binS, density=True, 
                    color="#3ab41f", alpha=0.7, label=r'$\mathcal{B} \,( \mu_w)$',histtype='step',linewidth=3)
            ax.set_ylabel(r'PDF [$\%$]', fontsize=15)
            ax.set_xlabel(r'$\mu$', fontsize=15)
            ax.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.5)
            ax.minorticks_on()

            sigmas_loc = [0.6,0.5,0.4,0.3,0.2,0.1]
            sigmas_color = ["#b41f1f","#1f30b4","#b41faa","#090a08","#994607","#0d9edc"]
            sigma_val = [self.sigma_W,self.sigma_B,[self.sigma_cl_plus,self.sigma_cl_minus],self.sigma_C,self.sigma_L,self.sigma_Lcorr]
            label_sigma = [r'$\sigma_{w}$',r'$\sigma_{B}$',r'$\sigma_{cl{\pm}}$',r'$\sigma_{C}$',r'$\sigma_{\mathcal{L}}$',r'$\sigma_{\mathcal{L} corr}$']

            for A, B, C, D in zip(sigmas_loc,sigmas_color,sigma_val,label_sigma):

                line_height = ax.get_ylim()[1] * A

                if isinstance(C, list):
                    lower_limit = self.mu_W - C[1]
                    upper_limit = self.mu_W  + C[0]
                    ax.hlines(y=line_height, 
                        xmin=lower_limit, 
                        xmax=upper_limit,
                        alpha = 0.6,
                        linewidth=5,color = B, label = rf"{D}: $^{{+{C[0]:.3f}}}_{{-{C[1]:.3f}}}$")
                else:
                    lower_limit = self.mu_W - C
                    upper_limit = self.mu_W + C
                    ax.hlines(y=line_height, 
                        xmin=lower_limit, 
                        xmax=upper_limit,
                        alpha = 0.6,
                        linewidth=5,color = B, label=f'{D}: {C:.3f}')


                ax.vlines(x=lower_limit, 
                    ymin=line_height - (ax.get_ylim()[1]*0.02), 
                    ymax=line_height + (ax.get_ylim()[1]*0.02), 
                    alpha = 0.6,linestyles='-',
                    linewidth=3,color = B)
                ax.vlines(x=upper_limit, 
                        ymin=line_height - (ax.get_ylim()[1]*0.02), 
                        ymax=line_height + (ax.get_ylim()[1]*0.02), alpha = 0.6,
                        linewidth=3,color = B)
            ax.legend()


            PATH_img = os.path.join(self.dataset.replace('.csv',''), f'Subplots/{self.galaxy_name}/{self.galaxy_name}_distributions_plot.png')
            fig = ax.get_figure() 
            fig.savefig(PATH_img, bbox_inches='tight')
            return ax


    def pdf_arts_plot(self, ax=None,mode=0):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        if self.input_data_frame  is not None:
            binS = np.histogram_bin_edges(self.bootstrap_dist, bins='doane')
            ax.hist(self.nested_sampling_dist, bins=binS, density=True, 
                    color='#1f77b4', alpha=0.7, label=r'$\mathcal{P} \,( \mu |D)$',histtype='step',linewidth=3)
            ax.hist(self.bootstrap_dist, bins=binS, density=True, 
                    color="#3ab41f", alpha=0.7, label=r'$\mathcal{B} \,( \mu_w)$',histtype='step',linewidth=3)
            ax.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.5)
            ax.minorticks_on()
            ax.set_ylabel(r'PDF [$\%$]', fontsize=15)
            ax.set_xlabel(r'$\mu$', fontsize=15)

            br_hist_ran = np.linspace(binS.min(), binS.max(), 1000)


            left_border = []
            right_border = []

            for a,b in zip(self.moduli,self.moduli_error):
                tmp1 = a - b
                tmp2 = a + b
                left_border.append(tmp1)
                right_border.append(tmp2)

            range_prov = np.linspace(np.min(left_border),
                                    np.max(right_border),
                                    1000)

            cmap = plt.cm.tab10   # buen colormap categórico
            author_color = {}     # autor -> color
            plotted_authors = set()

            sigma = np.array(self.input_data_frame[self.error_selection])
            for u in range(self.N_records):
                author = self.input_data_frame['citation'].iloc[u].replace('&', r'$\&$')
                # Asignar color si es la primera vez que aparece el autor
                if author not in author_color:
                    author_color[author] = cmap(len(author_color) % cmap.N)
                color = author_color[author]
                # Solo agregar label la primera vez
                label = author if author not in plotted_authors else None

                if mode==0:
                    pdf = norm.pdf(br_hist_ran, loc=self.input_data_frame['modulus'].iloc[u], scale=sigma[u])
                    ax.plot(
                        br_hist_ran,
                        pdf,
                        lw=1.7,
                        color=color,
                        alpha=0.7,
                        label=label,
                        linestyle='--'
                    )
                if mode==1:
                    pdf = norm.pdf(range_prov, loc=self.input_data_frame['modulus'].iloc[u], scale=sigma[u]/2)
                    ax.plot(
                        range_prov,
                        pdf,
                        lw=1.7,
                        color=color,
                        alpha=0.7,
                        label=label,
                        linestyle='--'
                    )
                
                plotted_authors.add(author)

            PATH_img = os.path.join(self.dataset.replace('.csv',''), f'Subplots/{self.galaxy_name}/{self.galaxy_name}_articles_PDF_plot.png')
            fig = ax.get_figure() 
            fig.savefig(PATH_img, bbox_inches='tight')

            return ax

    def summary_plot(self):
        if self.input_data_frame  is not None:
            # 1. Crear la figura base
            fig = plt.figure(figsize=(25, 15))

            # 2. Configurar la cuadrícula (GridSpec) de 2 filas y 2 columnas
            gs = gridspec.GridSpec(
                2, 2,
                figure=fig,
                width_ratios=[0.8, 1.0],      # Proporción de ancho igual entre izq y der
                height_ratios=[0.6, 1.0],     # Proporción de altura para los paneles derechos
                wspace=0.15,
                hspace=0.05                   # hspace bajo ayuda a que se vea la conexión del eje X compartido
            )
            gs.update(left=0.05, bottom=0.05, right=0.98)

            ax1 = fig.add_subplot(gs[:, 0])
            self.main_dist_plot(ax=ax1)

            ax2 = fig.add_subplot(gs[0, 1])
            self.pdf_arts_plot(ax=ax2,mode=1)

            ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)
            self.forest_plot(ax=ax3)

            # para evitar que se encimen con el panel inferior
            plt.setp(ax2.get_xticklabels(), visible=False)





            PATH_img = os.path.join(self.dataset.replace('.csv',''), f'{self.galaxy_name}_summary.png')
            fig.savefig(PATH_img, bbox_inches='tight')















#subset_creation(input_df = 'PLRC_general_v1.1.csv',
#                condition = [('metal_correction','==',False),('random_error','>=',0),('citation','!=','Macri et al.2001')],
#                subset_name = 'PLRC_nZ_eR.csv')
#subset_creation(input_df = 'PLRC_general_v1.1.csv',
#                condition = [('metal_correction','==',False),('total_error','>=',0),('citation','!=','Macri et al.2001')],
#                subset_name = 'PLRC_nZ_eT.csv')
#subset_creation(input_df = 'PLRC_general_v1.1.csv',
#                condition = [('metal_correction','==',True),('random_error','>=',0),('citation','!=','Macri et al.2001')],
#                subset_name = 'PLRC_Z_eR.csv')
#subset_creation(input_df = 'PLRC_general_v1.1.csv',
#                condition = [('metal_correction','==',True),('total_error','>=',0),('citation','!=','Macri et al.2001')],
#                subset_name = 'PLRC_Z_eT.csv')
#subset_creation(input_df = 'TRGB_general_v1.1.csv',
#                condition = [('random_error','>=',0),('citation','!=','Macri et al.2001')],
#                subset_name = 'TRGB_eR.csv')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--DATASET', type=str, help='Dataset de donde se toman los archivos (subgrupos)')
    parser.add_argument('--TIPO_E', type=str, default='random_error', help='Tipo de error: random o total')
    parser.add_argument('--GALAXIA', type=str, help='Nombre de la galaxia')

    
    args = parser.parse_args()


    Stats_calc = moduli_stats_calculator(dataset=args.DATASET,
                                        error_selection=args.TIPO_E,
                                        galaxy_name=args.GALAXIA)

    del Stats_calc
    


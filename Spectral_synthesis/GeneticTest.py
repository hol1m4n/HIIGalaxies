import os
import farsight as fs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import synth_reader as sr
from astropy.io import fits
from astropy.table import Table
from joblib import Parallel, delayed

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from astropy.table import join

from sklearn.decomposition import PCA

import numpy as np
from matplotlib import pyplot as plt
import os
from astropy.table import Table
from astropy.cosmology import FlatwCDM
from getdist import plots, MCSamples
import zeus
from zeus import ChainManager
import numpy as np
import scipy.optimize as op
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pysr import PySRRegressor



FADO = Table.read(fits.open('FadoGordonBest_Run.fits')[1])

class Data_get:
    def __init__(self,data,outliers,select):
        self.data = data
        self.outliers = outliers
        self.data_clean = None
        self.select = select #Debe ser una lista de Tuplas
        self.set = None
        self.remove_outliers()
        self.set_selection()


    def remove_outliers(self):
        tmp_dataFrame = self.data
        for x in self.outliers:
            tmp_dataFrame = tmp_dataFrame[tmp_dataFrame['TAB5_INDEX']!=x]
        self.data_clean = tmp_dataFrame 

    def set_selection(self):
        tmp_dataFrame = self.data_clean
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

    def figure_merit(self):
        x = np.array(self.set['ADEV'])
        y = np.array(self.set['chi2_red'])
        punto_ref = np.array([0, 1])
        distancias = np.sqrt((x - punto_ref[0])**2 + (y - punto_ref[1])**2)
        distancia_promedio = np.mean(distancias)

        return distancia_promedio


        
class Plotter(Data_get):
    def __init__(self,data,outliers,select):
        super().__init__(data,outliers,select)
        self.remove_outliers()
        self.set_selection()

DATA = Data_get(FADO,[56],
                [('RED_LAW','==','Gordon et al. 2003 - SMC Bar'),#]).set
                 ('SDSS_SNR','>=',10),
                 ('ADEV','>=',0),('ADEV','<=',25),
                 ('chi2_red','>=',0.5),('chi2_red','<=',1.5)]).set



DATA['lgSFR'] = np.log10(DATA['SFR'])
DATA['lgsSFR'] = np.log10(DATA['sSFR'])
for col in DATA.colnames:
    if np.issubdtype(DATA[col].dtype, np.number):
        mask = (DATA[col] == -999.0) | (DATA[col] == 999.0)
        DATA[col][mask] = np.random.normal(loc=0.1,scale=0.1)

tabla6_2014 = pd.read_csv('~/HIIGalaxies/Spectral_synthesis/Table6_art2014.csv')
def OH_to_Z(OH):
    return 10**(OH - 8.69)

tabla6_2014['Z_convOH'] = OH_to_Z(tabla6_2014['12+logO/H'])

TABLA6_astropy = Table.from_pandas(tabla6_2014)

tabla3_2014 = pd.read_csv('~/HIIGalaxies/Spectral_synthesis/Table3_art2014.csv')

TABLA3_astropy = Table.from_pandas(tabla3_2014)

merger_i = join(DATA, TABLA6_astropy, keys='TAB5_INDEX') # MERGER 1: Tabla 6 del articulo con los datos de FADO

merger_f = join(merger_i, TABLA3_astropy, keys='TAB5_INDEX') # MERGER 1: Tabla 3 del articulo con el merger anterior


# Calculos de proporciones de masas y demas.

burstAge_L = []
burstAge_M = []

Pop_5Myr  = []
Pop_10Myr = []
Pop_30Myr = []
Pop_100Myr = []




def nebular_fraction_at_lambda(wave, F_neb, F_star, lambda0=4861, window=20):
    mask = (wave > lambda0 - window) & (wave < lambda0 + window)

    neb = np.nanmedian(F_neb[mask])
    star = np.nanmedian(F_star[mask])

    return neb / (neb + star)

def nebular_fraction_integrated(wave, F_neb, F_star, lmin=3800, lmax=7000):
    mask = (wave >= lmin) & (wave <= lmax)

    neb_int = np.trapz(F_neb[mask], wave[mask])
    total_int = np.trapz((F_neb[mask] + F_star[mask]), wave[mask])

    return neb_int / total_int


fneb_4020_list = []
fneb_4861_list = []
fneb_opt_list = []



for x in range(len(merger_f)):

    obj = merger_f['TAB5_INDEX'][x]
    obj =str(obj)

    if len(obj)!=3:
        if len(obj) == 2:
            obj = '0' + obj
        else:
            obj = '00' + obj

    spec = fs.Nebulix(home = os.path.expanduser('~') +'/HIIGalaxies/FADO/output_4020/',
               file = f'{obj}GOR1.MILES150_1D.fits', 
               distance = merger_f[merger_f['TAB5_INDEX']>=int(obj)]['Z'][0])
    
    SSPs = spec.population_vector.copy()

    # Convertir masa corregida a unidades físicas
    SSPs['Mcor_jMo'] = (SSPs['Mcor_j'] / 100.0) * 10**spec.fado_ensemble['lg_Mp']

    # Máscara de población joven (t < 10 Myr)
    you_mask = SSPs['logage_j'] <= 7.0

    # Extraer columnas (astropy Table soporta esto directamente)
    x = SSPs['x_j'][you_mask]          # fracción de luz (%)
    m = SSPs['Mcor_jMo'][you_mask]     # masa
    logt = SSPs['logage_j'][you_mask]  # log edad

    # (Opcional) convertir % a fracción
    x = x / 100.0

    # Inicializar valores por defecto
    Io_by_L = np.nan
    Io_by_M = np.nan

    # Promedio ponderado por luz
    if np.sum(x) > 0:
        Io_by_L = np.sum(x * logt) / np.sum(x)

    # Promedio ponderado por masa
    if np.sum(m) > 0:
        Io_by_M = np.sum(m * logt) / np.sum(m)

    # Guardar resultados
    burstAge_L.append(Io_by_L)
    burstAge_M.append(Io_by_M)


    wave = spec.spectrum_bestfit['Lambda']
    F_star = spec.spectrum_bestfit['Flux_ste']
    F_total = spec.spectrum_bestfit['Flux_syn']

    F_neb = F_total - F_star
    F_neb = np.where(F_neb > 0, F_neb, 0)

    fneb_4020 = nebular_fraction_at_lambda(
        wave, F_neb, F_star, lambda0=4020, window=20
    )

    fneb_4861 = nebular_fraction_at_lambda(
        wave, F_neb, F_star, lambda0=4861, window=20
    )

    fneb_opt = nebular_fraction_integrated(
        wave, F_neb, F_star, lmin=3800, lmax=7000
    )

    fneb_4020_list.append(fneb_4020)
    fneb_4861_list.append(fneb_4861)
    fneb_opt_list.append(fneb_opt)

    x_5Myr  = np.sum(SSPs['x_j'][SSPs['logage_j'] <= np.log10(5e6)])
    x_10Myr = np.sum(SSPs['x_j'][SSPs['logage_j'] <= 7.0])
    x_30Myr = np.sum(SSPs['x_j'][SSPs['logage_j'] <= np.log10(3e7)])
    x_100Myr = np.sum(SSPs['x_j'][SSPs['logage_j'] <= 8.0])

    Pop_5Myr.append(x_5Myr)
    Pop_10Myr.append(x_10Myr)
    Pop_30Myr.append(x_30Myr)
    Pop_100Myr.append(x_100Myr)



merger_f['burstAge_L'] = np.array(burstAge_L)
merger_f['burstAge_M'] = np.array(burstAge_M)

merger_f['fneb_4020'] = np.array(fneb_4020_list)
merger_f['fneb_4861'] = np.array(fneb_4861_list)
merger_f['fneb_opt'] = np.array(fneb_opt_list)

merger_f['Pop_5Myr'] = np.array(Pop_5Myr)
merger_f['Pop_10Myr'] = np.array(Pop_10Myr)
merger_f['Pop_30Myr'] = np.array(Pop_30Myr)
merger_f['Pop_100Myr'] = np.array(Pop_100Myr)


merger_f['log_EWHb'] = np.log10(merger_f['EW(Hb)'])


cols = [
    'lgt_av_M',
    'Z_av_M',
    'lg_Mp',
    'logL(Hb)',
    'log_sigma(Hb)',
    'log_sigma([OIII])',
    'Z',
    'lg_QH',
    'log_EWHb','burstAge_L','burstAge_M','fneb_4020','fneb_opt','Pop_5Myr','Pop_10Myr','Pop_30Myr','Pop_100Myr','logR_u','12+logO/H','logSFR','lg_MppAGB','A_neb','T_e','n_e','A_v'
]

df = merger_f.to_pandas()   # o tu tabla ya mergeada

df = df[cols].apply(pd.to_numeric, errors='coerce')
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

data = df

data['OHratio'] = data['12+logO/H']











'''


# Calculamos tus residuos observacionales reales usando tu alfa y beta óptimos:
alpha_best, beta_best = 33.2169717140405, 5.046006556006849 
log_L_obs = data['logL(Hb)']

# Tu variable objetivo (Target) es el residuo que quieres pulverizar:
data['residual'] = log_L_obs - (beta_best * data['log_sigma(Hb)'] + alpha_best)

# Definimos las matrices X (parámetros físicos) e y (residuos)
features = ['lgt_av_M','burstAge_L','burstAge_M','logR_u','OHratio','A_neb','n_e','A_v','Pop_10Myr','fneb_4020','T_e']
X = data[features].values
y = data['residual'].values



model = PySRRegressor(
    niterations=500,
    populations=30,                    # Tus 30 poblaciones configuradas
    
    # --- PARALELIZACIÓN ADAPTADA A TUS 8 CPUS ---
    procs=8,                           # Usa tus 8 núcleos al 100%
    parallelism="multiprocessing",     # Cambiado a multiprocessing para gestionar excedente de poblaciones
    ncyclesperiteration=700,           # Incrementado para que los núcleos no esperen entre ciclos
    # --------------------------------------------
    
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log10", "exp"], 
    constraints={
        'log10': 5, 
        'exp': 5,
        '/': (-1, 9) 
    },
    select_k_features=None,
    progress=True,
    loss="loss(prediction, target) = (prediction - target)^2"
)

print("Iniciando búsqueda genética de ecuaciones...")
model.fit(X, y, variable_names=features)

# ==========================================
# 3. VISUALIZACIÓN DE RESULTADOS
# ==========================================
# Imprime el set de ecuaciones candidatas ordenadas por complejidad y precisión (Score)
print("\n--- Ecuaciones encontradas por PySR ---")
print(model.equations_)

# Puedes exportar el frente de Pareto a un archivo de texto o LaTeX
model.equations_.to_csv("pysr_lsigma_residualsTestsoft.csv")

# Para usar la mejor ecuación en tu código directamente:
best_equation_prediction = model.predict(X)

'''


### Analisis con Hb



# Calculamos tus residuos observacionales reales usando tu alfa y beta óptimos:
alpha_best, beta_best = 33.2169717140405, 5.046006556006849 
log_L_obs = data['logL(Hb)']

# Tu variable objetivo (Target) es el residuo que quieres pulverizar:
data['residual'] = log_L_obs - (beta_best * data['log_sigma(Hb)'] + alpha_best)


'lgt_av_M',
'Z_av_M',
'lg_Mp',
'Z', # Esto es redshift, no metalicidad
'lg_QH',
'log_EWHb',
'burstAge_L',
'burstAge_M',
'fneb_4020',
'fneb_opt',
'Pop_5Myr',
'Pop_10Myr',
'Pop_30Myr',
'Pop_100Myr',
'logR_u',
'OHratio',
'logSFR',
'lg_MppAGB',
'A_neb',
'T_e',
'n_e',
'A_v'



# Definimos las matrices X (parámetros físicos) e y (residuos)
features = ['lgt_av_M',
    'Z_av_M',
    'lg_Mp',
    'Z', # Esto es redshift, no metalicidad
    'lg_QH',
    'log_EWHb',
    'burstAge_L',
    'burstAge_M',
    'fneb_4020',
    'fneb_opt',
    'Pop_5Myr',
    'Pop_10Myr',
    'Pop_30Myr',
    'Pop_100Myr',
    'logR_u',
    'OHratio',
    'logSFR',
    'lg_MppAGB',
    'A_neb',
    'T_e',
    'n_e',
    'A_v'
    ]

X = data[features].values
y = data['residual'].values



model = PySRRegressor(
    niterations=1000,
    populations=50,                    # Tus 50 poblaciones configuradas
    
    # --- PARALELIZACIÓN ADAPTADA A TUS 8 CPUS ---
    procs=8,                           # Usa tus 8 núcleos al 100%
    parallelism="multiprocessing",     # Cambiado a multiprocessing para gestionar excedente de poblaciones
    ncyclesperiteration=1000,          # Mayor carga de trabajo por ciclo debido a las 50 poblaciones
    # --------------------------------------------
    
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log10", "exp"], 
    constraints={
        'log10': 5, 
        'exp': 5,
        '/': (-1, 9)
    },
    select_k_features=None,
    progress=True,
    loss="loss(prediction, target) = (prediction - target)^2"
)

print("Iniciando búsqueda genética de ecuaciones...")
model.fit(X, y, variable_names=features)

# ==========================================
# 3. VISUALIZACIÓN DE RESULTADOS
# ==========================================
# Imprime el set de ecuaciones candidatas ordenadas por complejidad y precisión (Score)
print("\n--- Ecuaciones encontradas por PySR ---")
print(model.equations_)

# Puedes exportar el frente de Pareto a un archivo de texto o LaTeX
model.equations_.to_csv("pysr_lsigma_HardHb.csv")

# Para usar la mejor ecuación en tu código directamente:
best_equation_prediction = model.predict(X)


# Calculamos tus residuos observacionales reales usando tu alfa y beta óptimos:
alpha_best, beta_best = 33.2169717140405, 5.046006556006849 
log_L_obs = data['logL(Hb)']

# Tu variable objetivo (Target) es el residuo que quieres pulverizar:
data['residual'] = log_L_obs - (beta_best * data['log_sigma([OIII])'] + alpha_best)

# Definimos las matrices X (parámetros físicos) e y (residuos)
features = ['lgt_av_M',
    'Z_av_M',
    'lg_Mp',
    'Z', # Esto es redshift, no metalicidad
    'lg_QH',
    'log_EWHb',
    'burstAge_L',
    'burstAge_M',
    'fneb_4020',
    'fneb_opt',
    'Pop_5Myr',
    'Pop_10Myr',
    'Pop_30Myr',
    'Pop_100Myr',
    'logR_u',
    'OHratio',
    'logSFR',
    'lg_MppAGB',
    'A_neb',
    'T_e',
    'n_e',
    'A_v'
    ]

X = data[features].values
y = data['residual'].values



model = PySRRegressor(
    niterations=1000,
    populations=50,                    # Tus 50 poblaciones configuradas
    
    # --- PARALELIZACIÓN ADAPTADA A TUS 8 CPUS ---
    procs=8,                           # Usa tus 8 núcleos al 100%
    parallelism="multiprocessing",     # Cambiado a multiprocessing para gestionar excedente de poblaciones
    ncyclesperiteration=1000,          # Mayor carga de trabajo por ciclo debido a las 50 poblaciones
    # --------------------------------------------
    
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log10", "exp"], 
    constraints={
        'log10': 5, 
        'exp': 5,
        '/': (-1, 9)
    },
    select_k_features=None,
    progress=True,
    loss="loss(prediction, target) = (prediction - target)^2"
)

print("Iniciando búsqueda genética de ecuaciones...")
model.fit(X, y, variable_names=features)

# ==========================================
# 3. VISUALIZACIÓN DE RESULTADOS
# ==========================================
# Imprime el set de ecuaciones candidatas ordenadas por complejidad y precisión (Score)
print("\n--- Ecuaciones encontradas por PySR ---")
print(model.equations_)

# Puedes exportar el frente de Pareto a un archivo de texto o LaTeX
model.equations_.to_csv("pysr_lsigma_HardOIII.csv")

# Para usar la mejor ecuación en tu código directamente:
best_equation_prediction = model.predict(X)




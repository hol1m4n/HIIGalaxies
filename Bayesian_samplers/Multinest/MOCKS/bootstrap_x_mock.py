import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def weighted_average_x_bootstrap(var, error):
    """Calcula el promedio ponderado."""
    var, error = np.array(var), np.array(error)
    err_sq_inv = 1 / (error ** 2)
    num = np.sum(err_sq_inv * var)
    dem = np.sum(err_sq_inv)
    return num / dem

def process_single_mock(ML, AG, Main_path):
    """Procesa un único par de Mock_lbl y anchors_group."""
    folder_path = os.path.expanduser(os.path.join(Main_path, ML, AG))
    csv_path = os.path.join(folder_path, 'Results.csv')
    
    if not os.path.exists(csv_path):
        print(f"Archivo no encontrado: {csv_path}")
        return

    print(f'Running bootstrap for {folder_path}')
    df = pd.read_csv(csv_path)

    h = df['valor'].values
    h_e = df['error'].values
    size = len(h)
    n_bootstraps = 200000

    # Bucle de bootstrap optimizado (ejecutado secuencialmente por archivo en cada hilo/proceso)
    bootstrap_statistics = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        bootstrap_sample = np.random.choice(size, size=size, replace=True)
        modulus_sample = h[bootstrap_sample]
        error_sample = h_e[bootstrap_sample]
        # Corrección: usar las muestras remuestreadas
        bootstrap_statistics[i] = weighted_average_x_bootstrap(modulus_sample, error_sample)
        bootstrap_statistics[i] = np.mean(modulus_sample)

    # Estadísticas y percentiles
    br_mean = np.mean(bootstrap_statistics)
    lower_bound = np.percentile(bootstrap_statistics, 15.87)
    upper_bound = np.percentile(bootstrap_statistics, 84.13)

    # Guardar gráfico
    binS = np.histogram_bin_edges(bootstrap_statistics, bins='doane')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    ax.hist(bootstrap_statistics, bins=binS, density=True, 
            color="#3ab41f", alpha=0.7, label=r'$\mathcal{B} \,( h)$= ' + f'{br_mean}', histtype='step', linewidth=3)
    ax.set_ylabel(r'PDF [$\%$]', fontsize=20)
    ax.set_xlabel(r'$h$', fontsize=15)
    ax.grid(True, which="both", ls=":", color = 'gray', linewidth = 0.5)
    ax.minorticks_on()
    ax.legend()
    
    os.makedirs(folder_path, exist_ok=True)
    fig.savefig(os.path.join(folder_path, 'Bootstrap.png'))
    plt.close(fig)
    
    print(f'Completado: {folder_path}')

def bootstrap_mock_parallel():
    Main_path = os.path.join('~', 'HIIGalaxies/Bayesian_samplers/Multinest/MOCKS/Results/')
    Mock_lbls = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J']
    anchors_groups = ['Joint_eR', 'PLRC_all_eR', 'PLRC_nZ_eR', 'PLRC_nZ_eT', 'PLRC_Z_eR', 'PLRC_Z_eT', 'TRGB_eR']

    # Creamos todas las combinaciones posibles de (ML, AG)
    tasks = [(ml, ag) for ml in Mock_lbls for ag in anchors_groups]

    # Ejecutamos en paralelo usando todos los núcleos disponibles (n_jobs=-1)
    Parallel(n_jobs=6)(
        delayed(process_single_mock)(ml, ag, Main_path) for ml, ag in tasks
    )

if __name__ == '__main__':
    bootstrap_mock_parallel()
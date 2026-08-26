import json
import numpy as np
import pymultinest
import time

inicio = time.time()


modulus = np.array([29.49,29.44,29.51,29.38,29.4,29.4,29.4,29.43,29.44,29.48,29.53,29.65,29.53,29.32,29.42,29.47,29.49,29.51,29.64,29.45,29.31,29.38,29.352,29.37,29.397,29.401,29.44,29.49,29.63,29.1,29.12,29.27,29.28,29.28,29.22,29.28,29.12,29.35,29.28,29.15,29.18,29.4,29.49,29.28,29.345,29.24,29.18,29.4])
error = np.array([0.12,0.07,0.07,0.07,0.06,0.1,0.1,0.1,0.06,0.09,0.07,0.09,0.1,0.14,0.09,0.09,0.14,0.14,0.09,0.07,0.06,0.06,0.057,0.056,0.058,0.058,0.056,0.06,0.05,0.04,0.04,0.04,0.05,0.08,0.03,0.1,0.23,0.12,0.03,0.21,0.21,0.13,0.21,0.05,0.004,0.08,0.23,0.08])

#modulus = np.array([29.49,29.44,29.51,29.38,29.4])
#error = np.array([0.12,0.07,0.07,0.07,0.06])


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

parameters = ["modulus"]
n_dims = 1
result  = pymultinest.solve(
    LogLikelihood=loglike_for_mnest,
    Prior=prior_transform,
    n_dims=n_dims,
    outputfiles_basename='out/',
    evidence_tolerance=0.5,
    n_live_points=1000,
    multimodal=True,
    verbose=True,
)

json.dump(parameters, open('out/params.json', 'w'))

print(f"\nLog-Evidence (ln Z): {result['logZ']:.2f} ± {result['logZerr']:.2f}")

# mpirun -np 15 python3 ex_mydata.py

fin = time.time()
# Calcular la duración
duracion = fin - inicio
print(f"Tiempo transcurrido: {duracion:.5f} segundos")
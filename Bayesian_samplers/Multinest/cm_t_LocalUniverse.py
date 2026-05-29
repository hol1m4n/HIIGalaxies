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

def group_reader(DF,group):
    DF = DF.to_pandas()
    df_group = pd.read_csv(group)
    galaxies_in_group = np.unique(df_group['Galaxia'])
    filtro = DF[DF['GEHR_id'].isin(galaxies_in_group)]
    new_DF = Table.from_pandas(filtro)
    return new_DF

pc_to_cm = 3.08567758e18

LSdata_df = pd.read_csv('Compilation2026.csv',comment='#',index_col=False, dtype={'GEHR_id': str})
LS_tab = Table.from_pandas(LSdata_df)

initial_select = LS_tab[(LS_tab['origin_id'] == 0.0) | ((LS_tab['origin_id'] != 0.0) & (LS_tab['z_or_mu'] <= 0.10))]


local_universe = group_reader(initial_select, group='cm_t.csv') #### Grupos aqui

Encabezado = "Ho_ConMetalicidad_eTotales"





def lnlike_LU(theta,DF):
    alpha, beta, h0 = theta
    cosmo = FlatwCDM(H0=h0*100.0, Om0=0.3, w0=-1.0)

    G = local_universe['origin_id'] == 0.0
    H = local_universe['origin_id'] != 0.0


    Mum = DF['z_or_mu'] * 0.0
    MumErr = DF['e_z_or_e_mu'] * 0.0

    Mum[G] = DF[G]['z_or_mu']
    MumErr[G] = DF[G]['e_z_or_e_mu']
    Mum[H] = 5.0*np.log10(cosmo.luminosity_distance(DF[H]['z_or_mu']).value) + 25.0
    MumErr[H] = (5.0/np.log(10.0))*(DF[H]['e_z_or_e_mu']/DF[H]['z_or_mu'])

    Mu = 2.5*(beta*DF['log_sigma'] + alpha) - 2.5*DF['log_f_Hbeta'] - 100.19477738511641
    MuErr = 2.5*np.sqrt((DF['e_log_f_Hbeta'])**2 + beta**2*(DF['e_log_sigma'])**2)

    R = (Mu - Mum)
    W = 1.0/(MuErr**2 + MumErr**2)

    xsq = np.sum(R**2 * W)
    llq = -0.5*xsq
    return llq

def prior_transform(u):
    alpha = 20.0 + 20.0 * u[0]     # 20->40
    beta  =  0.0 + 10.0 * u[1]     # 0->10
    h0    =  0.5 +  0.5 * u[2]     # 0.5->1.0
    return np.array([alpha, beta, h0])


def nll(theta):
    return -lnlike_LU(theta,local_universe)

res = op.minimize(nll, x0=[32.0, 5.0, 0.75], method="Nelder-Mead")
print("ML guess:", res.x)


outdir = os.path.join(os.path.expanduser('~/HIIGalaxies/Bayesian_samplers/Multinest'), f"{Encabezado}")
os.makedirs(outdir, exist_ok=True)
prefix = os.path.join(outdir, "hii_")

def loglike_for_LU(theta):
    return lnlike_LU(theta,local_universe)


n_dims = 3
result = pymultinest.solve(
    LogLikelihood=loglike_for_LU,
    Prior=prior_transform,
    n_dims=n_dims,
    outputfiles_basename=prefix,
    evidence_tolerance=0.5,
    n_live_points=1500,
    multimodal=True,
    verbose=True
)


print("\nEvidence lnZ = %.3f ± %.3f" % (result["logZ"], result["logZerr"]))


samples = result["samples"]   # shape = (Nsamples, 3)
parameters = [r"\alpha", r"\beta", r"h"]
print("Posterior means/stdev:")
for name, col in zip(parameters, samples.T):
    print(f"{name:>6s}: {col.mean():.3f} ± {col.std():.3f}")


with open(prefix + "params.json", "w") as f:
    json.dump(parameters, f, indent=2)

fig = corner.corner(
    samples,
    labels=[r"$\alpha$", r"$\beta$", r"$h$"],
    show_titles=True,
    title_fmt=".3f",
    quantiles=[0.16, 0.5, 0.84]
)
fig.savefig(prefix + "corner.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Corner guardado en:", prefix + "corner.png")



gds = MCSamples(
    samples=samples,
    names=["alpha", "beta", "h"],
    labels=[r"\alpha", r"\beta", r"h"]  
)
g = plots.getSubplotPlotter()
g.settings.num_plot_contours = 2
g.triangle_plot([gds], filled=True, title_limit=1)
g.fig.suptitle(f"Universo Local: {Encabezado}", fontsize=16, y=1.03)
g.export(prefix + "triangle_getdist.png")
print("GetDist triangle en:", prefix + "triangle_getdist.png")

print("\nListo. Archivos en:", outdir)




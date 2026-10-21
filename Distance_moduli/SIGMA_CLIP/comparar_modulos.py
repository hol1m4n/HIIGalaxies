#!/usr/bin/env python3
"""Comparaciones pareadas de modulos de distancia (mag).

Instalacion: python -m pip install numpy pandas matplotlib scipy
Ejemplo: python comparar_modulos.py /ruta/datos --pattern '*/mediciones.csv'
Patron por defecto: */*.csv (una tabla por carpeta inmediata).
Cada CSV: galaxia,mu_W,sigma_W. sigma_W debe ser incertidumbre 1 sigma.

Salidas: megaplot.png, resumen_comparaciones.csv, detalle_galaxias.csv,
         inventario.csv, metodologia.txt.
No elimina outliers. Valida errores, nombres vacios y galaxias duplicadas.
Normaliza nombres solo quitando espacios exteriores y pasando a mayusculas;
no identifica automaticamente alias como M33 y NGC0598.

Modelo: d=mu_B-mu_A, v=sigma_A^2+sigma_B^2-2*rho*sigma_A*sigma_B.
Se asumen errores gaussianos y galaxias independientes. rho es una opcion
simplificada de correlacion A/B para la misma galaxia, NO una matriz de
covarianza completa. No usar significancias como definitivas si hay errores
sistematicos compartidos entre galaxias, errores estimados poco fiables o
muestras fuertemente solapadas con covarianzas desconocidas.

Pruebas globales: igualdad 1:1 (chi2, N gl), sesgo constante (normal),
dispersion residual tras ajustar sesgo (chi2, N-1 gl). Los p globales se
corrigen juntos con Holm, sobre todas las parejas y los tres contrastes.
Pruebas individuales: diferencia frente a identidad y residuo respecto al
sesgo estimado SIN la galaxia evaluada (leave-one-out, N>=3). Se corrigen
juntas con Holm sobre todas las parejas/galaxias y ambos contrastes.
Las dos familias (global e individual) se controlan por separado.
IC95 del sesgo: marginal, sin correccion simultanea, con varianzas conocidas.
Significancia no demuestra una diferencia fisica fundamental ni causalidad.
Ausencia de significancia no establece equivalencia.
"""

# python comparar_modulos.py /home/hollman/HIIGalaxies/Distance_moduli/MAIN --pattern "*/*.csv" --out resultados_comparacion --ncols 3 --dpi 100

from pathlib import Path
from itertools import combinations
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2


def holm(values):
    """p ajustados Holm; conserva NaN para contrastes no evaluables."""
    p = np.asarray(values, dtype=float)
    out = np.full(p.shape, np.nan)
    ids = np.flatnonzero(np.isfinite(p))
    order = ids[np.argsort(p[ids])]
    out[order] = np.minimum(1., np.maximum.accumulate(
        (len(order) - np.arange(len(order))) * p[order]))
    return out


def load_table(path):
    df = pd.read_csv(path, sep=None, engine='python', dtype=str)
    df.columns = df.columns.str.strip().str.lstrip('\ufeff')
    required = ['galaxy', 'mu_W', 'sigma_W']
    if not set(required).issubset(df.columns):
        raise ValueError(f'{path}: faltan columnas {required}')
    df = df[required].copy()
    df['galaxy'] = df.galaxy.str.strip().str.upper()
    if df.galaxy.isna().any() or df.galaxy.eq('').any():
        raise ValueError(f'{path}: nombres de galaxia vacios')
    if df.galaxy.duplicated().any():
        raise ValueError(f'{path}: galaxias duplicadas; resolver antes de comparar')
    for c in ['mu_W', 'sigma_W']:
        df[c] = pd.to_numeric(df[c], errors='raise')
    if not np.isfinite(df[['mu_W', 'sigma_W']].to_numpy()).all():
        raise ValueError(f'{path}: valores no finitos')
    if (df.sigma_W <= 0).any() or df.empty:
        raise ValueError(f'{path}: tabla vacia o incertidumbres no positivas')
    return df


def compare(a, b, name_a, name_b, rho):
    t = a.merge(b, on='galaxy', suffixes=('_A', '_B'), validate='one_to_one')
    n = len(t)
    row = dict(A=name_a, B=name_b, N=n, solo_A=len(a)-n, solo_B=len(b)-n,
               sesgo=np.nan, e_sesgo=np.nan, IC95_inf=np.nan, IC95_sup=np.nan,
               chi2_identidad=np.nan, gl_identidad=n, p_identidad=np.nan,
               p_sesgo=np.nan, chi2_residual=np.nan, gl_residual=max(n-1, 0),
               chi2_reducido_residual=np.nan, p_dispersion=np.nan)
    t.insert(0, 'B', name_b)
    t.insert(0, 'A', name_a)
    if not n:
        return row, t
    d = (t.mu_W_B - t.mu_W_A).to_numpy()
    v = (t.sigma_W_A**2 + t.sigma_W_B**2 -
         2*rho*t.sigma_W_A*t.sigma_W_B).to_numpy()
    if (v <= 0).any() or not np.isfinite(v).all():
        raise ValueError('Varianzas de diferencias no validas')
    w = 1/v
    bias = np.sum(w*d)/np.sum(w)
    err = np.sqrt(1/np.sum(w))
    q0 = np.sum(d*d/v)
    qr = np.sum((d-bias)**2/v)
    row.update(sesgo=bias, e_sesgo=err, IC95_inf=bias-norm.ppf(.975)*err,
               IC95_sup=bias+norm.ppf(.975)*err, chi2_identidad=q0,
               p_identidad=chi2.sf(q0, n), p_sesgo=2*norm.sf(abs(bias/err)),
               chi2_residual=qr if n>1 else np.nan,
               chi2_reducido_residual=qr/(n-1) if n>1 else np.nan,
               p_dispersion=chi2.sf(qr, n-1) if n>1 else np.nan)
    t['delta_mu'] = d
    t['sigma_delta'] = np.sqrt(v)
    t['z_identidad'] = d/np.sqrt(v)
    t['p_ind_identidad'] = 2*norm.sf(abs(t.z_identidad))
    t['z_loo'] = np.nan
    if n >= 3:
        # Exclusiones explicitas evitan cancelacion numerica al restar pesos.
        for i in range(n):
            keep = np.arange(n) != i
            other_bias = np.sum(w[keep]*d[keep])/np.sum(w[keep])
            t.loc[i, 'z_loo'] = (d[i]-other_bias)/np.sqrt(v[i]+1/np.sum(w[keep]))
    t['p_ind_loo'] = 2*norm.sf(abs(t.z_loo))
    return row, t


def correct_family(df, columns, alpha):
    values = df[columns].to_numpy()
    adjusted = holm(values.ravel()).reshape(values.shape)
    for j, col in enumerate(columns):
        df[col+'_holm'] = adjusted[:, j]
        # pd.NA distingue prueba no evaluable de no significativa.
        df[col+'_significativo'] = pd.array(
            [pd.NA if not np.isfinite(p) else p < alpha for p in adjusted[:, j]],
            dtype='boolean')


def plot_all(summary, detail, output, ncols, dpi):
    ncols = min(ncols, len(summary))
    nrows = math.ceil(len(summary)/ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2*ncols, 5.4*nrows),
                             squeeze=False, constrained_layout=True)
    for ax, (_, r) in zip(axes.flat, summary.iterrows()):
        t = detail[(detail.A == r.A) & (detail.B == r.B)]
        ax.set_title(f'{r.A} vs {r.B} | N={int(r.N)}', fontsize=10, wrap=True)
        ax.set_xlabel(f'{r.A}: $\u03bc_W$ [mag]')
        ax.set_ylabel(f'{r.B}: $\u03bc_W$ [mag]')
        ax.grid(alpha=.2)
        if t.empty:
            ax.text(.5, .5, 'Sin galaxias comunes', ha='center', transform=ax.transAxes)
            continue
        x, y, sx, sy = [t[c].to_numpy() for c in
                        ['mu_W_A', 'mu_W_B', 'sigma_W_A', 'sigma_W_B']]
        lo, hi = min((x-sx).min(), (y-sy).min()), max((x+sx).max(), (y+sy).max())
        pad = max(.08*(hi-lo), .05)
        limits = [lo-pad, hi+pad]
        ax.errorbar(x, y, xerr=sx, yerr=sy, fmt='o', ms=4, color='#2463a0',
                    ecolor='#94a8ba', capsize=2, alpha=.85)
        ax.plot(limits, limits, '--', color='black', lw=1, label='1:1')
        ax.plot(limits, np.array(limits)+r.sesgo, color='#db8d23', lw=1,
                label='1:1 + sesgo')
        bad = t.p_ind_identidad_significativo.fillna(False).to_numpy(dtype=bool)
        out = t.p_ind_loo_significativo.fillna(False).to_numpy(dtype=bool)
        ax.scatter(x[bad], y[bad], s=65, facecolors='none', edgecolors='crimson',
                   label='Discrepancia 1:1', zorder=4)
        ax.scatter(x[out], y[out], s=65, marker='x', color='purple',
                   label='Atipico tras sesgo', zorder=5)
        for k in np.flatnonzero(bad | out):
            ax.annotate(t.galaxy.iloc[k], (x[k], y[k]), xytext=(4, 4),
                        textcoords='offset points', fontsize=7)
        ax.set(xlim=limits, ylim=limits)
        ax.set_aspect('equal', adjustable='box')
        ax.text(.03, .97, f'Sesgo B-A = {r.sesgo:+.3f} +/- {r.e_sesgo:.3f} mag\n'
                f'p(Holm): 1:1={r.p_identidad_holm:.2g}; '
                f'sesgo={r.p_sesgo_holm:.2g}\n'
                f'dispersion={r.p_dispersion_holm:.2g}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        ax.legend(loc='lower right', fontsize=7)
    for ax in list(axes.flat)[len(summary):]:
        ax.set_visible(False)
    fig.suptitle('Comparaciones pareadas de modulos de distancia', fontsize=15)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', type=Path, help='Carpeta que contiene los conjuntos')
    p.add_argument('--pattern', default='*/*.csv', help='Patron glob relativo a root')
    p.add_argument('--out', type=Path, default=Path('comparacion_resultados'))
    p.add_argument('--alpha', type=float, default=.05)
    p.add_argument('--rho', type=float, default=0., help='Correlacion A/B misma galaxia; default 0')
    p.add_argument('--ncols', type=int, default=3)
    p.add_argument('--dpi', type=int, default=180)
    args = p.parse_args()
    if not args.root.is_dir():
        p.error('root debe ser una carpeta existente')
    if not 0 < args.alpha < 1 or not -1 < args.rho < 1:
        p.error('Se requiere 0<alpha<1 y -1<rho<1')
    if args.ncols < 1 or args.dpi < 1:
        p.error('ncols y dpi deben ser positivos')
    paths = sorted(f for f in args.root.glob(args.pattern)
                   if f.is_file() and args.out.resolve() not in f.resolve().parents)
    if len(paths) < 2:
        p.error('Se necesitan al menos dos CSV; revisa root y --pattern')
    parents = [f.parent for f in paths]
    if len(set(parents)) != len(parents):
        p.error('Hay varios CSV en una carpeta: especifica --pattern, por ejemplo */mediciones.csv')
    tables = {str(f.parent.relative_to(args.root)): load_table(f) for f in paths}
    rows, details = [], []
    for a, b in combinations(tables, 2):
        row, t = compare(tables[a], tables[b], a, b, args.rho)
        rows.append(row)
        if len(t):
            details.append(t)
    summary = pd.DataFrame(rows)
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame(
        columns=['A', 'B', 'galaxy', 'p_ind_identidad', 'p_ind_loo'])
    correct_family(summary, ['p_identidad', 'p_sesgo', 'p_dispersion'], args.alpha)
    correct_family(detail, ['p_ind_identidad', 'p_ind_loo'], args.alpha)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out/'resumen_comparaciones.csv', index=False)
    detail.to_csv(args.out/'detalle_galaxias.csv', index=False)
    pd.DataFrame([dict(conjunto=k, archivo=str(f.resolve()), N=len(tables[k]))
                  for k, f in zip(tables, paths)]).to_csv(args.out/'inventario.csv', index=False)
    (args.out/'metodologia.txt').write_text(__doc__ +
        f'\nEjecucion: alpha={args.alpha}, rho={args.rho}, pattern={args.pattern}\n', encoding='utf-8')
    plot_all(summary, detail, args.out/'megaplot.png', args.ncols, args.dpi)
    print(summary[['A', 'B', 'N', 'sesgo', 'e_sesgo', 'p_identidad_holm',
                   'p_sesgo_holm', 'p_dispersion_holm']].to_string(index=False))
    print(f'\nArchivos guardados en: {args.out.resolve()}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Diferencias A-B por galaxia en un solo forest plot.

Instalar: python -m pip install numpy pandas matplotlib
Ejemplo:
 python delta_galaxias.py /ruta/A /ruta/B --value mu_W --error sigma_W

Se elige exactamente un CSV por carpeta (por defecto *.csv). Si hay varios,
indicar --pattern-a y --pattern-b, por ejemplo 'mediciones.csv'. No combina
CSV ni promedia duplicados. Las carpetas y archivos elegidos deben ser distintos.
Se empareja por galaxy, quitando espacios exteriores y pasando a mayusculas.
No se resuelven alias astronomicos. Galaxias sin pareja se reportan, no se
rellenan con cero. Datos invalidos o duplicados causan un error explicito.

sigma_delta = sqrt(sigma_A^2 + sigma_B^2 - 2*rho*sigma_A*sigma_B).
Los errores de entrada deben ser desviaciones estandar simetricas (1 sigma).
--rho es una correlacion comun A/B para cada galaxia; default 0 (independencia).
No representa covarianzas completas ni estima correlaciones a partir de datos.

Barras: por defecto +-1 sigma_delta; --nsigma 1.96 muestra aproximadamente
intervalos marginales del 95% bajo el modelo normal. El color rojo se basa en
p bilateral normal corregido con Holm sobre TODAS las galaxias emparejadas
(familia de pruebas de esta ejecucion), con alpha=0.05. Cambiar nsigma no
cambia los p ni el criterio de color. Holm admite dependencia entre pruebas,
pero requiere p individuales validos (errores y rho correctamente especificados).
No interpreta una diferencia significativa como evidencia causal o fisica.

Salidas: delta_galaxias.png, delta_galaxias.pdf, diferencias.csv,
sin_pareja.csv, configuracion.json. No realiza pruebas globales de sesgo.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def choose_csv(folder, pattern):
    if not folder.is_dir():
        raise ValueError(f'No existe la carpeta: {folder}')
    paths = sorted(f for f in folder.glob(pattern) if f.is_file())
    if len(paths) != 1:
        raise ValueError(f'{folder}: se encontraron {len(paths)} archivos con {pattern!r}. '
                         'Selecciona exactamente uno mediante --pattern-a o --pattern-b.')
    return paths[0].resolve()


def read_table(path, value, error, galaxy, sep):
    df = pd.read_csv(path, sep=sep, engine='python', dtype=str,
                     keep_default_na=False, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError(f'{path}: encabezados duplicados')
    needed = {galaxy, value, error}
    if not needed.issubset(df.columns):
        raise ValueError(f'{path}: faltan columnas {sorted(needed-set(df.columns))}')
    t = pd.DataFrame({'galaxy':df[galaxy].str.strip().str.upper(),
                      'value':pd.to_numeric(df[value], errors='coerce'),
                      'sigma':pd.to_numeric(df[error], errors='coerce')})
    if t.empty or t.galaxy.eq('').any():
        raise ValueError(f'{path}: tabla vacia o nombres vacios')
    if t.galaxy.duplicated().any():
        names = t.loc[t.galaxy.duplicated(keep=False), 'galaxy'].unique()
        raise ValueError(f'{path}: galaxias duplicadas {list(names)}; resolver antes de comparar')
    if not np.isfinite(t[['value','sigma']].to_numpy()).all() or (t.sigma<=0).any():
        raise ValueError(f'{path}: valores no finitos o errores no positivos')
    return t


def compare(a, b, rho, alpha, nsigma):
    t = a.merge(b, on='galaxy', how='inner', suffixes=('_A','_B'), validate='one_to_one')
    if t.empty:
        raise ValueError('No hay galaxias compartidas entre los dos archivos')
    t['delta'] = t.value_A-t.value_B
    # Forma equivalente y estable, especialmente cuando rho es cercano a 1.
    variance = (t.sigma_A-t.sigma_B)**2 + 2*(1-rho)*t.sigma_A*t.sigma_B
    t['sigma_delta'] = np.sqrt(variance)
    if not np.isfinite(t[['delta','sigma_delta']].to_numpy()).all() or (t.sigma_delta<=0).any():
        raise ValueError('Diferencias o varianzas no validas')
    t['z'] = t.delta/t.sigma_delta
    t['p_two_sided'] = [math.erfc(abs(z)/math.sqrt(2)) for z in t.z]
    order = np.argsort(t.p_two_sided.to_numpy())
    adjusted = np.empty(len(t))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        (len(t)-np.arange(len(t)))*t.p_two_sided.to_numpy()[order]))
    t['p_holm'] = adjusted
    t['significant_holm'] = t.p_holm < alpha
    t['lower_shown'] = t.delta-nsigma*t.sigma_delta
    t['upper_shown'] = t.delta+nsigma*t.sigma_delta
    t['interval_excludes_zero'] = (t.lower_shown>0) | (t.upper_shown<0)
    return t


def plot(t, args, label_a, label_b):
    n = len(t)
    fig, (ax, numbers) = plt.subplots(1, 2, figsize=(11, max(3.8, .38*n+2.3)),
                                     sharey=True, layout='constrained',
                                     gridspec_kw={'width_ratios':[3.6,2.4]})
    for i, row in t.iterrows():
        for axis in (ax, numbers):
            if i%2==0:
                axis.axhspan(i-.5, i+.5, color='#f1f4f8', zorder=0)
        color = '#bd3340' if row.significant_holm else '#24649b'
        ax.errorbar(row.delta, i, xerr=args.nsigma*row.sigma_delta, fmt='o',
                    color=color, capsize=4, ms=6, elinewidth=1.6, zorder=3)
        numbers.text(.02, i, f'{row.delta:+.{args.decimals}f} ± '
                     f'{row.sigma_delta:.{args.decimals}f}    '
                     f'{row.z:+.2f}    {row.p_holm:.3g}',
                     va='center', fontsize=9, color=color, family='monospace')
    ax.axvline(0, color='black', ls='--', lw=1.2)
    ax.set_yticks(range(n), t.galaxy, fontsize=9)
    ax.set_ylim(n-.5, -.5)
    extent = float((abs(t.delta)+args.nsigma*t.sigma_delta).max())*1.15
    ax.set_xlim(-extent, extent)
    ax.set_xlabel(args.xlabel or f'Δ {args.value} = A − B')
    ax.grid(axis='x', alpha=.25)
    ax.set_axisbelow(True)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.spines[['top','right']].set_visible(False)
    numbers.set_xlim(0,1)
    numbers.axis('off')
    numbers.set_title('Δ ± σΔ (1σ)          z       p Holm', fontsize=10, loc='left')
    fig.suptitle(f'{label_a} − {label_b} | {args.value}\n'
                 f'{n} galaxias compartidas · barras ±{args.nsigma:g}σΔ · ρ={args.rho:g}', fontsize=13)
    fig.legend(handles=[
        Line2D([], [], color='#bd3340', marker='o', ls='', label=f'p Holm < {args.alpha:g}'),
        Line2D([], [], color='#24649b', marker='o', ls='', label='No significativo tras Holm')],
        loc='outside lower center', ncol=2, fontsize=9)
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder_a', type=Path)
    p.add_argument('folder_b', type=Path)
    p.add_argument('--value', required=True)
    p.add_argument('--error', required=True)
    p.add_argument('--galaxy-col', default='galaxy')
    p.add_argument('--pattern-a', default='*.csv')
    p.add_argument('--pattern-b', default='*.csv')
    p.add_argument('--label-a')
    p.add_argument('--label-b')
    p.add_argument('--sep', default=None)
    p.add_argument('--rho', type=float, default=0.)
    p.add_argument('--nsigma', type=float, default=1.)
    p.add_argument('--alpha', type=float, default=.05)
    p.add_argument('--xlabel')
    p.add_argument('--sort', choices=['galaxy','delta','significance'], default='galaxy')
    p.add_argument('--decimals', type=int, default=3)
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--out', type=Path, default=Path('delta_resultados'))
    args=p.parse_args()
    if not (-1 < args.rho < 1 and 0 < args.alpha < 1):
        p.error('Se requiere -1<rho<1 y 0<alpha<1')
    if not np.isfinite(args.nsigma) or args.nsigma<=0 or args.dpi<1 or not 0<=args.decimals<=12:
        p.error('nsigma debe ser positivo y finito; dpi positivo; decimals entre 0 y 12')
    a_dir,b_dir=args.folder_a.resolve(),args.folder_b.resolve()
    try:
        fa,fb=choose_csv(a_dir,args.pattern_a),choose_csv(b_dir,args.pattern_b)
        if fa==fb:
            raise ValueError('A y B deben ser archivos distintos')
        a=read_table(fa,args.value,args.error,args.galaxy_col,args.sep)
        b=read_table(fb,args.value,args.error,args.galaxy_col,args.sep)
        t=compare(a,b,args.rho,args.alpha,args.nsigma)
    except (ValueError,pd.errors.ParserError) as exc:
        p.error(str(exc))
    if args.sort=='significance':
        t=t.assign(abs_z=abs(t.z)).sort_values('abs_z',ascending=False,kind='stable')
    else:
        t=t.sort_values(args.sort,kind='stable')
    t=t.reset_index(drop=True)
    missing=pd.DataFrame([{'galaxy':g,'only_in':side}
        for side,names in [('A',set(a.galaxy)-set(b.galaxy)),('B',set(b.galaxy)-set(a.galaxy))]
        for g in sorted(names)],columns=['galaxy','only_in'])
    out=args.out.resolve()
    if out in [a_dir,b_dir]:
        p.error('Usa una carpeta de salida distinta de A y B')
    out.mkdir(parents=True,exist_ok=True)
    t.to_csv(out/'diferencias.csv',index=False)
    missing.to_csv(out/'sin_pareja.csv',index=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(file_A=str(fa),file_B=str(fb),definition='A-B',input_errors='1 sigma',
                  multiple_testing='Holm sobre galaxias compartidas',n_matched=len(t))
    (out/'configuracion.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    fig=plot(t,args,args.label_a or a_dir.name,args.label_b or b_dir.name)
    fig.savefig(out/'delta_galaxias.png',dpi=args.dpi,bbox_inches='tight')
    fig.savefig(out/'delta_galaxias.pdf',bbox_inches='tight')
    plt.close(fig)
    print(t[['galaxy','delta','sigma_delta','z','p_holm','significant_holm']].to_string(index=False))
    print(f'\n{len(t)} galaxias comparadas; {len(missing)} sin pareja. Resultados: {out}')


if __name__=='__main__':
    main()

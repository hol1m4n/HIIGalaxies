#!/usr/bin/env python3
"""Forest plot por galaxia, comparando mediciones de varios CSV.

Dependencias: python -m pip install numpy pandas matplotlib
Ejemplo:
 python forestplot_galaxias.py /ruta/datos --value mu_W --error sigma_W

Cada CSV debe tener galaxy y las columnas elegidas. Cada fila valida es una
medicion independiente en el grafico: duplicados NO se promedian ni suman;
se identifican por numero de fila de datos (primera fila despues del header=1).
La etiqueta incluye la ruta relativa completa del CSV. Los colores identifican
la carpeta contenedora. No se resuelven alias astronomicos; se normalizan solo
espacios exteriores y mayusculas en galaxy.

Errores simetricos: --error sigma_W.
Errores asimetricos: --error-low err_minus --error-high err_plus.
Estos ultimos son magnitudes de error positivas, NO limites del intervalo.
Por defecto se dibuja exactamente el error proporcionado. --error-factor
multiplica las barras, sin atribuirles automaticamente un nivel de confianza.

No se rellenan mediciones ausentes con ceros. Datos numericos invalidos o
faltantes causan un error; --skip-invalid permite omitirlos con reporte.
Columnas ausentes siempre causan error: use --pattern para seleccionar CSV.

Salidas: un PNG por galaxia, indice_graficos.csv, mediciones_utilizadas.csv,
filas_omitidas.csv (si las hay), y opcionalmente forestplots.pdf con --pdf.
No se calcula un promedio combinado ni se asume independencia estadistica.
"""
import argparse
import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_data(paths, root, args):
    rows, invalid = [], []
    for path in paths:
        df = pd.read_csv(path, sep=args.sep, engine='python', dtype=str,
                         keep_default_na=False, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        if df.columns.duplicated().any():
            raise ValueError(f'{path}: encabezados duplicados')
        low = args.error or args.error_low
        high = args.error or args.error_high
        required = {args.galaxy_col, args.value, low, high}
        if not required.issubset(df.columns):
            raise ValueError(f'{path}: faltan columnas {sorted(required-set(df.columns))}')
        values = pd.to_numeric(df[args.value], errors='coerce')
        lows = pd.to_numeric(df[low], errors='coerce')
        highs = pd.to_numeric(df[high], errors='coerce')
        source = path.relative_to(root).as_posix()
        folder = path.parent.relative_to(root).as_posix()
        for i in range(len(df)):
            name = df[args.galaxy_col].iloc[i].strip().upper()
            x, lo, hi = values.iloc[i], lows.iloc[i], highs.iloc[i]
            if not name or not np.isfinite([x, lo, hi]).all() or lo < 0 or hi < 0:
                invalid.append(dict(csv=source, data_row=i+1, galaxy=name,
                                    reason='Nombre vacio, valor/error no finito o error negativo'))
                continue
            rows.append(dict(galaxy=name, csv=source, subfolder=folder,
                             data_row=i+1, value=float(x), error_low=float(lo), error_high=float(hi)))
    if invalid and not args.skip_invalid:
        first = invalid[0]
        raise ValueError(f'{len(invalid)} filas invalidas; primera: {first}. '
                         'Corrige los CSV o utiliza --skip-invalid para omitirlas y registrarlas.')
    return pd.DataFrame(rows), pd.DataFrame(invalid)


def make_plot(galaxy, t, colors, args):
    t = t.sort_values(['csv', 'data_row'] if args.sort=='source' else ['value', 'csv'],
                      kind='stable').reset_index(drop=True)
    n = len(t)
    fig, (ax, text_ax) = plt.subplots(1, 2, figsize=(11, max(3, .43*n+1.8)),
                                    gridspec_kw={'width_ratios': [4, 2]},
                                    sharey=True, layout='constrained')
    labels = []
    repeated = t.csv.duplicated(keep=False)
    for i, row in t.iterrows():
        label = row.csv + (f' [fila {row.data_row}]' if repeated.iloc[i] else '')
        labels.append(label)
        if i % 2 == 0:
            for a in (ax, text_ax):
                a.axhspan(i-.5, i+.5, color='#f1f4f8', zorder=0)
        lo, hi = args.error_factor*row.error_low, args.error_factor*row.error_high
        ax.errorbar(row.value, i, xerr=np.array([[lo], [hi]]), fmt='o', ms=6,
                    color=colors[row.subfolder], capsize=4, elinewidth=1.7, zorder=3)
        f = f'.{args.decimals}f'
        number = (f'{row.value:{f}} ± {lo:{f}}' if args.error else
                  f'{row.value:{f}}  (−{lo:{f}}, +{hi:{f}})')
        text_ax.text(.03, i, number, va='center', fontsize=10)
    if args.reference is not None:
        ax.axvline(args.reference, color='#777777', ls='--', lw=1.2,
                   label=f'Referencia: {args.reference:g}')
        ax.legend(loc='best', fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=9)
    ax.set_ylim(n-.5, -.5)
    ax.set_xlabel(args.xlabel or args.value)
    ax.grid(axis='x', alpha=.25)
    ax.set_axisbelow(True)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.margins(x=.12)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    text_ax.set_xlim(0, 1)
    text_ax.axis('off')
    text_ax.set_title('Medición y error mostrado', fontsize=10, loc='left')
    fig.suptitle(f'{galaxy} — {args.value}\n'
                 f'{n} mediciones · {t.csv.nunique()} CSV · '
                 f'Barras: {args.error_factor:g} × error del CSV', fontsize=13)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', type=Path)
    p.add_argument('--value', required=True, help='Columna de la medicion')
    p.add_argument('--error', help='Columna del error simetrico')
    p.add_argument('--error-low', help='Magnitud del error inferior')
    p.add_argument('--error-high', help='Magnitud del error superior')
    p.add_argument('--galaxy-col', default='galaxy')
    p.add_argument('--pattern', default='**/*.csv')
    p.add_argument('--out', type=Path, default=Path('forestplot_resultados'))
    p.add_argument('--sep', default=None, help='Separador; por defecto autodetectado')
    p.add_argument('--xlabel', help='Etiqueta del eje X, incluyendo unidades')
    p.add_argument('--error-factor', type=float, default=1.)
    p.add_argument('--reference', type=float, help='Linea vertical opcional, igual en todos los plots')
    p.add_argument('--galaxies', nargs='+', help='Opcional: seleccionar solo estas galaxias')
    p.add_argument('--sort', choices=['source', 'value'], default='source')
    p.add_argument('--decimals', type=int, default=3)
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--pdf', action='store_true', help='Guardar tambien PDF multipagina')
    p.add_argument('--skip-invalid', action='store_true')
    args = p.parse_args()
    if bool(args.error) == bool(args.error_low or args.error_high):
        p.error('Usa --error, o bien --error-low y --error-high, sin mezclarlos')
    if not args.error and not (args.error_low and args.error_high):
        p.error('Se requieren ambas columnas de error asimetrico')
    if not np.isfinite(args.error_factor) or args.error_factor <= 0:
        p.error('--error-factor debe ser positivo y finito')
    if args.reference is not None and not np.isfinite(args.reference):
        p.error('--reference debe ser finito')
    if args.dpi < 1 or not 0 <= args.decimals <= 12:
        p.error('dpi debe ser positivo y decimals estar entre 0 y 12')
    root, out = args.root.resolve(), args.out.resolve()
    if not root.is_dir() or root == out:
        p.error('root debe ser una carpeta existente y distinta de out')
    paths = sorted(f for f in root.glob(args.pattern)
                   if f.is_file() and out not in f.resolve().parents)
    if not paths:
        p.error('No se encontraron archivos con ese patron')
    try:
        data, invalid = load_data(paths, root, args)
    except (ValueError, pd.errors.ParserError) as exc:
        p.error(str(exc))
    if data.empty:
        p.error('No hay mediciones validas para graficar')
    if args.galaxies:
        selected = {g.strip().upper() for g in args.galaxies}
        missing = selected-set(data.galaxy)
        if missing:
            print('Sin mediciones validas: '+', '.join(sorted(missing)))
        data = data[data.galaxy.isin(selected)].copy()
    if data.empty:
        p.error('No hay mediciones para las galaxias seleccionadas')
    out.mkdir(parents=True, exist_ok=True)
    data['value_column'] = args.value
    data['error_low_column'] = args.error or args.error_low
    data['error_high_column'] = args.error or args.error_high
    data['error_factor'] = args.error_factor
    data.to_csv(out/'mediciones_utilizadas.csv', index=False)
    # Escribir siempre: evita conservar un reporte viejo tras una nueva ejecucion.
    invalid.reindex(columns=['csv', 'data_row', 'galaxy', 'reason']).to_csv(out/'filas_omitidas.csv', index=False)
    folders = sorted(data.subfolder.unique())
    cmap = plt.get_cmap('tab20')
    colors = {folder: cmap(i % 20) for i, folder in enumerate(folders)}
    index = []
    pdf = PdfPages(out/'forestplots.pdf') if args.pdf else None
    try:
        for galaxy, t in data.groupby('galaxy', sort=True):
            safe = re.sub(r'[^A-Za-z0-9_-]+', '_', galaxy).strip('_')[:70] or 'galaxy'
            digest = hashlib.sha256(galaxy.encode()).hexdigest()[:10]
            filename = f'forest_{safe}_{digest}.png'
            fig = make_plot(galaxy, t, colors, args)
            try:
                fig.savefig(out/filename, dpi=args.dpi, bbox_inches='tight')
                if pdf is not None:
                    pdf.savefig(fig, bbox_inches='tight')
            finally:
                plt.close(fig)
            index.append(dict(galaxy=galaxy, n_measurements=len(t), n_csv=t.csv.nunique(), png=filename))
    finally:
        if pdf is not None:
            pdf.close()
    pd.DataFrame(index).to_csv(out/'indice_graficos.csv', index=False)
    print(f'{len(index)} forest plots generados en {out}')
    print(f'{len(invalid)} filas omitidas. Ver indice_graficos.csv y mediciones_utilizadas.csv.')


if __name__ == '__main__':
    main()

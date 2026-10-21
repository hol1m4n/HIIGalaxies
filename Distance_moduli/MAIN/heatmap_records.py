#!/usr/bin/env python3
"""Heatmap de records por galaxy y CSV/subcarpeta.

Instalar: python -m pip install numpy pandas matplotlib
Uso: python heatmap_records.py /ruta/datos --pattern '**/*.csv'
     python heatmap_records.py /ruta/datos --group-by subfolder

Cada CSV debe contener galaxy y records. Se SUMAN los valores de records,
no se cuentan las filas. Si una galaxia se repite en un CSV, se suman sus
records. Las galaxias ausentes reciben cero; records vacios/invalidos generan
un error para no confundir datos desconocidos con ausencia de registros.
Nombres: se quitan espacios exteriores y se convierten a mayusculas; no se
resuelven alias astronomicos. CSV vacios con encabezados son aceptados.

Por defecto cada columna es un CSV, identificado por su ruta relativa.
--group-by subfolder suma todos los CSV de cada carpeta contenedora exacta
(no mezcla una carpeta con sus descendientes). No deduplica mediciones
compartidas entre archivos. Los totales son sumas, no registros unicos.

Salidas: heatmap_records.png, matriz_por_csv.csv, matriz_por_subcarpeta.csv,
resumen_por_csv.csv, resumen_por_subcarpeta.csv.
Las matrices incluyen la union de galaxias de todos los archivos seleccionados.
Los resumenes distinguen galaxias listadas y galaxias con records > 0.
"""
from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, FuncNorm
from matplotlib.ticker import MaxNLocator


def read_counts(path, sep=None):
    df = pd.read_csv(path, sep=sep, engine='python', dtype=str,
                     keep_default_na=False, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    if not {'galaxy', 'records'}.issubset(df.columns):
        raise ValueError(f'{path}: se requieren las columnas galaxy y records')
    names = df.galaxy.str.strip().str.upper()
    if names.eq('').any():
        raise ValueError(f'{path}: nombre de galaxy vacio')
    numbers = pd.to_numeric(df.records, errors='coerce')
    values = numbers.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values < 0).any()
            or (values != np.floor(values)).any() or (values > 2**53-1).any()):
        raise ValueError(f'{path}: records debe contener enteros no negativos, sin vacios')
    if names.duplicated().any():
        warnings.warn(f'{path}: galaxias repetidas; se sumaran sus records')
    # Python int evita overflow silencioso al sumar conteos.
    series = pd.Series([int(v) for v in values], index=names, dtype=object)
    return series.groupby(level=0).sum().rename_axis('galaxy')


def build_tables(root, paths, sep):
    counts, folders, listed = {}, {}, {}
    for path in paths:
        label = path.relative_to(root).as_posix()
        folders[label] = path.parent.relative_to(root).as_posix()
        counts[label] = read_counts(path, sep)
        listed[label] = set(counts[label].index)
    galaxies = sorted(set().union(*listed.values()))
    matrix = pd.DataFrame({c: s.reindex(galaxies, fill_value=0)
                           for c, s in counts.items()}, dtype=object)
    matrix.index.name = 'galaxy'
    grouped = pd.DataFrame({folder: matrix[[c for c in matrix if folders[c] == folder]].sum(axis=1)
                            for folder in sorted(set(folders.values()))}, index=matrix.index)
    grouped.index.name = 'galaxy'
    per_csv = pd.DataFrame([
        dict(csv=c, subfolder=folders[c], galaxies_listed=len(listed[c]),
             galaxies_with_records=int((matrix[c]>0).sum()), total_records=int(matrix[c].sum()))
        for c in matrix])
    per_folder = pd.DataFrame([
        dict(subfolder=f, n_csv=sum(v==f for v in folders.values()),
             galaxies_listed=len(set().union(*(listed[c] for c in matrix if folders[c]==f))),
             galaxies_with_records=int((grouped[f]>0).sum()), total_records=int(grouped[f].sum()))
        for f in grouped])
    return matrix, grouped, per_csv, per_folder


def plot_heatmap(table, output, dpi=200, log=False, annotate=True):
    nrows, ncols = table.shape
    width, height = max(8, 1.5*ncols+3), max(5, .34*nrows+3.8)
    fig = plt.figure(figsize=(width, height), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, max(1, .34*nrows)],
                           width_ratios=[max(3, 1.5*ncols), .25])
    summary_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])
    cax = fig.add_subplot(grid[1, 1])
    sums = table.sum(axis=0)
    ngal = (table > 0).sum(axis=0)
    # Resumen separado: los totales no alteran la escala del mapa principal.
    summary_ax.imshow(np.zeros((2, ncols)), cmap='Greys', vmin=0, vmax=1, aspect='auto')
    for j in range(ncols):
        summary_ax.text(j, 0, str(int(ngal.iloc[j])), ha='center', va='center')
        summary_ax.text(j, 1, str(int(sums.iloc[j])), ha='center', va='center')
    summary_ax.set_yticks([0, 1], ['Galaxias con records', 'Total records'])
    summary_ax.set_xticks([])
    summary_ax.set_xlim(-.5, ncols-.5)
    data = table.to_numpy(dtype=float)
    vmax = max(1., float(data.max()))
    norm = FuncNorm((np.log1p, np.expm1), vmin=0, vmax=vmax) if log else Normalize(0, vmax)
    cmap = plt.get_cmap('YlGnBu')
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    if annotate:
        for i in range(nrows):
            for j in range(ncols):
                r, g, b, _ = cmap(norm(data[i, j]))
                color = 'white' if .2126*r+.7152*g+.0722*b < .48 else '#17212b'
                ax.text(j, i, str(int(table.iloc[i, j])), ha='center', va='center',
                        fontsize=8, color=color)
    ax.set_xticks(range(ncols), [c.replace('/', '/\n') for c in table.columns],
                  rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(nrows), table.index, fontsize=9)
    ax.set_ylabel('Galaxia')
    ax.set_xlabel('Subcarpeta / CSV' if any(c.endswith('.csv') for c in table.columns)
                  else 'Subcarpeta')
    ax.set_xticks(np.arange(ncols+1)-.5, minor=True)
    ax.set_yticks(np.arange(nrows+1)-.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=.7)
    ax.tick_params(which='minor', bottom=False, left=False)
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_label('Records' + (' (color: log(1 + records))' if log else ''))
    colorbar.locator = MaxNLocator(nbins=6, integer=True)
    colorbar.update_ticks()
    positive = int((table > 0).any(axis=1).sum())
    fig.suptitle(f'Records por galaxia | {nrows} galaxias listadas, {positive} con records\n'
                 f'{ncols} conjuntos | Suma de records: {int(sums.sum())}', fontsize=13)
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', type=Path)
    p.add_argument('--pattern', default='**/*.csv', help='Patron relativo a root')
    p.add_argument('--out', type=Path, default=Path('heatmap_resultados'))
    p.add_argument('--group-by', choices=['csv', 'subfolder'], default='csv')
    p.add_argument('--sep', default=None, help='Separador; por defecto autodetectado')
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--log', action='store_true', help='Color log(1+records), conserva ceros y etiquetas originales')
    p.add_argument('--no-annot', action='store_true', help='Ocultar numeros dentro de celdas')
    p.add_argument('--sort', choices=['name', 'total'], default='name')
    args = p.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if not root.is_dir() or args.dpi < 1:
        p.error('root debe existir y dpi debe ser positivo')
    if root == out:
        p.error('La carpeta de salida debe ser distinta de la carpeta de entrada')
    paths = sorted(f for f in root.glob(args.pattern)
                   if f.is_file() and out not in f.resolve().parents)
    if not paths:
        p.error('No se encontraron CSV con ese patron')
    try:
        matrix, grouped, per_csv, per_folder = build_tables(root, paths, args.sep)
    except (ValueError, pd.errors.ParserError) as exc:
        p.error(str(exc))
    if matrix.empty:
        p.error('Los CSV seleccionados no contienen ninguna galaxia')
    out.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out/'matriz_por_csv.csv')
    grouped.to_csv(out/'matriz_por_subcarpeta.csv')
    per_csv.to_csv(out/'resumen_por_csv.csv', index=False)
    per_folder.to_csv(out/'resumen_por_subcarpeta.csv', index=False)
    selected = matrix if args.group_by == 'csv' else grouped
    if args.sort == 'total':
        selected = selected.loc[selected.sum(axis=1).sort_values(ascending=False, kind='stable').index]
    plot_heatmap(selected, out/'heatmap_records.png', args.dpi, args.log, not args.no_annot)
    print(per_csv.to_string(index=False))
    print(f'\nGalaxias unicas listadas: {len(matrix)}')
    print(f'Resultados: {out}')


if __name__ == '__main__':
    main()

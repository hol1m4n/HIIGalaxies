#!/usr/bin/env python3
"""Tres imagenes por carpeta: histogramas/estadistica y violines de H0 y e_H0.
Entrada: RAIZ/grupo/subgrupo/Results.csv (o Result.csv).

Instalacion: pip install numpy pandas matplotlib scipy
Uso: python analizar_results.py /ruta/a/RAIZ
Opcional: --carpetas A B --bins 20 --salida /ruta/salida --sep ';'

Cada fila del CSV se trata como una realizacion/medicion. No se eliminan
outliers. SEM, IC95 y ponderacion suponen observaciones independientes;
no representan una combinacion valida de ajustes que comparten datos.
e_H0 se interpreta como incertidumbre 1-sigma simetrica, en unidades de H0.
"""
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats


def describir(x):
    """Estadistica de valores finitos; varianza y desviacion muestrales."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    keys = ['n', 'media', 'mediana', 'std', 'varianza', 'sem', 'min', 'max',
            'rango', 'p2.5', 'p16', 'p25', 'p75', 'p84', 'p97.5', 'iqr',
            'mad', 'mad_normal', 'asimetria', 'curtosis_exceso',
            'ic95_media_inf', 'ic95_media_sup', 'outliers_iqr']
    d = dict.fromkeys(keys, np.nan)
    d['n'] = n
    if not n:
        return d
    p = np.percentile(x, [2.5, 16, 25, 75, 84, 97.5])
    mean, median = np.mean(x), np.median(x)
    sd = np.std(x, ddof=1) if n > 1 else np.nan
    sem = sd / np.sqrt(n)
    iqr = p[3] - p[2]
    mad = np.median(np.abs(x - median))
    delta = stats.t.ppf(.975, n - 1) * sem if n > 1 else np.nan
    d.update(media=mean, mediana=median, std=sd, varianza=sd**2, sem=sem,
             min=x.min(), max=x.max(), rango=np.ptp(x),
             iqr=iqr, mad=mad, mad_normal=1.4826022185*mad,
             ic95_media_inf=mean-delta, ic95_media_sup=mean+delta,
             outliers_iqr=int(np.sum((x < p[2]-1.5*iqr) | (x > p[3]+1.5*iqr))))
    d.update(zip(['p2.5', 'p16', 'p25', 'p75', 'p84', 'p97.5'], p))
    if np.ptp(x) > 0:
        if n >= 3:
            d['asimetria'] = stats.skew(x, bias=False)
        if n >= 4:
            d['curtosis_exceso'] = stats.kurtosis(x, fisher=True, bias=False)
    return d


def analizar(path, sep):
    df = pd.read_csv(path, sep=sep, engine='python')
    df.columns = df.columns.str.strip().str.lstrip('\ufeff')
    missing = {'H0', 'e_H0'} - set(df.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    h = pd.to_numeric(df['H0'], errors='coerce').to_numpy(dtype=float)
    e = pd.to_numeric(df['e_H0'], errors='coerce').to_numpy(dtype=float)
    mh = np.isfinite(h)
    me = np.isfinite(e) & (e > 0)
    pair = mh & me
    x, errors = h[mh], e[me]
    a, b = describir(x), describir(errors)
    weighted = dict(n_pares=int(pair.sum()), media_ponderada=np.nan,
                    error_media_ponderada=np.nan, chi2_reducido=np.nan,
                    n_efectivo_pesos=np.nan, pearson_r=np.nan, spearman_r=np.nan)
    if pair.any():
        hp, ep = h[pair], e[pair]
        # Pesos escalados para evitar desbordamiento de 1/e**2.
        weights = (ep.min() / ep)**2
        w = weights / weights.sum()
        mean = np.sum(w * hp)
        weighted.update(media_ponderada=mean,
                        error_media_ponderada=ep.min()/np.sqrt(weights.sum()),
                        n_efectivo_pesos=1/np.sum(w**2))
        if len(hp) > 1:
            weighted['chi2_reducido'] = np.sum(((hp-mean)/ep)**2)/(len(hp)-1)
        if len(hp) >= 3 and np.ptp(hp) > 0 and np.ptp(ep) > 0:
            weighted['pearson_r'] = stats.pearsonr(hp, ep)[0]
            weighted['spearman_r'] = stats.spearmanr(hp, ep)[0]
    return dict(nombre=path.parent.name, h=x, e=errors, a=a, b=b,
                total=len(df), invalid_h=int((~mh).sum()),
                invalid_e=int((~me).sum()), **weighted)


def fmt(v):
    return f'{v:.4g}' if np.isfinite(v) else '—'


def bordes(arrays, bins):
    nonempty = [a for a in arrays if len(a)]
    if not nonempty:
        return np.array([0., 1.])
    values = np.concatenate(nonempty)
    edges = np.histogram_bin_edges(values, bins=bins)
    # Limita los bins automaticos ante outliers extremos; conserva todo el rango.
    if len(edges) > 81 and isinstance(bins, str):
        edges = np.linspace(values.min(), values.max(), 81)
    return edges


def hist(ax, x, edges, color, label, d):
    if len(x):
        ax.hist(x, bins=edges, color=color, alpha=.72, edgecolor='white')
        ax.axvline(d['media'], color='#172554', lw=1.7, label='Media')
        ax.axvline(d['mediana'], color='#9f1239', ls='--', lw=1.7, label='Mediana')
        ax.axvspan(d['p16'], d['p84'], color=color, alpha=.13, label='P16–P84')
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, 'Sin datos válidos', ha='center', transform=ax.transAxes)
    ax.set(xlabel=label, ylabel='Número de registros', xlim=(edges[0], edges[-1]))
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)


def crear_figura(grupo, items, output, bins, dpi):
    eh = bordes([r['h'] for r in items], bins)
    ee = bordes([r['e'] for r in items], bins)
    fig, axes = plt.subplots(len(items), 3, figsize=(18, 6.5*len(items)),
                             squeeze=False, gridspec_kw={'width_ratios': [1, 1, 1.25]})
    colors = plt.get_cmap('tab10')
    for i, r in enumerate(items):
        axh, axe, axt = axes[i]
        color = colors(i % 10)
        hist(axh, r['h'], eh, color, r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]', r['a'])
        hist(axe, r['e'], ee, color, r'$e_{H_0}$ [km s$^{-1}$ Mpc$^{-1}$]', r['b'])
        axh.set_title(r['nombre'], loc='left', weight='bold', fontsize=14)
        axe.set_title('Incertidumbres reportadas', fontsize=12)
        axt.axis('off')
        labels = [('n', 'N válido'), ('media', 'Media'), ('mediana', 'Mediana'),
                  ('std', 'Desv. estándar (n−1)'), ('varianza', 'Varianza'),
                  ('sem', 'SEM*'), ('min', 'Mínimo'), ('max', 'Máximo'),
                  ('rango', 'Rango'), ('p16', 'P16'), ('p84', 'P84'),
                  ('p2.5', 'P2.5'), ('p97.5', 'P97.5'), ('p25', 'Q1'),
                  ('p75', 'Q3'), ('iqr', 'IQR'), ('mad', 'MAD'),
                  ('mad_normal', 'MAD × 1.4826'), ('asimetria', 'Asimetría'),
                  ('curtosis_exceso', 'Exceso de curtosis'),
                  ('outliers_iqr', 'Atípicos (1.5 IQR)')]
        rows = [[label, fmt(r['a'][key]), fmt(r['b'][key])] for key, label in labels]
        rows += [['IC95 media*',
                  f"[{fmt(r['a']['ic95_media_inf'])}, {fmt(r['a']['ic95_media_sup'])}]",
                  f"[{fmt(r['b']['ic95_media_inf'])}, {fmt(r['b']['ic95_media_sup'])}]"],
                 ['Media ponderada ± error*', f"{fmt(r['media_ponderada'])} ± {fmt(r['error_media_ponderada'])}", ''],
                 ['N pares / N efectivo pesos', str(r['n_pares']), fmt(r['n_efectivo_pesos'])],
                 ['χ² reducido*', fmt(r['chi2_reducido']), ''],
                 ['r Pearson / ρ Spearman (H0,e)', fmt(r['pearson_r']), fmt(r['spearman_r'])],
                 ['Excluidos / total filas', f"{r['invalid_h']} / {r['total']}", f"{r['invalid_e']} / {r['total']}"]]
        table = axt.table(cellText=rows, colLabels=['Estadístico', 'H0', 'e_H0'],
                          colWidths=[.46, .29, .25], cellLoc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#e2e8f0')
            if row == 0:
                cell.set_facecolor('#172554')
                cell.set_text_props(color='white', weight='bold')
            elif row % 2 == 0:
                cell.set_facecolor('#f1f5f9')
    fig.suptitle(f'{grupo} · Distribuciones y estadística', fontsize=18, weight='bold')
    fig.text(.5, .006,
             '* SEM e IC95 t de Student: suponen filas independientes; IC95 es aproximado para datos no normales.\n'
             'Ponderación: w = 1/e², error formal = 1/√Σw. χ² evalúa un valor común con errores independientes.\n'
             'Sin recorte de atípicos. H0: valores finitos; e_H0: finitos y positivos. — = no definido.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .055/len(items), 1, 1-.04/len(items)], h_pad=3)
    target = output / f'{grupo}_histogramas_estadistica.png'
    fig.savefig(target, dpi=dpi, facecolor='white')
    plt.close(fig)
    return target


def crear_forest_violines(grupo, items, output, dpi, variable='H0'):
    """Distribuciones sin ponderar; subcarpetas de izquierda a derecha; KDE de Scott.

    Cada violin tiene el mismo ancho maximo y se limita al rango observado.
    Para un unico valor o una muestra constante se dibuja una marca horizontal.
    """
    if variable not in ('H0', 'e_H0'):
        raise ValueError('variable debe ser H0 o e_H0')
    key = 'h' if variable == 'H0' else 'e'
    ylabel = r'$H_0$' if variable == 'H0' else r'$e_{H_0}$'
    fig, ax = plt.subplots(figsize=(max(9, 1.7*len(items)+2), 6.5))
    colors = plt.get_cmap('tab10')
    labels = []
    for i, r in enumerate(items):
        x = r[key]
        color = colors(i % 10)
        labels.append(f"{r['nombre']}\n(N={len(x)})")
        if not len(x):
            ax.text(i, .05, 'Sin datos válidos', transform=ax.get_xaxis_transform(),
                    ha='center', color='#64748b')
            continue
        if len(x) >= 2 and np.ptp(x) > 0:
            parts = ax.violinplot([x], positions=[i], vert=True, widths=.72,
                                  showmeans=False, showmedians=False,
                                  showextrema=False, points=512, bw_method='scott')
            for body in parts['bodies']:
                body.set_alpha(None)
                body.set_facecolor((*color[:3], .65))
                body.set_edgecolor('black')
                body.set_linewidth(1.3)
            grid = np.linspace(x.min(), x.max(), 512)
            density = stats.gaussian_kde(x, bw_method='scott')(grid)
            mode = grid[np.argmax(density)]  # Maximo global de la KDE, aproximado en la malla.
        else:
            ax.plot([i-.25, i+.25], [x[0], x[0]], color='black', lw=2)
            mode = x[0]
        q1, median, q3 = np.percentile(x, [25, 50, 75])
        # Barra central: intervalo intercuartilico, no incertidumbre e_H0.
        ax.vlines(i, q1, q3, color='black', linewidth=6, zorder=3)
        ax.scatter(i, mode, s=95, marker='D', facecolor='#dc2626',
                   edgecolor='black', linewidth=.8, zorder=4)
        ax.scatter(i, median, s=34, marker='o', facecolor='white',
                   edgecolor='black', linewidth=.9, zorder=5)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_xlim(-.5, len(items)-.5)
    ax.set_xlabel('Subcarpeta')
    ax.set_ylabel(ylabel + r' [km s$^{-1}$ Mpc$^{-1}$]', fontsize=12)
    ax.set_title(f'{grupo} · Distribuciones de {variable} por subcarpeta',
                 fontsize=15, weight='bold', pad=18)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.margins(y=.06)
    handles = [
        Line2D([0], [0], color='black', lw=6, label='P25–P75 (IQR)'),
        Line2D([0], [0], ls='', marker='o', mfc='white', mec='black',
               markersize=6, label='Mediana'),
        Line2D([0], [0], ls='', marker='D', mfc='#dc2626', mec='black',
               markersize=7, label='Moda (máximo KDE)'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .10),
               ncol=3, frameon=False)
    fig.text(.5, .025,
             'Ancho: densidad KDE (Scott), normalizada por subcarpeta; moda aproximada en una malla de 512 puntos.\n'
             'Sin ponderación ni recorte de atípicos; muestra constante: marca horizontal.\n'
             'H0: valores finitos; e_H0: valores finitos y positivos.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .19, 1, 1])
    target = output / f'{grupo}_forest_violines_{variable}.png'
    fig.savefig(target, dpi=dpi, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('raiz', type=Path, help='Directorio que contiene A, B, C, ...')
    parser.add_argument('--salida', type=Path, default=None)
    parser.add_argument('--carpetas', nargs='+', help='Opcional: analizar solo A B ...')
    parser.add_argument('--bins', default='fd', help='Número de bins o regla numpy: fd, auto, sturges...')
    parser.add_argument('--sep', default=None, help='Separador CSV; por defecto se detecta')
    parser.add_argument('--dpi', type=int, default=180)
    args = parser.parse_args()
    root = args.raiz.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'No existe el directorio: {root}')
    output = (args.salida.expanduser() if args.salida else root/'analisis_histogramas').resolve()
    output.mkdir(parents=True, exist_ok=True)
    bins = int(args.bins) if args.bins.isdigit() else args.bins
    try:
        np.histogram_bin_edges([0., 1.], bins=bins)
    except (ValueError, TypeError) as exc:
        parser.error(f'--bins inválido: {exc}')
    count = 0
    for group in sorted(root.iterdir()):
        if not group.is_dir() or group.resolve() == output:
            continue
        if args.carpetas and group.name not in args.carpetas:
            continue
        items = []
        for sub in sorted(group.iterdir()):
            if not sub.is_dir():
                continue
            candidates = [p for p in (sub/'Results.csv', sub/'Result.csv') if p.is_file()]
            if not candidates:
                warnings.warn(f'No existe Results.csv ni Result.csv en {sub}; se omite.')
                continue
            path = candidates[0]
            if len(candidates) > 1:
                warnings.warn(f'Existen ambos nombres en {sub}; se utiliza Results.csv.')
            try:
                items.append(analizar(path, args.sep))
            except (ValueError, OSError, pd.errors.ParserError) as exc:
                warnings.warn(f'No se pudo leer {path}: {exc}')
        if not items:
            continue
        image = crear_figura(group.name, items, output, bins, args.dpi)
        violin_image = crear_forest_violines(group.name, items, output, args.dpi, 'H0')
        error_violin_image = crear_forest_violines(group.name, items, output, args.dpi, 'e_H0')
        # Respaldo numérico de TODAS las estadísticas, sin redondearlas a 4 cifras.
        records = []
        for r in items:
            row = {'carpeta': group.name, 'subcarpeta': r['nombre']}
            row.update({f'H0_{k}': v for k, v in r['a'].items()})
            row.update({f'e_H0_{k}': v for k, v in r['b'].items()})
            row.update({k: v for k, v in r.items() if k not in ['nombre', 'h', 'e', 'a', 'b']})
            records.append(row)
        pd.DataFrame(records).to_csv(output/f'{group.name}_estadisticas.csv', index=False)
        print(f'Guardado: {image}')
        print(f'Guardado: {violin_image}')
        print(f'Guardado: {error_violin_image}')
        count += 1
    if not count:
        print('No se encontraron archivos válidos. Estructura esperada: RAIZ/A/TRGB_eR/Results.csv')


if __name__ == '__main__':
    main()

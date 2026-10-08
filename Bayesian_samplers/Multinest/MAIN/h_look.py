"""Lee resultados CodeID_Ancla y genera comparaciones sin cortes de redshift.

Dependencias: numpy, pandas, matplotlib y pymultinest (con MultiNest instalado).
Ejecutar: python h_look.py. Editar primero CONFIGURACION.
No ejecuta el muestreo ni modifica las cadenas existentes.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- CONFIGURACION -----------------------------
DIRECTORIO = Path("/home/hollman/HIIGalaxies/Bayesian_samplers/Multinest/MAIN")
CODE_IDS = ["A.1"]                  # Ejemplo: ["A.1", "A.2", "B.1"]
SUBGRUPOS = None                   # None: todos los encontrados para cada ID.
# SUBGRUPOS = ["TRGB_eR", "PLRC_Z_eR", "Joint_eR"]  # Sin el code ID.
VARIABLE = "h"
PARAMETROS = ["alpha", "beta", "h"] # MISMO orden que en MultiNest.
PREFIJO = None                     # Detectar un unico *stats.dat por carpeta.
# PREFIJO = "{code_id}_"           # Ejemplo: A.1_stats.dat.
MODO = 0                          # Conserva la seleccion del script original.
CONVERTIR_H_A_H0 = True            # h -> H0=100*h; tambien escala su error.
MOSTRAR = False
FIGURAS_POR_CODE_ID = True         # Todas las anclas de un mismo ID.
FIGURAS_POR_SUBGRUPO = False        # Un ancla, comparada entre los IDs elegidos.
SALIDA = DIRECTORIO / "comparisons"
FORMATOS = ("png",) #"pdf")

# Orden visual estable; eR/eT comparten familia de color y marcador.
ESTILOS = {
    "Joint_eR":    {"color": "#8C564B", "marker": "D", "label": "Cefeidas + TRGB"},
    "PLRC_all_eR": {"color": "#17365D", "marker": "o", "label": "Todas las Cefeidas; eR"},
    "PLRC_nZ_eR":  {"color": "#2563A6", "marker": "s", "label": "Cefeidas sin Z; eR"},
    "PLRC_nZ_eT":  {"color": "#79A9D1", "marker": "s", "label": "Cefeidas sin Z; eT"},
    "PLRC_Z_eR":   {"color": "#6A3D9A", "marker": "h", "label": "Cefeidas con Z; eR"},
    "PLRC_Z_eT":   {"color": "#AF8DC3", "marker": "h", "label": "Cefeidas con Z; eT"},
    "TRGB_eR":     {"color": "#D65F2E", "marker": "^", "label": "TRGB; eR"},
}


def resultados_multinest(code_id, variable, parametros, directorio=".",
                        prefijo=None, modo=0, subgrupos=None):
    """Devuelve una fila por carpeta <code_id>_<subgrupo>.

    Se examinan solo las carpetas hijas de directorio. El subgrupo conserva
    TODOS sus guiones bajos. prefijo es relativo a cada carpeta y admite
    {code_id} y {subgrupo}; None exige un unico *stats.dat, incluso en subcarpetas.
    valor/error son media/sigma del modo elegido, en las unidades originales.
    Los fallos de una corrida quedan en estado; no se representan en figuras.
    """
    parametros = list(parametros)
    if variable not in parametros or len(set(parametros)) != len(parametros):
        raise ValueError("VARIABLE debe estar en PARAMETROS, sin nombres duplicados.")
    if not isinstance(modo, int) or modo < 0:
        raise ValueError("modo debe ser un entero >= 0.")
    if not code_id or any(c in code_id for c in "/\\"):
        raise ValueError("code_id debe ser un nombre no vacio, sin rutas.")
    raiz = Path(directorio).expanduser()
    if not raiz.is_dir():
        raise FileNotFoundError(f"No existe el directorio: {raiz}")
    if isinstance(subgrupos, str):
        subgrupos = [subgrupos]
    seleccion = None if subgrupos is None else set(subgrupos)
    inicio = f"{code_id}_"
    carpetas = sorted(p for p in raiz.iterdir()
                      if p.is_dir() and p.name.startswith(inicio)
                      and p.name[len(inicio):]
                      and (seleccion is None or p.name[len(inicio):] in seleccion))
    if not carpetas:
        raise FileNotFoundError(f"No hay carpetas {inicio}* seleccionadas en {raiz}")
    if seleccion is not None:
        faltantes = seleccion - {p.name[len(inicio):] for p in carpetas}
        if faltantes:
            warnings.warn(f"{code_id}: subgrupos sin carpeta: {sorted(faltantes)}")

    # Importacion diferida: importar este script no inicia MultiNest.
    import pymultinest

    filas = []
    indice = parametros.index(variable)
    for carpeta in carpetas:
        subgrupo = carpeta.name[len(inicio):]
        fila = dict(carpeta=carpeta.name, code_id=code_id, subgrupo=subgrupo,
                    variable=variable, valor=np.nan, error=np.nan,
                    modo=modo, n_modos=np.nan, estado="OK")
        try:
            if prefijo is None:
                archivos = sorted(carpeta.rglob("*stats.dat"))
                if not archivos:
                    raise FileNotFoundError("No se encontro *stats.dat.")
                if len(archivos) != 1:
                    raise ValueError("Hay varias corridas: especifica PREFIJO.")
                basename = str(archivos[0])[:-len("stats.dat")]
            else:
                relativo = prefijo.format(code_id=code_id, subgrupo=subgrupo)
                if Path(relativo).is_absolute() or ".." in Path(relativo).parts:
                    raise ValueError("PREFIJO debe ser relativo a la carpeta.")
                basename = str(carpeta) + "/" + relativo
                if not Path(basename + "stats.dat").is_file():
                    raise FileNotFoundError(f"No existe {basename}stats.dat")
            analizador = pymultinest.Analyzer(
                n_params=len(parametros), outputfiles_basename=basename,
                verbose=False,
            )
            modos = analizador.get_mode_stats()["modes"]
            fila["n_modos"] = len(modos)
            if modo >= len(modos):
                raise IndexError(f"modo={modo}; solo hay {len(modos)} modos.")
            if len(modos) > 1:
                warnings.warn(f"{carpeta.name}: {len(modos)} modos; se usa solo {modo}.")
            resultado = modos[modo]
            if any(len(resultado[k]) != len(parametros) for k in ("mean", "sigma")):
                raise ValueError("El numero de parametros no coincide con PARAMETROS.")
            valor, error = (float(resultado[k][indice]) for k in ("mean", "sigma"))
            if not np.isfinite(valor) or not np.isfinite(error) or error < 0:
                raise ValueError("Media/sigma no finitas o sigma negativa.")
            fila.update(valor=valor, error=error)
        except (OSError, ValueError, IndexError, KeyError, TypeError) as exc:
            fila["estado"] = f"{type(exc).__name__}: {exc}"
        filas.append(fila)
    orden = {nombre: i for i, nombre in enumerate(ESTILOS)}
    filas.sort(key=lambda f: (orden.get(f["subgrupo"], len(orden)), f["subgrupo"]))
    return pd.DataFrame(filas)


def _nombre_seguro(texto):
    return re.sub(r"[^\w.\-]+", "_", texto)


def crear_figuras(tabla, salida, convertir_h_a_h0=True, por_code_id=True,
                  por_subgrupo=True, mostrar=True, formatos=("png", "pdf")):
    """Guarda figuras de media +/- sigma; devuelve las rutas creadas.

    No aplica filtros de redshift ni promedia resultados entre carpetas.
    Los limites verticales se comparten entre todas las figuras de esta llamada.
    """
    variables = tabla["variable"].unique()
    if len(variables) != 1:
        raise ValueError("Cada llamada debe representar una sola variable.")
    variable = variables[0]
    validos = tabla.loc[tabla["estado"].eq("OK")].copy()
    if validos.empty:
        warnings.warn("No hay resultados validos para representar.")
        return []
    if validos.duplicated(["code_id", "subgrupo"]).any():
        raise ValueError("Hay resultados duplicados para un mismo code ID y subgrupo.")
    for nombre in validos["subgrupo"].unique():
        if nombre not in ESTILOS:
            warnings.warn(f"Subgrupo {nombre!r} sin estilo: se mostrara en gris.")
    factor = 100.0 if variable == "h" and convertir_h_a_h0 else 1.0
    nombre_variable = "H0" if factor == 100 else variable
    ylabel = (r"$H_0\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$" if nombre_variable == "H0"
              else {"h": "$h$", "alpha": r"$\alpha$", "beta": r"$\beta$"}.get(variable, variable))
    validos["y"] = factor * validos["valor"]
    validos["ey"] = abs(factor) * validos["error"]
    bajo = (validos["y"] - validos["ey"]).min()
    alto = (validos["y"] + validos["ey"]).max()
    margen = max((alto - bajo) * 0.18, abs(alto) * 0.005, 1e-6)
    salida = Path(salida).expanduser()
    salida.mkdir(parents=True, exist_ok=True)
    rutas, figuras = [], []
    trabajos = []
    if por_code_id:
        for code_id, grupo in validos.groupby("code_id", sort=False):
            trabajos.append((grupo, "subgrupo", f"Code ID: {code_id}",
                             f"{code_id}_{nombre_variable}_anclas"))
    if por_subgrupo:
        for subgrupo, grupo in validos.groupby("subgrupo", sort=False):
            ids = "__".join(grupo["code_id"])
            descripcion = ESTILOS.get(subgrupo, {}).get("label", subgrupo)
            trabajos.append((grupo, "code_id", f"{subgrupo} — {descripcion}",
                             f"{ids}_{subgrupo}_{nombre_variable}"))

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#777777", "axes.linewidth": 0.8,
                         "figure.facecolor": "white", "axes.facecolor": "white"}):
        for grupo, eje_x, titulo, nombre in trabajos:
            fig, ax = plt.subplots(figsize=(max(5.5, 1.3 * len(grupo) + 2), 4.7),
                                   layout="constrained")
            for x, (_, fila) in enumerate(grupo.iterrows()):
                estilo = ESTILOS.get(fila["subgrupo"], {"color": "#666666", "marker": "o"})
                ax.errorbar(x, fila["y"], yerr=fila["ey"], fmt=estilo["marker"],
                            color=estilo["color"], ecolor=estilo["color"],
                            markersize=10, markeredgecolor="white", markeredgewidth=0.7,
                            elinewidth=1.8, capsize=5, capthick=1.5, zorder=3)
                ax.annotate(f"{fila['y']:.3f} ± {fila['ey']:.3f}",
                            (x, fila["y"] + fila["ey"]), xytext=(0, 9),
                            textcoords="offset points", ha="center", fontsize=9,
                            color=estilo["color"])
            etiquetas = grupo[eje_x].tolist()
            if eje_x == "subgrupo":
                etiquetas = [s.replace("_", "\n", 1) for s in etiquetas]
            ax.set_xticks(np.arange(len(grupo)), etiquetas)
            ax.set_xlim(-0.6, len(grupo) - 0.4)
            ax.set_ylim(bajo - margen, alto + 1.6 * margen)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_xlabel("Ancla / subgrupo" if eje_x == "subgrupo" else "Code ID", labelpad=10)
            ax.set_title(titulo + "\nMedia ± sigma del modo seleccionado", fontsize=13, pad=14)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E0E0E0", linewidth=0.8)
            ax.tick_params(axis="x", length=0, pad=9)
            for extension in formatos:
                ruta = salida / f"{_nombre_seguro(nombre)}.{extension}"
                fig.savefig(ruta, dpi=300, bbox_inches="tight", facecolor="white")
                rutas.append(ruta)
            figuras.append(fig)
            if not mostrar:
                plt.close(fig)
    if mostrar and figuras:
        plt.show()
        for fig in figuras:
            plt.close(fig)
    return rutas


def main():
    tablas = []
    for code_id in dict.fromkeys(CODE_IDS):
        try:
            tablas.append(resultados_multinest(
                code_id=code_id, variable=VARIABLE, parametros=PARAMETROS,
                directorio=DIRECTORIO, prefijo=PREFIJO, modo=MODO,
                subgrupos=SUBGRUPOS,
            ))
        except FileNotFoundError as exc:
            warnings.warn(str(exc))
    if not tablas:
        raise SystemExit("No se encontraron resultados. Revisa DIRECTORIO y CODE_IDS.")
    tabla = pd.concat(tablas, ignore_index=True)
    if VARIABLE == "h" and CONVERTIR_H_A_H0:
        tabla["H0"] = 100 * tabla["valor"]
        tabla["e_H0"] = 100 * tabla["error"]
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    salida = Path(SALIDA).expanduser()
    salida.mkdir(parents=True, exist_ok=True)
    ids = "__".join(dict.fromkeys(CODE_IDS))
    ruta_tabla = salida / f"{_nombre_seguro(ids)}_{_nombre_seguro(VARIABLE)}_resultados.csv"
    tabla.to_csv(ruta_tabla, index=False)
    rutas = crear_figuras(tabla, salida, CONVERTIR_H_A_H0,
                          FIGURAS_POR_CODE_ID, FIGURAS_POR_SUBGRUPO,
                          MOSTRAR, FORMATOS)
    print(f"\nTabla guardada: {ruta_tabla}")
    for ruta in rutas:
        print(f"Figura guardada: {ruta}")


if __name__ == "__main__":
    main()

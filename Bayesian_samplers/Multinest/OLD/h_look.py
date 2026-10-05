from pathlib import Path
import warnings

import pandas as pd
import pymultinest


def resultados_multinest(
    grupo,
    variable,
    parametros,
    directorio=".",
    prefijo=None,
    modo=0,
):
    """
    Lee carpetas con nombres Grupo_Rango_Subgrupo.

    parametros : nombres en el MISMO orden utilizado en MultiNest.
    prefijo    : prefijo de salida relativo a cada carpeta.
                 Ejemplo: "1-" para archivos "1-stats.dat".
                 None: detectarlo automáticamente.
    modo       : índice del modo cuya media y sigma se extraen.

    Devuelve un DataFrame con los resultados y el estado de cada carpeta.
    """
    parametros = list(parametros)
    if variable not in parametros:
        raise ValueError(
            f"Variable {variable!r} no encontrada en {parametros}."
        )
    if modo < 0:
        raise ValueError("modo debe ser un entero >= 0.")

    indice = parametros.index(variable)
    raiz = Path(directorio).expanduser()
    inicio = f"{grupo}_"

    carpetas = sorted(
        p for p in raiz.iterdir()
        if p.is_dir() and p.name.startswith(inicio)
    )

    if not carpetas:
        raise FileNotFoundError(
            f"No se encontraron carpetas {grupo}_* en {raiz}"
        )

    filas = []

    for carpeta in carpetas:
        # Permite subgrupos que contienen guiones bajos.
        rango, separador, subgrupo = carpeta.name[len(inicio):].partition("_")
        if not separador or rango not in {"Local", "Intermediate", "Full"}:
            continue

        fila = {
            "carpeta": carpeta.name,
            "rango": rango,
            "subgrupo": subgrupo,
            "variable": variable,
            "valor": float("nan"),
            "error": float("nan"),
            "estado": "OK",
        }

        try:
            if prefijo is None:
                archivos = sorted(carpeta.rglob("*stats.dat"))

                if not archivos:
                    raise FileNotFoundError("No se encontró *stats.dat.")

                if len(archivos) > 1:
                    raise ValueError(
                        "Hay varias corridas: especifica prefijo."
                    )

                # Conserva exactamente el prefijo, incluso si está vacío.
                basename = str(archivos[0])[:-len("stats.dat")]
            else:
                basename = str(carpeta) + "/" + prefijo

            analizador = pymultinest.Analyzer(
                n_params=len(parametros),
                outputfiles_basename=basename,
                verbose=False,
            )

            modos = analizador.get_mode_stats()["modes"]

            if modo >= len(modos):
                raise IndexError(
                    f"Se solicitó modo={modo}, pero hay {len(modos)} modos."
                )

            if len(modos) > 1:
                warnings.warn(
                    f"{carpeta.name}: hay {len(modos)} modos; "
                    f"se extraerá únicamente el modo {modo}."
                )

            resultado = modos[modo]

            if len(resultado["mean"]) != len(parametros):
                raise ValueError(
                    "El número de parámetros no coincide con la corrida."
                )

            fila["valor"] = float(resultado["mean"][indice])
            fila["error"] = float(resultado["sigma"][indice])

        except (OSError, ValueError, IndexError, KeyError) as exc:
            fila["estado"] = str(exc)

        filas.append(fila)

    if not filas:
        raise ValueError("No hay carpetas con el patrón Grupo_Rango_Subgrupo.")

    return pd.DataFrame(filas)


tabla = resultados_multinest(
    grupo="MockE",
    variable="h",
    parametros=["alpha", "beta", "h"],  # Orden real de tu corrida
    directorio="/home/holman/HIIGalaxies/Bayesian_samplers/Multinest",
)

print(tabla.to_string(index=False))








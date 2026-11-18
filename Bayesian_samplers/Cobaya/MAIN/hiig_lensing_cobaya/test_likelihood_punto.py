#!/usr/bin/env python3
"""Compara D.py (MultiNest) y LSigmaHIIGLensing (Cobaya) en UN punto fijo.

Evalua los metodos REALES de ambos archivos, extraidos con ast para no
importar pymultinest, getdist, cobaya o ejecutar ningun sampler.
Requiere: numpy, pandas, scipy, astropy (Python >= 3.12).

Uso con datos reales:
  python test_likelihood_punto.py --data Compilation2026_main.csv \
      --anchors TRGB_eR_recal.csv --alpha 33 --beta 5 --H0 70 --Om 0.3

Uso con un conjunto sintetico de prueba:
  python test_likelihood_punto.py --demo
"""
from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatwCDM
from astropy.table import Table
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp


C_KM_S = 299792.458
B_LENS = 2.5 / np.log(10.0)
MU_CONSTANT = 100.19477738511641
GH_ORDER = 20
X_GH, W_GH = hermgauss(GH_ORDER)


def source_class(path: Path, original_name: str, methods: list[str], name: str):
    """Extrae SOLO los metodos originales sin ejecutar imports/samplers del archivo.

    No reproduce/reescribe el likelihood: compila el codigo fuente del metodo.
    Esta extraccion necesita Python >= 3.12 para analizar el f-string de D.py.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    original = next(
        (node for node in tree.body
         if isinstance(node, ast.ClassDef) and node.name == original_name),
        None,
    )
    if original is None:
        raise ValueError(f"No se encontro la clase {original_name} en {path}")
    selected = [
        node for node in original.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in methods
    ]
    missing = set(methods) - {node.name for node in selected}
    if missing:
        raise ValueError(f"Faltan metodos en {path}: {sorted(missing)}")
    new_class = ast.ClassDef(
        name=name, bases=[], keywords=[], body=selected,
        decorator_list=[],
    )
    module_ast = ast.fix_missing_locations(ast.Module(body=[new_class], type_ignores=[]))
    scope = {
        "__file__": str(path), "np": np, "pd": pd, "os": os,
        "Table": Table, "FlatwCDM": FlatwCDM, "u": u, "const": const,
        "logsumexp": logsumexp, "hermgauss": hermgauss,
        "Optional": Optional,
        # Constantes usadas en D.py
        "LENS_GH_X": X_GH, "LENS_GH_W": W_GH,
        "LENS_DM_COEFF": B_LENS,
        # Constantes usadas en el likelihood de Cobaya
        "_LENS_DM_COEFF": B_LENS, "_C_KM_S": C_KM_S,
    }
    exec(compile(module_ast, filename=str(path), mode="exec"), scope)
    return scope[name]


class AstropyProvider:
    """Interfaz de Cobaya con exactamente la misma cosmologia usada por D.py.

    En una prueba de integracion real Cobaya pediria estos valores a CLASS.
    Aqui usamos Astropy para aislar solamente las diferencias de likelihood.
    """
    def __init__(self, H0: float, Om: float):
        self.cosmo = FlatwCDM(
            H0=H0, Om0=Om, w0=-1.0, Tcmb0=2.7255 * u.K
        )

    def get_angular_diameter_distance(self, z):
        return np.asarray(self.cosmo.angular_diameter_distance(z).to_value(u.Mpc))

    def get_Hubble(self, z, units="km/s/Mpc"):
        if units != "km/s/Mpc":
            raise ValueError("Unidad no contemplada en este test")
        return np.asarray(self.cosmo.H(z).to_value(u.km / u.s / u.Mpc))


def make_synthetic_data(directory: Path, params: argparse.Namespace):
    """Caso de demostracion: 3 anclas en 2 anfitrionas + 4 HIIG.

    Se genera exclusivamente para validar el test, NO son datos observados.
    """
    cosmo = FlatwCDM(H0=params.H0, Om0=params.Om, w0=-1.0,
                     Tcmb0=2.7255 * u.K)
    entries = [
        # GEHR: z_or_mu es modulo de distancia placeholder (se reemplaza)
        ("M33", 0, 24.6, 0.10, 1.42, 0.02, 0.05),
        ("M33", 0, 24.6, 0.10, 1.54, 0.02, 0.05),
        ("NGC4258", 0, 29.4, 0.12, 1.68, 0.02, 0.05),
        ("HIIG_A", 1, 0.05, 0.0001, 1.50, 0.025, 0.07),
        ("HIIG_B", 2, 0.80, 0.0005, 1.68, 0.025, 0.07),
        ("HIIG_C", 3, 2.30, 0.0010, 1.73, 0.025, 0.07),
        ("HIIG_D", 4, 5.40, 0.0020, 1.82, 0.025, 0.07),
    ]
    rows = []
    for i, (host, origin, val, err, ls, els, elf) in enumerate(entries):
        true_mu = val if origin == 0 else (5 * np.log10(cosmo.luminosity_distance(val).value) + 25)
        # Residuales no nulos pero pequenos para ejercitar el likelihood.
        residual = [0.12, -0.10, 0.05, 0.2, -0.13, 0.25, -0.2][i]
        logf = params.alpha + params.beta * ls - (true_mu + residual + MU_CONSTANT) / 2.5
        rows.append({
            "GEHR_id": host, "origin_id": origin,
            "z_or_mu": val, "e_z_or_e_mu": err,
            "log_sigma": ls, "e_log_sigma": els,
            "log_f_Hbeta": logf, "e_log_f_Hbeta": elf,
        })
    raw_path = directory / "demo_Lsigma.csv"
    anchors_path = directory / "demo_anchors.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    pd.DataFrame([
        {"galaxy": "M33", "mu_W": 24.6, params.error_column: 0.10},
        {"galaxy": "NGC4258", "mu_W": 29.4, params.error_column: 0.12},
    ]).to_csv(anchors_path, index=False)
    return raw_path, anchors_path


def assert_same_objects(legacy_df: Table, modern):
    """Asegura mismo orden, objetos, modulos de anclas y valores observados."""
    if len(legacy_df) != len(modern._hosts):
        raise AssertionError(
            f"DISTINTAS muestras: D.py n={len(legacy_df)}, Cobaya n={len(modern._hosts)}"
        )
    host_d = np.char.strip(np.asarray(legacy_df["GEHR_id"]).astype(str))

    def clean_to_int(val):
        # Check if the string is 'nan' (case-insensitive) or empty
        if str(val).strip().lower() in ('nan', 'none', ''):
            return ''
        # Convert via float first in case the string is '5.0'
        return str(val)

    modern._hosts = np.array([clean_to_int(x) for x in modern._hosts])


    if not np.array_equal(host_d, modern._hosts):
        print(host_d)
        print(type(host_d))
        print(modern._hosts)
        print(type(modern._hosts))
        raise AssertionError("El orden o los IDs de los objetos son diferentes")
    pairs = [
        ("z_or_mu", modern._z_or_mu),
        ("e_z_or_e_mu", modern._e_z_or_e_mu),
        ("log_sigma", modern._log_sigma),
        ("e_log_sigma", modern._e_log_sigma),
        ("log_f_Hbeta", modern._log_flux),
        ("e_log_f_Hbeta", modern._e_log_flux),
    ]
    origin_d = np.asarray(legacy_df["origin_id"], dtype=float)
    anchor_d = origin_d == 0.0
    if not np.array_equal(anchor_d, modern._is_anchor):
        raise AssertionError("Las mascaras GEHR/HIIG difieren")
    # La columna z_or_mu de Cobaya en GEHR es un placeholder original;
    # D.py la reemplaza por el modulo de la tabla de anclas.
    for col, modern_array in pairs:
        d = np.asarray(legacy_df[col], dtype=float)
        if col == "z_or_mu":
            modern_array = np.asarray(modern_array, dtype=float).copy()
            modern_array[anchor_d] = modern._anchor_mu[anchor_d]
        if col == "e_z_or_e_mu":
            modern_array = np.asarray(modern_array, dtype=float).copy()
            modern_array[anchor_d] = modern._anchor_muerr[anchor_d]
        if not np.allclose(d, modern_array, atol=1e-12, rtol=1e-12):
            idx = np.where(~np.isclose(d, modern_array, atol=1e-12, rtol=1e-12))[0]
            raise AssertionError(f"Datos distintos en '{col}', primeras filas {idx[:8].tolist()}")
    return int(anchor_d.sum()), int((~anchor_d).sum())


def compare(args: argparse.Namespace, raw_path: Path, anchors_path: Path):
    legacy_file = Path(args.legacy_source).expanduser().resolve()
    cobaya_file = Path(args.cobaya_source).expanduser().resolve()
    Legacy = source_class(
        legacy_file, "Lsig_Ho_sampler", ["group_reader", "lens_sigma_eff",
        "lens_lognormal_params", "prepare_lensing_cache", "lensing_loglike_gh",
        "lnlike"], "LegacyLikelihoodOnly"
    )
    Modern = source_class(
        cobaya_file, "LSigmaHIIGLensing", ["initialize", "_resolve_file",
        "_require_columns", "_load_host_covariance", "_prepare_lensing_cache",
        "logp", "_anchor_loglike", "_diag_gaussian_loglike",
        "_lensing_loglike_gh", "_lensing_loglike_analytic"],
        "CobayaLikelihoodOnly"
    )

    # Exactamente el mismo corte y agrupacion que en D.py.
    raw = pd.read_csv(raw_path, comment="#", index_col=False,
                      dtype={"GEHR_id": str})
    if args.zmax is not None:
        origin = np.asarray(raw["origin_id"], dtype=float)
        raw = raw.loc[(origin == 0.0) | (
            (origin != 0.0) & (raw["z_or_mu"] <= args.zmax)
        )].copy()
    legacy = Legacy()  # No llama al constructor original ni ejecuta MultiNest.
    df_legacy = legacy.group_reader(Table.from_pandas(raw),
                                     str(anchors_path), args.error_column)
    cache = legacy.prepare_lensing_cache(df_legacy, zmax=20.0)

    # Inicializa el codigo REAL de Cobaya sin ejecutar MCMC ni importar CLASS.
    modern = Modern()
    cfg = {
        "data_file": str(raw_path), "anchor_file": str(anchors_path),
        "anchor_cov_file": None,
        "data_host_column": "GEHR_id", "origin_column": "origin_id",
        "z_or_mu_column": "z_or_mu", "z_or_mu_error_column": "e_z_or_e_mu",
        "log_sigma_column": "log_sigma", "e_log_sigma_column": "e_log_sigma",
        "log_flux_column": "log_f_Hbeta", "e_log_flux_column": "e_log_f_Hbeta",
        "anchor_host_column": "galaxy", "anchor_mu_column": "mu_W",
        "anchor_error_column": args.error_column,
        "mu_constant": MU_CONSTANT,
        "zmax": args.zmax,
        "include_logdet": False,         # D.py normalized=False
        "correlate_host_mu": False,      # D.py asume anclas independientes
        "sigma_v_pec_kms": 0.0,
        "sigma_sys_mag": 0.0,
        "sigma_sys_column": None,
        "apply_lensing": True,
        "lensing_sigma_coeff": 0.088,
        "lensing_zmin": 0.0,
        "lensing_zmax": np.nextafter(20.0, -np.inf),  # D.py usa z < 20
        "lensing_method": "gh",         # D.py usa Gauss-Hermite
        "gh_order": GH_ORDER,
    }
    for attr, value in cfg.items():
        setattr(modern, attr, value)
    modern.log = logging.getLogger("comparacion_likelihoods")
    modern.provider = AstropyProvider(args.H0, args.Om)
    modern.initialize()

    ngehr, nhiig = assert_same_objects(df_legacy, modern)
    dmask = np.asarray(cache["mask"], dtype=bool)
    c_mask = np.zeros(len(df_legacy), dtype=bool)
    c_mask[modern._is_hiig] = modern._lens_mask_hiig
    if not np.array_equal(dmask, c_mask):
        raise AssertionError("La mascara de lensing difiere entre los codigos")

    theta_d = np.array([args.alpha, args.beta, args.H0 / 100.0,
                        args.Om, -1.0], dtype=float)
    l_d = float(legacy.lnlike(theta=theta_d, DF=df_legacy, lens_cache=cache))
    l_c = float(modern.logp(alpha_ls=args.alpha, beta_ls=args.beta,
                             sigma_int=0.0))

    # Contribucion de las GEHR calculada directamente por el metodo Cobaya.
    mu_ls = 2.5 * (args.alpha + args.beta * modern._log_sigma
                   - modern._log_flux) - MU_CONSTANT
    var_ls = 2.5**2 * (modern._e_log_flux**2 +
                       args.beta**2 * modern._e_log_sigma**2)
    anchor_c = float(modern._anchor_loglike(mu_ls, var_ls))
    hiig_c = l_c - anchor_c

    # Contribuciones separadas de D.py, llamando a su lnlike original.
    a_df = df_legacy[np.asarray(df_legacy["origin_id"], float) == 0]
    h_df = df_legacy[np.asarray(df_legacy["origin_id"], float) != 0]
    anchor_d = float(legacy.lnlike(theta=theta_d, DF=a_df,
                     lens_cache=legacy.prepare_lensing_cache(a_df, zmax=20.0)))
    hiig_d = float(legacy.lnlike(theta=theta_d, DF=h_df,
                   lens_cache=legacy.prepare_lensing_cache(h_df, zmax=20.0)))

    print("\n" + "=" * 78)
    print("PRUEBA DE UN PUNTO: D.py (MultiNest) vs likelihood Cobaya")
    print("=" * 78)
    print(f"alpha={args.alpha:.8f}  beta={args.beta:.8f}  "
          f"H0={args.H0:.8f}  Omega_m={args.Om:.8f}")
    print("Fijos: w_DE=-1, sigma_int=0 dex; GH=20; logdet=OFF; cov_hosts=OFF")
    print(f"Objetos coincidentes: {len(df_legacy)} "
          f"(GEHR={ngehr}, HIIG={nhiig}, con lensing={dmask.sum()})")
    print("Proveedor cosmologico: Astropy FlatwCDM para ambos (NO CLASS)")
    print("-" * 78)
    print(f"{'Contribucion':<19}{'D.py':>20}{'Cobaya':>20}{'Diferencia':>19}")
    for label, a, b in [
        ("GEHR", anchor_d, anchor_c),
        ("HIIG", hiig_d, hiig_c),
        ("TOTAL", l_d, l_c),
    ]:
        print(f"{label:<19}{a:>20.12f}{b:>20.12f}{(a-b):>19.3e}")
    print("-" * 78)
    print(f"Consistencia D.py: (GEHR + HIIG) - TOTAL = {anchor_d + hiig_d - l_d:+.3e}")
    tolerance = float(args.atol)
    good = (np.all(np.isfinite([l_d, l_c, anchor_d, anchor_c, hiig_d, hiig_c]))
            and abs(l_d - l_c) <= tolerance
            and abs(anchor_d - anchor_c) <= tolerance
            and abs(hiig_d - hiig_c) <= tolerance)
    print(f"Resultado: {'PASS - equivalentes' if good else 'FAIL - revisar diferencias'}"
          f" (tolerancia absoluta {tolerance:g} en ln L)")
    return 0 if good else 1


def main():
    ap = argparse.ArgumentParser(description="Prueba de equivalencia en un unico punto")
    ap.add_argument("--data", help="CSV de Compilation2026_main")
    ap.add_argument("--anchors", help="CSV anfitrionas, con galaxy, mu_W, sigma_W")
    ap.add_argument("--legacy-source", default=str(Path(__file__).with_name("D.py")))
    ap.add_argument("--cobaya-source", default=str(
        Path(__file__).with_name("lsigma_hiig_lensing_likelihood.py")))
    ap.add_argument("--alpha", type=float, default=33.0)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--H0", type=float, default=70.0)
    ap.add_argument("--Om", type=float, default=0.30)
    ap.add_argument("--zmax", type=float, default=14.0)
    ap.add_argument("--error-column", default="sigma_W")
    ap.add_argument("--atol", type=float, default=1e-7)
    ap.add_argument("--demo", action="store_true", help="Probar con dataset sintetico")
    args = ap.parse_args()
    if args.demo:
        print("ATENCION: modo DEMO, se emplean datos sinteticos, NO resultados cientificos.")
        with tempfile.TemporaryDirectory() as temp:
            raw, anchors = make_synthetic_data(Path(temp), args)
            return compare(args, raw, anchors)
    if not args.data or not args.anchors:
        ap.error("Debes indicar --data y --anchors, o usar --demo")
    return compare(args, Path(args.data).expanduser().resolve(),
                   Path(args.anchors).expanduser().resolve())


if __name__ == "__main__":
    sys.exit(main())

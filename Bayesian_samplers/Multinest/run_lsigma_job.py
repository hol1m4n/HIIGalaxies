import os
import sys
import pandas as pd
import numpy as np

from astropy.table import Table

import nested_sampler as ns


def parse_zmax(value):
    """
    Convierte el valor de zmax leído desde jobs.csv.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.lower() in ["none", "nan", ""]:
        return None

    return float(value)


def select_redshift_cut(tab, zmax):
    """
    Mantiene siempre la muestra ancla origin_id == 0.
    Para objetos no-ancla, aplica z_or_mu <= zmax.
    Si zmax is None, usa todos los objetos.
    """

    origin = np.asarray(tab["origin_id"], dtype=float)

    if zmax is None:
        mask = np.ones(len(tab), dtype=bool)
    else:
        z_or_mu = np.asarray(tab["z_or_mu"], dtype=float)

        mask = (
            (origin == 0.0)
            |
            ((origin != 0.0) & (z_or_mu <= zmax))
        )

    return tab[mask]


def main():

    if len(sys.argv) != 2:
        raise ValueError(
            "Uso: python3 run_lsigma_job.py JOB_INDEX\n"
            "JOB_INDEX debe ser el índice de fila de jobs.csv, empezando en 0."
        )

    job_index = int(sys.argv[1])

    jobs = pd.read_csv("jobs.csv")

    if job_index < 0 or job_index >= len(jobs):
        raise IndexError(
            f"job_index={job_index} fuera de rango. "
            f"jobs.csv tiene {len(jobs)} filas."
        )

    job = jobs.iloc[job_index]

    print("=" * 80)
    print(f"Running job index: {job_index}")
    print(job)
    print("=" * 80)

    # Lee datos L-sigma
    LSdata_df = pd.read_csv(
        "Compilation2026.csv",
        comment="#",
        index_col=False,
        dtype={"GEHR_id": str},
    )

    LS_tab = Table.from_pandas(LSdata_df)

    zmax = parse_zmax(job["zmax"])
    data_cut = select_redshift_cut(LS_tab, zmax)

    print(f"Total input objects: {len(LS_tab)}")
    print(f"Objects after cut:   {len(data_cut)}")
    print(f"zmax:                {zmax}")

    ns.Lsig_Ho_sampler(
        data_frame=data_cut,
        distance_estimator_set=job["distance_estimator_set"],
        estimator_error_kind=job["estimator_error_kind"],
        main_title=job["main_title"],
        folder_name=job["folder_name"],
        analysis_mode=job["analysis_mode"],
        id_prefix=job["id_prefix"],
    )





    print("=" * 80)
    print(f"Finished job index: {job_index}")
    print("=" * 80)


if __name__ == "__main__":
    main()